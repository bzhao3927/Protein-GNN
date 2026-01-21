# evaluate.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from data_module import ProteinDataModule
from model import EquivariantGNNLightning
from config import Config

sns.set_style("whitegrid")

def load_ensemble(pattern="Equivariant_seed*"):
    """Load all trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = Path(Config.CHECKPOINT_DIR)
    
    models = []
    seeds = []
    
    for ckpt in ckpt_dir.glob(f"{pattern}*.ckpt"):
        if "last" in ckpt.name:
            continue
        
        try:
            seed = int(ckpt.name.split('seed')[1].split('_')[0])
            model = EquivariantGNNLightning.load_from_checkpoint(ckpt)
            model.to(device).eval()
            models.append(model)
            seeds.append(seed)
            print(f"Loaded seed {seed}: {ckpt.name}")
        except Exception as e:
            print(f"Skipping {ckpt.name}: {e}")
    
    return models, seeds

def get_predictions(models, data_loader, device):
    """Get predictions from all models"""
    all_preds = []
    targets = None
    
    with torch.no_grad():
        for model in models:
            preds = []
            targs = []
            for batch in data_loader:
                batch = batch.to(device)
                p = model(batch).cpu().numpy().flatten()
                t = batch.y.cpu().numpy().flatten()
                preds.extend(p)
                targs.extend(t)
            all_preds.append(np.array(preds))
            targets = np.array(targs)
    
    return np.array(all_preds), targets

def calibrate(val_preds, val_targets, test_preds):
    """Isotonic regression calibration"""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_preds, val_targets)
    return calibrator.transform(test_preds)

def compute_r2(preds, targets):
    """Compute R²"""
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("EQUIVARIANT MODEL EVALUATION")
    print("="*70)
    
    # Load models
    models, seeds = load_ensemble()
    print(f"\nLoaded {len(models)} models")
    
    if len(models) == 0:
        print("No models found!")
        return
    
    # Load data
    summary_df = pd.read_csv(Config.SUMMARY_CSV)
    protein_ids = summary_df['target'].unique().tolist()
    data_module = ProteinDataModule(protein_ids)
    data_module.setup()
    
    # Get validation predictions for calibration
    print("\nGetting validation predictions...")
    val_preds, val_targets = get_predictions(models, data_module.val_dataloader(), device)
    val_ensemble = np.mean(val_preds, axis=0)
    
    # Get test predictions
    print("Getting test predictions...")
    test_preds, test_targets = get_predictions(models, data_module.test_dataloader(), device)
    test_ensemble = np.mean(test_preds, axis=0)
    
    # Calibrate
    print("Applying calibration...")
    test_calibrated = calibrate(val_ensemble, val_targets, test_ensemble)
    
    # Compute metrics
    individual_r2s = [compute_r2(test_preds[i], test_targets) for i in range(len(models))]
    ensemble_r2 = compute_r2(test_ensemble, test_targets)
    calibrated_r2 = compute_r2(test_calibrated, test_targets)
    
    ensemble_mae = np.mean(np.abs(test_ensemble - test_targets))
    calibrated_mae = np.mean(np.abs(test_calibrated - test_targets))
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Test samples: {len(test_targets)}")
    print(f"RBF resolution: {Config.NUM_RBF} gaussians over {Config.RBF_MIN}-{Config.RBF_MAX}Å")
    print(f"Ångström resolution: ~{(Config.RBF_MAX - Config.RBF_MIN) / Config.NUM_RBF:.2f}Å per gaussian")
    print("-"*70)
    
    print("\nIndividual Models:")
    for seed, r2 in zip(seeds, individual_r2s):
        print(f"  Seed {seed:4d}: R² = {r2:.4f}")
    
    print(f"\nEnsemble (mean):       R² = {ensemble_r2:.4f}, MAE = {ensemble_mae:.4f}")
    print(f"Ensemble (calibrated): R² = {calibrated_r2:.4f}, MAE = {calibrated_mae:.4f}")
    print("="*70)
    
    if calibrated_r2 >= 0.95:
        print("\n🎯🎯🎯 95% TARGET ACHIEVED! 🎯🎯🎯")
    elif calibrated_r2 >= 0.93:
        print(f"\n📈 {calibrated_r2:.1%} - Very close to 95% target!")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw ensemble
    ax = axes[0]
    errors = np.abs(test_targets - test_ensemble)
    scatter = ax.scatter(test_targets, test_ensemble, alpha=0.6, s=30,
                        c=errors, cmap='RdYlGn_r', edgecolors='black', linewidth=0.3)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True lDDT', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted lDDT', fontsize=12, fontweight='bold')
    ax.set_title(f'Ensemble (Raw)\nR² = {ensemble_r2:.4f}, MAE = {ensemble_mae:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Error')
    
    # Calibrated
    ax = axes[1]
    errors = np.abs(test_targets - test_calibrated)
    scatter = ax.scatter(test_targets, test_calibrated, alpha=0.6, s=30,
                        c=errors, cmap='RdYlGn_r', edgecolors='black', linewidth=0.3)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True lDDT', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted lDDT', fontsize=12, fontweight='bold')
    ax.set_title(f'Ensemble (Calibrated)\nR² = {calibrated_r2:.4f}, MAE = {calibrated_mae:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Error')
    
    plt.tight_layout()
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "equivariant_results.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_dir / 'equivariant_results.png'}")
    
    plt.close()

if __name__ == "__main__":
    evaluate()