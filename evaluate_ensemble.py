# evaluate_ensemble.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
from config import Config
from data_module import ProteinDataModule
from model import HybridGNNLightning

sns.set_style("whitegrid")

def load_ensemble_models():
    """Load all 5 models"""
    seeds = [42, 123, 456, 789, 1011]
    models = []
    
    for seed in seeds:
        pattern = f"ensemble_seed{seed}_*.ckpt"
        ckpts = list(Config.CHECKPOINT_DIR.glob(pattern))
        
        if not ckpts:
            print(f"Warning: No checkpoint found for seed {seed}")
            continue
            
        best_ckpt = min(ckpts, key=lambda x: float(x.stem.split('val_loss=')[1]))
        print(f"Loading seed {seed}: {best_ckpt.name}")
        
        model = HybridGNNLightning.load_from_checkpoint(
            best_ckpt,
            node_input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            num_heads=Config.NUM_ATTENTION_HEADS,
            num_rbf=Config.NUM_RBF
        )
        model.eval()
        models.append(model)
    
    return models

def evaluate_ensemble():
    # Initialize W&B
    wandb.init(
        project="protein-quality-gnn",
        name="eval-ensemble-final",
        job_type="evaluation"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load all models
    print("\n" + "="*60)
    print("LOADING ENSEMBLE MODELS")
    print("="*60)
    models = load_ensemble_models()
    print(f"\nLoaded {len(models)} models")
    
    for model in models:
        model.to(device)
    
    # Load test data
    summary_df = pd.read_csv(Config.SUMMARY_CSV)
    protein_ids = summary_df['target'].unique().tolist()
    
    data_module = ProteinDataModule(protein_ids=protein_ids)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Output directory
    output_dir = Path("evaluation_results_ensemble")
    output_dir.mkdir(exist_ok=True)
    
    # Get individual model predictions
    print("\nEvaluating individual models...")
    individual_results = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            model_preds = []
            model_targets = []
            
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).cpu().numpy()
                model_preds.extend(pred.flatten())
                model_targets.extend(batch.y.cpu().numpy().flatten())
            
            model_preds = np.array(model_preds)
            model_targets = np.array(model_targets)
            
            model_r2 = 1 - (np.sum((model_targets - model_preds) ** 2) / 
                           np.sum((model_targets - np.mean(model_targets)) ** 2))
            model_mae = np.mean(np.abs(model_preds - model_targets))
            
            individual_results.append({
                'preds': model_preds,
                'r2': model_r2,
                'mae': model_mae
            })
            
            print(f"  Model {i+1} (seed {[42, 123, 456, 789, 1011][i]}): R²={model_r2:.4f}, MAE={model_mae:.4f}")
    
    # Ensemble predictions
    print("\nComputing ensemble predictions...")
    all_preds_array = np.array([res['preds'] for res in individual_results])
    ensemble_preds = np.mean(all_preds_array, axis=0)
    targets = model_targets
    
    # Compute ensemble metrics
    r2 = 1 - (np.sum((targets - ensemble_preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    mse = np.mean((ensemble_preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ensemble_preds - targets))
    
    # Performance scatter plot
    print("\nGenerating performance plots...")
    fig_scatter = plt.figure(figsize=(12, 12))
    errors = np.abs(targets - ensemble_preds)
    scatter = plt.scatter(targets, ensemble_preds, alpha=0.6, s=40, 
                         c=errors, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=3, label='Perfect Prediction', alpha=0.8)
    plt.xlabel('True lDDT', fontsize=16, fontweight='bold')
    plt.ylabel('Ensemble Predicted lDDT', fontsize=16, fontweight='bold')
    plt.title(f'Ensemble Performance ({len(models)} Models)\nR²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}',
              fontsize=18, fontweight='bold')
    plt.colorbar(scatter, label='Absolute Error')
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Error distribution
    fig_error = plt.figure(figsize=(12, 7))
    errors_signed = ensemble_preds - targets
    plt.hist(errors_signed, bins=60, alpha=0.7, edgecolor='black', color='steelblue')
    plt.axvline(0, color='red', linestyle='--', linewidth=3, label='Zero Error')
    plt.xlabel('Prediction Error (Predicted - True)', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.title('Ensemble Error Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Individual vs Ensemble comparison plot
    fig_comparison = plt.figure(figsize=(14, 8))
    
    # Bar chart of R² scores
    ax1 = plt.subplot(1, 2, 1)
    seeds = [42, 123, 456, 789, 1011]
    r2_scores = [res['r2'] for res in individual_results]
    colors_bar = ['steelblue'] * len(seeds) + ['darkgreen']
    
    x_pos = list(range(len(seeds))) + [len(seeds) + 0.5]
    heights = r2_scores + [r2]
    labels = [f'Seed {s}' for s in seeds] + ['Ensemble']
    
    bars = ax1.bar(x_pos, heights, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
    ax1.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax1.set_title('Model Comparison', fontsize=16, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    ax2 = plt.subplot(1, 2, 2)
    mae_scores = [res['mae'] for res in individual_results]
    heights_mae = mae_scores + [mae]
    
    bars_mae = ax2.bar(x_pos, heights_mae, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('MAE (Mean Absolute Error)', fontsize=14, fontweight='bold')
    ax2.set_title('Error Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars_mae:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Log to W&B
    wandb.log({
        "ensemble/r2_score": r2,
        "ensemble/mse": mse,
        "ensemble/rmse": rmse,
        "ensemble/mae": mae,
        "ensemble/num_models": len(models),
        "visuals/performance_scatter": wandb.Image(fig_scatter),
        "visuals/error_distribution": wandb.Image(fig_error),
        "visuals/model_comparison": wandb.Image(fig_comparison)
    })
    
    # Log individual model results
    for i, res in enumerate(individual_results):
        seed = [42, 123, 456, 789, 1011][i]
        wandb.log({
            f"individual/seed_{seed}_r2": res['r2'],
            f"individual/seed_{seed}_mae": res['mae']
        })
    
    # Save figures
    fig_scatter.savefig(output_dir / "ensemble_performance.png", dpi=150, bbox_inches='tight')
    fig_error.savefig(output_dir / "ensemble_errors.png", dpi=150, bbox_inches='tight')
    fig_comparison.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    
    # Print results
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)
    print(f"Number of models: {len(models)}")
    print(f"Test samples:     {len(targets)}")
    print("-"*70)
    print("\nIndividual Models:")
    for i, res in enumerate(individual_results):
        seed = [42, 123, 456, 789, 1011][i]
        print(f"  Seed {seed:4d}: R²={res['r2']:.4f}, MAE={res['mae']:.4f}")
    
    print("\nEnsemble (Average of All Models):")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MSE:       {mse:.6f}")
    print(f"  RMSE:      {rmse:.6f}")
    print(f"  MAE:       {mae:.6f}")
    print("="*70)
    
    best_individual_r2 = max(res['r2'] for res in individual_results)
    improvement = r2 - best_individual_r2
    print(f"\nComparison to best individual model:")
    print(f"  Best individual R²: {best_individual_r2:.4f}")
    print(f"  Ensemble R²:        {r2:.4f}")
    print(f"  Difference:         {improvement:+.4f}")
    
    if r2 >= 0.90:
        print("\n🎉🎉🎉 SUCCESS! Ensemble achieved 90%+ R² 🎉🎉🎉")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"W&B Run: {wandb.run.url}")
    print("="*70)
    
    plt.close('all')
    wandb.finish()

if __name__ == "__main__":
    evaluate_ensemble()