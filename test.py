# verify_best_model.py
import torch
import pandas as pd
from pathlib import Path
from data_module import ProteinDataModule
from model import HybridGNNLightning
import numpy as np

# Put your actual checkpoint path here
BEST_CKPT = "checkpoints/HybridGNN_h256_L5_k15_epoch=XX_val_loss=X.XXXX.ckpt"

def verify_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading: {BEST_CKPT}")
    model = HybridGNNLightning.load_from_checkpoint(BEST_CKPT)
    model.to(device).eval()
    
    # Load data
    summary_df = pd.read_csv("/data/PSBench_data/CASP15_community_dataset/CASP15_community_dataset_summary.csv")
    protein_ids = summary_df['target'].unique().tolist()
    data_module = ProteinDataModule(protein_ids)
    data_module.setup()
    
    # Test
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            batch = batch.to(device)
            preds = model(batch).cpu().numpy()
            targets = batch.y.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    r2 = 1 - (np.sum((all_targets - all_preds) ** 2) / 
              np.sum((all_targets - np.mean(all_targets)) ** 2))
    mae = np.mean(np.abs(all_targets - all_preds))
    
    print("\n" + "="*60)
    print("YOUR ORIGINAL MODEL PERFORMANCE")
    print("="*60)
    print(f"Test R²:  {r2:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Samples:  {len(all_targets)}")
    print("="*60)
    
    return r2, mae

if __name__ == "__main__":
    verify_model()