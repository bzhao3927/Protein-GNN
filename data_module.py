# data_module.py
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from config import Config

AMINO_ACIDS = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
}

def parse_pdb_ca_atoms(pdb_path):
    """Extract CA atom coordinates AND amino acid types from PDB file"""
    coords = []
    residues = []
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    if atom_name == 'CA':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                        residue_name = line[17:20].strip()
                        residues.append(residue_name)
    except Exception:
        return np.array([]), []
    
    return np.array(coords), residues

def residue_to_onehot(residue_name):
    """Convert amino acid 3-letter code to one-hot vector"""
    onehot = np.zeros(20, dtype=np.float32)
    if residue_name in AMINO_ACIDS:
        onehot[AMINO_ACIDS[residue_name]] = 1.0
    else:
        onehot[:] = 0.05  # Uniform for unknown
    return onehot

def create_equivariant_graph(pdb_path, target_score):
    """
    Create TRULY EQUIVARIANT graph:
    - Node features = ONLY amino acid type (rotation invariant)
    - pos = 3D coordinates (used ONLY for computing distances)
    - Edge features computed from distances at runtime
    
    This ensures the model is invariant to rotations and translations!
    """
    coords, residues = parse_pdb_ca_atoms(pdb_path)
    
    if len(coords) < 20:
        return None
    
    coords = np.array(coords, dtype=np.float32)
    
    # Center coordinates (translation invariance)
    coords_centered = coords - coords.mean(axis=0)
    
    # Node features = ONLY amino acid identity
    # ✓ This is naturally equivariant (doesn't depend on orientation)
    aa_features = np.array([residue_to_onehot(r) for r in residues], dtype=np.float32)
    
    # Build KNN graph from 3D positions
    k = min(Config.K_NEIGHBORS, len(coords) - 1)
    adj = kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    edge_index = np.array(adj.nonzero())
    
    return Data(
        x=torch.FloatTensor(aa_features),           # 20D amino acid features
        pos=torch.FloatTensor(coords_centered),     # 3D coords for distance computation
        edge_index=torch.LongTensor(edge_index),
        y=torch.FloatTensor([target_score])
    )


class ProteinDataset(Dataset):
    """Dataset with equivariant graph construction"""
    def __init__(self, protein_ids):
        super().__init__()
        self.data_list = []
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading EQUIVARIANT dataset (target: {Config.TARGET_SCORE})...")
        print(f"RBF resolution: {Config.NUM_RBF} gaussians over {Config.RBF_MIN}-{Config.RBF_MAX} Å")
        
        for p_id in tqdm(protein_ids, desc="Processing proteins"):
            scores_file = Config.QUALITY_SCORES_DIR / f"{p_id}_quality_scores.csv"
            pdb_folder = Config.PDB_DIR / p_id
            
            if not scores_file.exists() or not pdb_folder.exists():
                continue
                
            df = pd.read_csv(scores_file)
            for _, row in df.iterrows():
                pdb_name = row['model_name']
                pdb_path = pdb_folder / pdb_name
                
                if pdb_path.exists():
                    data = create_equivariant_graph(str(pdb_path), float(row[Config.TARGET_SCORE]))
                    if data is not None:
                        self.data_list.append(data)
        
        print(f"Loaded {len(self.data_list)} protein structures")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class ProteinDataModule(pl.LightningDataModule):
    """Encapsulates all data logic"""
    def __init__(self, protein_ids):
        super().__init__()
        self.protein_ids = protein_ids
        self.dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = ProteinDataset(self.protein_ids)
        
        total = len(self.dataset)
        train_len = int(Config.TRAIN_SPLIT * total)
        val_len = int(Config.VAL_SPLIT * total)
        test_len = total - train_len - val_len
        
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            self.dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
        )
        
        print(f"Dataset splits - Train: {train_len}, Val: {val_len}, Test: {test_len}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=Config.BATCH_SIZE, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=Config.BATCH_SIZE, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )