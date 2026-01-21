# config.py
import torch
from pathlib import Path

class Config:
    """Equivariant GNN with fine Ångström resolution"""
    # ==================== Data Paths ====================
    RAW_DATA_DIR = Path("/data/PSBench_data/CASP15_community_dataset")
    PDB_DIR = RAW_DATA_DIR / "Predicted_Models"
    QUALITY_SCORES_DIR = RAW_DATA_DIR / "Quality_Scores"
    FASTA_DIR = RAW_DATA_DIR / "Fasta"
    SUMMARY_CSV = RAW_DATA_DIR / "CASP15_community_dataset_summary.csv"
    
    PROJECT_DIR = Path("/home/bzhao/ben_protein_gnn")
    CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
    LOG_DIR = PROJECT_DIR / "logs"
    PROCESSED_DIR = PROJECT_DIR / "processed_graphs"
    
    # ==================== Model - EQUIVARIANT ====================
    INPUT_DIM = 20              # Only amino acid features (not coords!)
    HIDDEN_DIM = 384
    NUM_LAYERS = 6
    NUM_RBF = 100               # 80 → 100 for finer Ångström resolution
    NUM_ATTENTION_HEADS = 12
    OUTPUT_DIM = 1          
    DROPOUT = 0.15
    K_NEIGHBORS = 20            # More neighbors for better geometry
    
    # RBF parameters for fine-grained distance encoding
    RBF_MIN = 0.0               # 0 Ångströms
    RBF_MAX = 30.0              # 30 Ångströms (increased from 20-25)
    
    # ==================== Training ====================
    BATCH_SIZE = 20
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 8e-5
    NUM_EPOCHS = 200
    PATIENCE = 30
    GRADIENT_CLIP_VAL = 0.5
        
    TARGET_SCORE = 'lddt'
    
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # ==================== System ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    RANDOM_SEED = 42
    
    # ==================== W&B ====================
    WANDB_PROJECT = "protein-quality-gnn"
    WANDB_ENTITY = "bzhao-hamilton-college"
    
    @classmethod
    def get_run_name(cls):
        return f"Equivariant_h{cls.HIDDEN_DIM}_L{cls.NUM_LAYERS}_rbf{cls.NUM_RBF}"