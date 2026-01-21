# Equivariant GNN for Protein Structure Quality Prediction

Rotation-invariant Graph Attention Network achieving **96.8% R²** on CASP15 dataset (8,640+ structures).

## Key Innovation

**True equivariance** through geometric encoding:
- Node features: Only amino acid identity (20D)
- Edge features: Distance-based with 100 RBF gaussians (0.3Å resolution)
- Result: Same protein → same prediction regardless of orientation

## Architecture

- 6-layer Graph Attention Network (GATv2)
- 384 hidden dimensions, 12 attention heads
- Multi-scale pooling (mean + max)
- 3.4M parameters

## Results

| Split | R² | MAE |
|-------|------|------|
| Train | 99.1% | - |
| Val | 94.5% | - |
| **Test** | **96.8%** | **0.016** |

## Installation
```bash
conda create -n protein_gnn python=3.10
conda activate protein_gnn

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric pytorch-lightning torchmetrics wandb scikit-learn matplotlib seaborn
```

## Usage
```bash
# Train single model
python train.py

# Train ensemble (5 seeds)
python train_ensemble.py

# Evaluate
python evaluate.py
```

## Project Structure
```
├── config.py           # Hyperparameters
├── data_module.py      # Data loading & graph construction
├── model.py            # Equivariant GNN architecture
├── train.py            # Training script
├── evaluate.py         # Evaluation & visualization
└── checkpoints/        # Saved models
```

## Configuration

Edit `config.py`:
```python
HIDDEN_DIM = 384
NUM_LAYERS = 6
NUM_RBF = 100              # 0.3Å resolution
K_NEIGHBORS = 20
BATCH_SIZE = 20
LEARNING_RATE = 0.0005
```

## Why It Works

**Baseline (92% R²):** Coordinates in node features → breaks equivariance  
**This approach (96.8% R²):** Distance-only encoding → learns fundamental geometric relationships

Protein quality depends on **relative geometry**, not absolute positions.

## Monitoring

W&B project: [protein-quality-gnn](https://wandb.ai/bzhao-hamilton-college/protein-quality-gnn)

## Requirements

- PyTorch 2.0+
- PyTorch Geometric 2.3+
- PyTorch Lightning 2.0+
- CASP15 dataset

## Contact

Benjamin Zhao  
bzhao@hamilton.edu | [GitHub](https://github.com/benz3927)

## License

MIT
