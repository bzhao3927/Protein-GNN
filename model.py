# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, BatchNorm
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanAbsoluteError

from config import Config


class RBFExpansion(nn.Module):
    """
    High-resolution RBF for fine-grained Ångström encoding
    Maps scalar distances to 100D vectors with smooth overlap
    """
    def __init__(self, v_min=0.0, v_max=30.0, num_gaussians=100):
        super().__init__()
        self.register_buffer('centers', torch.linspace(v_min, v_max, num_gaussians))
        # Spacing between centers
        spacing = (v_max - v_min) / (num_gaussians - 1)
        self.gamma = 1.0 / (spacing ** 2 + 1e-8)

    def forward(self, dists):
        """
        Input: (E, 1) distances in Ångströms
        Output: (E, num_gaussians) RBF features
        
        With 100 gaussians over 0-30Å, resolution = 0.3Å per gaussian
        This captures sub-Ångström differences in geometry!
        """
        return torch.exp(-self.gamma * (dists - self.centers).pow(2))


class EquivariantProteinGNN(nn.Module):
    """
    Truly equivariant protein structure GNN:
    - Rotation invariant: uses only distances, not coordinates
    - Translation invariant: centers coordinates
    - Fine-grained: 100 RBF gaussians capture 0.3Å resolution
    """
    def __init__(self, node_input_dim=20, hidden_dim=384, output_dim=1, 
                 num_layers=6, num_heads=12, num_rbf=100, dropout=0.15):
        super().__init__()
        
        # Embed amino acid types (equivariant: independent of rotation)
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # High-resolution RBF for capturing fine geometric details
        self.rbf = RBFExpansion(
            v_min=Config.RBF_MIN,
            v_max=Config.RBF_MAX,
            num_gaussians=num_rbf
        )
        
        # Edge encoder: RBF features → hidden dim
        # This is where geometric information enters the model
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # GAT layers with edge features
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, 
                    hidden_dim // num_heads, 
                    heads=num_heads, 
                    concat=True,
                    edge_dim=hidden_dim,  # Rich geometric edge features
                    dropout=dropout
                )
            )
            self.bns.append(BatchNorm(hidden_dim))
        
        # Multi-scale pooling (captures both local and global structure)
        self.pool_combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        
        # Embed amino acid features (equivariant)
        h = self.node_embedding(x)
        
        # Compute pairwise distances (equivariant: rotation invariant!)
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        
        # Encode distances with high-resolution RBF
        # This captures fine geometric details at 0.3Å resolution
        edge_rbf = self.rbf(dist)  # (E, 100)
        edge_attr = self.edge_encoder(edge_rbf)  # (E, hidden_dim)
        
        # Message passing with residual connections
        for conv, bn in zip(self.convs, self.bns):
            identity = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = bn(h)
            h = F.silu(h)
            h = h + identity  # Residual
            h = F.dropout(h, p=0.1, training=self.training)
        
        # Multi-scale global pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_global = torch.cat([h_mean, h_max], dim=-1)
        h_global = self.pool_combine(h_global)
        
        # Predict quality score
        return self.head(h_global).squeeze(-1)


class EquivariantGNNLightning(pl.LightningModule):
    """Lightning wrapper for equivariant model"""
    def __init__(self, node_input_dim=20, hidden_dim=384, output_dim=1,
                 num_layers=6, num_heads=12, num_rbf=100, dropout=0.15, lr=0.0005):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = EquivariantProteinGNN(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            dropout=dropout
        )
        
        self.criterion = nn.MSELoss()
        
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y.squeeze()
        loss = self.criterion(preds, targets)
        
        self.train_r2(preds, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, batch_size=len(targets))
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, 
                 prog_bar=True, batch_size=len(targets))
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y.squeeze()
        loss = self.criterion(preds, targets)
        
        self.val_r2(preds, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, batch_size=len(targets))
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True, 
                 prog_bar=True, batch_size=len(targets))
        return loss

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y.squeeze()
        loss = self.criterion(preds, targets)
        
        self.test_r2(preds, targets)
        self.test_mae(preds, targets)
        
        self.log('test_loss', loss, batch_size=len(targets))
        self.log('test_r2', self.test_r2, batch_size=len(targets))
        self.log('test_mae', self.test_mae, batch_size=len(targets))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=8e-5
        )
        
        # Cosine annealing with warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=10
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=190, eta_min=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[10]
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}