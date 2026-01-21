# train_ensemble.py
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from config import Config
from data_module import ProteinDataModule
from model import EquivariantGNNLightning

def train_one_model(seed):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(seed)
    
    summary_df = pd.read_csv(Config.SUMMARY_CSV)
    protein_ids = summary_df['target'].unique().tolist()
    data_module = ProteinDataModule(protein_ids)
    
    model = EquivariantGNNLightning(
        node_input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        num_rbf=Config.NUM_RBF,
        dropout=Config.DROPOUT,
        lr=Config.LEARNING_RATE
    )
    
    wandb_logger = WandbLogger(
        project=Config.WANDB_PROJECT,
        entity=Config.WANDB_ENTITY,
        name=f"Equivariant_seed{seed}_{Config.get_run_name()}"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename=f"Equivariant_seed{seed}_{{epoch:02d}}_{{val_loss:.4f}}",
        monitor='val_loss',
        mode='min',
        save_top_k=2
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.PATIENCE,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=Config.NUM_EPOCHS,
        accelerator='auto',
        devices=1,
        gradient_clip_val=Config.GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=wandb_logger,
        enable_progress_bar=True
    )
    
    try:
        trainer.fit(model, data_module)
        trainer.test(model, data_module, ckpt_path='best')
        print(f"\n✓ Seed {seed} complete. Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    except Exception as e:
        print(f"\n✗ Seed {seed} error: {e}")
    finally:
        import wandb
        wandb.finish()

def main():
    seeds = [42, 123, 456, 789, 1011]
    
    print("\n" + "="*70)
    print("EQUIVARIANT ENSEMBLE TRAINING")
    print("="*70)
    print(f"Seeds: {seeds}")
    print(f"RBF resolution: {Config.NUM_RBF} gaussians over {Config.RBF_MIN}-{Config.RBF_MAX}Å")
    print(f"Ångström resolution: ~{(Config.RBF_MAX - Config.RBF_MIN) / Config.NUM_RBF:.2f}Å")
    print("="*70 + "\n")
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training model with seed {seed}")
        print(f"{'='*60}\n")
        train_one_model(seed)
    
    print("\n✅ All ensemble models trained!")

if __name__ == "__main__":
    main()