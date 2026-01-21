# train.py
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from config import Config
from data_module import ProteinDataModule
from model import EquivariantGNNLightning

def main():
    # Optimize for H200
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(Config.RANDOM_SEED)
    
    # Load data
    summary_df = pd.read_csv(Config.SUMMARY_CSV)
    protein_ids = summary_df['target'].unique().tolist()
    data_module = ProteinDataModule(protein_ids)
    
    # Model
    model = EquivariantGNNLightning(
        node_input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        num_rbf=Config.NUM_RBF,
        dropout=Config.DROPOUT,
        lr=Config.LEARNING_RATE
    )
    
    # W&B logger
    wandb_logger = WandbLogger(
        project=Config.WANDB_PROJECT,
        entity=Config.WANDB_ENTITY,
        name=Config.get_run_name(),
        log_model="all",
        save_dir=Config.LOG_DIR
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename=f"{Config.get_run_name()}_{{epoch:02d}}_{{val_loss:.4f}}",
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.PATIENCE,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=Config.NUM_EPOCHS,
        accelerator='auto',
        devices=1,
        gradient_clip_val=Config.GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print("\n" + "="*70)
    print("EQUIVARIANT PROTEIN GNN - FINE ÅNGSTRÖM RESOLUTION")
    print("="*70)
    print(f"Model: {Config.get_run_name()}")
    print(f"Input dim: {Config.INPUT_DIM} (amino acids only)")
    print(f"Hidden dim: {Config.HIDDEN_DIM}")
    print(f"Layers: {Config.NUM_LAYERS}")
    print(f"RBF gaussians: {Config.NUM_RBF} over {Config.RBF_MIN}-{Config.RBF_MAX}Å")
    print(f"Resolution: ~{(Config.RBF_MAX - Config.RBF_MIN) / Config.NUM_RBF:.2f}Å per gaussian")
    print("="*70 + "\n")
    
    try:
        trainer.fit(model, data_module)
        trainer.test(model, data_module, ckpt_path='best')
        
        print(f"\n✓ Training complete!")
        print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
        
    except KeyboardInterrupt:
        print("\n✗ Training interrupted")
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        raise
    finally:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()