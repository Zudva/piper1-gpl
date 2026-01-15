"""S3 checkpoint callback for PyTorch Lightning training.

Automatically uploads checkpoints to S3-compatible storage (Timeweb Cloud).
"""
import os
import subprocess
from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import Callback


class S3CheckpointCallback(Callback):
    """Upload checkpoints to S3 after they are saved."""
    
    def __init__(
        self,
        s3_sync_script: str = "script/s3_sync.sh",
        upload_on_train_epoch_end: bool = False,
        upload_on_save: bool = True,
    ):
        """
        Args:
            s3_sync_script: Path to s3_sync.sh script
            upload_on_train_epoch_end: Upload after each epoch (can be slow)
            upload_on_save: Upload when ModelCheckpoint saves (recommended)
        """
        super().__init__()
        self.s3_sync_script = s3_sync_script
        self.upload_on_train_epoch_end = upload_on_train_epoch_end
        self.upload_on_save = upload_on_save
        self._last_uploaded = None
        
    def _upload_checkpoint(self, ckpt_path: str) -> None:
        """Upload checkpoint to S3 using s3_sync.sh script."""
        if not Path(ckpt_path).exists():
            print(f"⚠️  Checkpoint not found, skipping upload: {ckpt_path}")
            return
            
        if ckpt_path == self._last_uploaded:
            print(f"ℹ️  Already uploaded: {Path(ckpt_path).name}")
            return
            
        try:
            print(f"📤 Uploading checkpoint to S3: {Path(ckpt_path).name}")
            subprocess.run(
                [self.s3_sync_script, "upload-checkpoint", ckpt_path],
                check=True,
                capture_output=False,
            )
            self._last_uploaded = ckpt_path
            print(f"✓ Uploaded: {Path(ckpt_path).name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upload checkpoint: {e}")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        """Called when Lightning saves a checkpoint."""
        if not self.upload_on_save:
            return
            
        # Get the path of the checkpoint that was just saved
        if hasattr(trainer.checkpoint_callback, "last_model_path"):
            ckpt_path = trainer.checkpoint_callback.last_model_path
            if ckpt_path:
                self._upload_checkpoint(ckpt_path)
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Called at the end of each training epoch."""
        if not self.upload_on_train_epoch_end:
            return
            
        if hasattr(trainer.checkpoint_callback, "best_model_path"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path:
                self._upload_checkpoint(ckpt_path)


class S3LogsCallback(Callback):
    """Periodically sync TensorBoard logs to S3."""
    
    def __init__(
        self,
        s3_sync_script: str = "script/s3_sync.sh",
        sync_every_n_epochs: int = 10,
    ):
        """
        Args:
            s3_sync_script: Path to s3_sync.sh script
            sync_every_n_epochs: Upload logs every N epochs
        """
        super().__init__()
        self.s3_sync_script = s3_sync_script
        self.sync_every_n_epochs = sync_every_n_epochs
        
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Upload logs every N epochs."""
        if trainer.current_epoch % self.sync_every_n_epochs == 0:
            try:
                print(f"📊 Syncing TensorBoard logs to S3 (epoch {trainer.current_epoch})")
                subprocess.run(
                    [self.s3_sync_script, "upload-logs"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✓ Logs synced to S3")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to sync logs: {e}")
