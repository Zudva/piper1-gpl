import logging
import os

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

# S3 integration
try:
    from .s3_callbacks import S3CheckpointCallback, S3LogsCallback
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

_LOGGER = logging.getLogger(__package__)


class SaveOnInterrupt(Callback):
    """Сохраняет чекпоинт при любом исключении (в том числе Ctrl+C/SIGTERM)."""

    def __init__(self, filename: str = "interrupt.ckpt") -> None:
        self.filename = filename

    def on_exception(self, trainer, pl_module, err) -> None:  # type: ignore[override]
        # Делаем бэкап даже если возникло KeyboardInterrupt.
        dirpath = trainer.log_dir or trainer.default_root_dir or "."
        ckpt_dir = os.path.join(dirpath, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, self.filename)
        trainer.save_checkpoint(ckpt_path)


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")
    
    def instantiate_trainer(self, **kwargs):
        # Add periodic checkpoint saving (epochs and steps) + best model
        if "callbacks" not in kwargs or kwargs["callbacks"] is None:
            kwargs["callbacks"] = []
        
        checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=5000,     # регулярное сохранение по шагам
            save_top_k=3,                 # хранить 3 лучших по val_loss
            monitor="val_loss",
            mode="min",
            filename="epoch={epoch}-step={step}-val_loss={val_loss:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        kwargs["callbacks"].append(checkpoint_callback)
        kwargs["callbacks"].append(SaveOnInterrupt())
        
        # Add S3 sync callbacks if enabled
        if S3_AVAILABLE and os.getenv("ENABLE_S3_SYNC", "0") == "1":
            _LOGGER.info("S3 sync enabled, adding S3 callbacks")
            kwargs["callbacks"].append(S3CheckpointCallback(upload_on_save=True))
            kwargs["callbacks"].append(S3LogsCallback(sync_every_n_epochs=10))
        
        return super().instantiate_trainer(**kwargs)


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel, VitsDataModule, trainer_defaults={"max_epochs": -1}
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
