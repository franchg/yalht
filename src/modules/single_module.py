from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss


class MNISTLitModule(BaseLitModule):
    """LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize MNIST LightningModule.

        Args:
            network: Network config with model, loss, output_activation
            optimizer: Optimizer config
            scheduler: Scheduler config
            logging: Logging config
            args: Additional arguments for BaseLitModule
            kwargs: Additional keyword arguments for BaseLitModule
        """
        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        self.output_activation = hydra.utils.instantiate(
            network.output_activation, _partial_=True
        )

        # Simple accuracy tracking
        self.train_correct = 0
        self.train_total = 0
        self.valid_correct = 0
        self.valid_total = 0
        self.test_correct = 0
        self.test_total = 0
        self.valid_acc_best = 0.0

        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def model_step(self, batch: Any) -> tuple:
        """Perform a single model step (used in train/val/test)."""
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = self.output_activation(logits)
        return loss, preds, y

    def on_train_epoch_start(self) -> None:
        """Reset training metrics at start of epoch."""
        self.train_correct = 0
        self.train_total = 0

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        """Training step."""
        loss, preds, targets = self.model_step(batch)

        # Calculate accuracy
        pred_classes = preds.argmax(dim=1)
        self.train_correct += (pred_classes == targets).sum().item()
        self.train_total += targets.size(0)

        # Log loss
        self.log(
            "loss/train",
            loss,
            **self.logging_params,
        )

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """Log training accuracy at end of epoch."""
        if self.train_total > 0:
            acc = self.train_correct / self.train_total
            self.log(
                "acc/train",
                acc,
                **self.logging_params,
            )

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at start of epoch."""
        self.valid_correct = 0
        self.valid_total = 0

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """Validation step."""
        loss, preds, targets = self.model_step(batch)

        # Calculate accuracy
        pred_classes = preds.argmax(dim=1)
        self.valid_correct += (pred_classes == targets).sum().item()
        self.valid_total += targets.size(0)

        # Log loss
        self.log(
            "loss/valid",
            loss,
            **self.logging_params,
        )

        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        """Log validation accuracy and track best."""
        if self.valid_total > 0:
            acc = self.valid_correct / self.valid_total
            self.log(
                "acc/valid",
                acc,
                **self.logging_params,
            )

            # Track best accuracy
            if acc > self.valid_acc_best:
                self.valid_acc_best = acc

            self.log(
                "acc/valid_best",
                self.valid_acc_best,
                **self.logging_params,
            )

    def on_test_epoch_start(self) -> None:
        """Reset test metrics at start of epoch."""
        self.test_correct = 0
        self.test_total = 0

    def test_step(self, batch: Any, batch_idx: int) -> dict:
        """Test step."""
        loss, preds, targets = self.model_step(batch)

        # Calculate accuracy
        pred_classes = preds.argmax(dim=1)
        self.test_correct += (pred_classes == targets).sum().item()
        self.test_total += targets.size(0)

        # Log loss
        self.log(
            "loss/test",
            loss,
            **self.logging_params,
        )

        return {"loss": loss}

    def on_test_epoch_end(self) -> None:
        """Log test accuracy at end of epoch."""
        if self.test_total > 0:
            acc = self.test_correct / self.test_total
            self.log(
                "acc/test",
                acc,
                **self.logging_params,
            )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Prediction step."""
        x, y = batch
        logits = self.forward(x)
        preds = self.output_activation(logits)
        return {"logits": logits, "preds": preds, "targets": y}
