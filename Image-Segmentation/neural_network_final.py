import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from segmentation_models_pytorch import Unet

class Network(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Implement the U-Net with the ResNet18 backbone and the weights of ImageNet
        self.unet = Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=2)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=2)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=2)

    def forward(self, x):
        embedding = self.unet(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.valid_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.valid_acc, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
