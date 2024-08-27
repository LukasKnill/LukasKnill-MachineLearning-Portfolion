
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision import models


###Exercise 2###
class CropsCNN(pl.LightningModule):
    def __init__(self):
        super(CropsCNN, self).__init__()
        self.model_ft = models.vgg16(pretrained=True)

        for param in self.model_ft.parameters():
            param.requires_grad = False

        num_ftrs = self.model_ft.classifier[6].in_features
        self.model_ft.classifier[6] = nn.Linear(num_ftrs, 10)

        self.train_acc = torchmetrics.Accuracy(num_classes=10, task="multiclass")
        self.val_acc = torchmetrics.Accuracy(num_classes=10, task="multiclass")
        self.test_acc = torchmetrics.Accuracy(num_classes=10, task="multiclass")


    def forward(self, x):
        x = self.model_ft(x)
        return x


###Exercise 3###
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


###Exercise 4###
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True)
        self.train_acc(y_hat, y)
        self.log("train_accuracy", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_step=True)
        self.val_acc(y_hat, y)
        self.log("val_accuracy", self.val_acc, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, on_step=True)
        self.test_acc(y_hat, y)
        self.log("test_accuracy", self.test_acc, on_epoch=True)
        return test_loss


