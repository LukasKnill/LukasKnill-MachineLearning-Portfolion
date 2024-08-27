
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl


###Exercise 2###
class ModifiedCNN(pl.LightningModule):
    def __init__(self):
        super(ModifiedCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16,(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        y = self.fc(x)
        return y


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


