from data_loading_final import DataLoading
from neural_network_final import Network
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch


if __name__ == "__main__":

    model= Network()

    data_loading = DataLoading()

    train_loader, val_loader, test_loader = data_loading.load_data()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer(devices=1,max_epochs=100, accelerator="auto", default_root_dir=r"D:\Studium\3. Semster\Bildanalyse\Assignments\6\logs", logger=tb_logger,log_every_n_steps=10,
                         callbacks=EarlyStopping(monitor="val_loss",min_delta= 0.01, patience=5, verbose=True))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(ckpt_path="best", dataloaders=test_loader)

    torch.save(model, r"D:\Studium\3. Semster\Bildanalyse\Assignments\6\models\segmentation_model_2.pt")
