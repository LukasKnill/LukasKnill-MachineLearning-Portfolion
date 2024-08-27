import torch
import torchvision as torchvision
from torch.utils.data import random_split, DataLoader
from Dataset_final import Dataset


class DataLoading:
    def __init__(self):
        self.batch_size = 8



    def load_data(self):
        dataset = Dataset(root_dir_rgbs=r"D:\Studium\3. Semster\Bildanalyse\Assignments\6\rgbs",
                          root_dir_masks=r"D:\Studium\3. Semster\Bildanalyse\Assignments\6\masks",
                          transform=torchvision.transforms.Compose([torchvision.transforms.Resize((192, 160))]))

        train_datasize = int(len(dataset) * 0.9)
        test_size = len(dataset) - train_datasize
        train_dataset, test_set = torch.utils.data.random_split(dataset, [train_datasize, test_size])

        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader