import torch
import torchvision as torchvision
from torch.utils.data import random_split, DataLoader


###Exercise 5###
class DataLoading:
    def __init__(self):
        self.filePath = "/files/"
        self.batch_size = 16
        self.num_workers = 0


    def load_data(self):

        train_set = torchvision.datasets.ImageFolder(root=r"D:\Studium\3. Semster\Bildanalyse\Assignments\3\Test",
                                                     transform=torchvision.transforms.Compose(
                                                         [torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Resize((100, 100)),
                                                          torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                           (0.5, 0.5, 0.5))]))

        test_set = torchvision.datasets.ImageFolder(root=r"D:\Studium\3. Semster\Bildanalyse\Assignments\3\Training",
                                                    transform=torchvision.transforms.Compose(
                                                        [torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize((100, 100)),
                                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                          (0.5, 0.5, 0.5))]))
        trainlength = round(0.9 * len(train_set))
        vallength = round(0.1 * len(train_set))
        train_set, val_set = random_split(train_set, [trainlength, vallength])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
