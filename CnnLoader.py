import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class CnnLoader:
    def __init__(self, batch_size=100, shuffle=True, num_workers=1):
        self.train = None
        self.test = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def download_dataset(self):
        self.train = datasets.MNIST(root='datasets', train=True, transform=ToTensor(), download=True)
        self.test = datasets.MNIST(root='datasets', train=False)

    def loaders(self):
        loaders = {
            'train': torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers),
            'test': torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers),
        }
        return loaders

def main():
    cnn = CnnLoader()
    cnn.download_dataset()


if __name__ == '__main__':
    main()
