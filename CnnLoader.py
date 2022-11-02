import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class CnnLoader:
    def __init__(self, shuffle=True, num_workers=1):
        self.train = None
        self.test = None
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

    def download_dataset(self):
        self.train = datasets.MNIST(root='datasets', train=True, transform=self.transform, download=True)
        self.test = datasets.MNIST(root='datasets', train=False, transform=self.transform, download=True)

    def loaders(self):
        loaders = {
            'train': torch.utils.data.DataLoader(self.train, batch_size=100, shuffle=self.shuffle, num_workers=self.num_workers),
            'test': torch.utils.data.DataLoader(self.test, batch_size=1000, shuffle=self.shuffle,
                                                 num_workers=self.num_workers),
        }
        return loaders

def main():
    cnn = CnnLoader()
    cnn.download_dataset()
    cnn.loaders()


if __name__ == '__main__':
    main()
