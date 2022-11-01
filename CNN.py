import torch.nn as nn
from torch import optim as o
from torch.autograd import Variable
from CnnLoader import *

class CNN(nn.Module):
    def __init__(self, conv1, conv2, loss=nn.CrossEntropyLoss()):
        super(CNN, self).__init__()
        # TODO: why is this the case
        self.conv1 = conv1
        self.conv2 = conv2
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.loss = loss
        loader = CnnLoader()
        loader.download_dataset()
        self.loaders = loader.loaders()
        self.optimizer = o.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        # TODO: ask ahmad
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

    def train_model(self, epochs):
        self.train()
        step = len(self.loaders['train'])
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self.loaders['train']):
                batch_x = Variable(imgs)
                batch_y = Variable(labels)
                output = self.forward(batch_x)[0]
                loss = self.loss(output, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i+1) % 100 == 0:
                    print('epoch {}/{}, step {}/{}, loss: {:.4f}'.format(epoch+1, epochs, i+1, step, loss.item()))
                    pass
            pass
        pass

    # TODO change
    def test(self):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.loaders['test']:
                test_output, last_layer = self.forward(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y==labels).sum().item()/float(labels.size(0))
                pass
        print('acc: %.2f' % accuracy)






def test():
    conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    conv2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    cnn = CNN(conv1, conv2, loss=nn.CrossEntropyLoss())
    optimizer = o.Adam(cnn.parameters(), lr=0.01)
    cnn.optimizer = optimizer
    print(cnn)
    print(optimizer)

    cnn.train_model(epochs=10)
    cnn.test()


if __name__ == '__main__':
    test()
