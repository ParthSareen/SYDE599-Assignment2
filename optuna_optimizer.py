import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import optuna
from optuna.trial import TrialState

import CnnLoader as loader


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Why
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # TODO Look through
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                epoch,
                batch_idx * len(inputs),
                len(train_loader.dataset),
                loss))

    print('Training loss: {:.6f}'.format(total_loss / len(train_loader.dataset) * len(inputs)))


def test(model, test_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss += nn.CrossEntropyLoss()(outputs, targets)
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct += predictions.eq(targets.view_as(predictions)).sum()

    loss = loss / len(test_loader.dataset) * len(inputs)

    print('Test loss: {:.6f}; Test accuracy: {}/{} ({:.1f}%)\n'.format(
        loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def obj_func(trial):
    """
    obj_func generates the objective funciton which optuna will optimize,
    hyperparams to be optimized: optimizer type, learning rate, dropout vals

    Args:
        trial (optuna.trial._trial.Trial): self explanatory
    """
    cnn_loader = loader.CnnLoader()
    cnn_loader.download_dataset()
    loaders = cnn_loader.loaders()
    train_loader = loaders['train']
    test_loader = loaders['test']

    # create the model
    model = Network()
    
    # get the optimizers
    select_optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
    optimizer = getattr(optim, select_optimizer)(model.parameters(), lr=learning_rate)

    for epoch in range(1, 5):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


def optimize_with_optuna():

    # params
    optuna_trials = 20

    # create an optuna study to do optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(obj_func, n_trials=10)

    # report stats from optimization
    best_trial = study.best_trial
    print("Optimized_param_vals: ")
    print(f'Value: {best_trial}')
    print('params')
    for key, val in best_trial.params.items():
        print(f'{key}, {val}')

    


def main():
    cnn_loader = loader.CnnLoader()
    cnn_loader.download_dataset()
    loaders = cnn_loader.loaders()
    train_loader = loaders['train']
    test_loader = loaders['test']

    model = Network()
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate,
    #                       momentum=momentum)
    optimizer = optim.Adam(model.parameters())
    optim

    for epoch in range(1, 5):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


if __name__ == '__main__':
    # main()
    optimize_with_optuna()
