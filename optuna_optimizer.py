import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import optuna
from optuna.trial import TrialState

import CnnLoader as loader


class Network(nn.Module):
    def __init__(self, num_conv1_nodes, num_conv2_nodes, conv2_drops, fc1_neurons):
        super(Network, self).__init__()
        # TODO: Why
        # generate list of convolutional layers
        # input_size = 28 # images are 28x28 pixels 
        # kernel_size = 5
        # self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_conv_nodes[0], kernel_size=kernel_size)])
        # output_size = (input_size - kernel_size + 1) / 2
        # for i in range(1, num_conv_layers):
        #     self.conv_layers.append(nn.Conv2d(in_channels=num_conv_nodes[i], out_channels=output_size, kernel_size=kernel_size))
        
        self.dropout = nn.Dropout2d(p=conv2_drops)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_conv1_nodes, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=num_conv1_nodes, out_channels=num_conv2_nodes, kernel_size=5)
        self.fc1_in_size = int(((((28 - 5 + 1) / 2) - 5 + 1) / 2) * num_conv2_nodes * num_conv2_nodes)
        self.fc1 = nn.Linear(self.fc1_in_size, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 10)

    # TODO Look through
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, int(self.fc1_in_size))
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


    num_conv1_nodes = trial.suggest_int("num_conv1_nodes", 16, 128, 16)
    num_conv2_nodes = trial.suggest_int("num_conv2_nodes", 16, 128, 16)
    conv_drops = trial.suggest_float("conv2_drop", 0.1, 0.5)
    num_fc1_neurons = trial.suggest_int("fc1_neurons", 10, 200, 10)

    # create the model
    model = Network(num_conv1_nodes, num_conv2_nodes, conv_drops, num_fc1_neurons)
    
    # get the optimizers
    select_optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
    optimizer = getattr(optim, select_optimizer)(model.parameters(), lr=learning_rate)

    # train the model
    for epoch in range(1, 5):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


def optimize_with_optuna():

    # params
    optuna_trials = 2

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

    df = study.trials_dataframe().sort_values('value')
    df.to_csv('optuna_results_1.csv', index=False)

    print('most important hyperparams')
    for key, val in optuna.importance.get_param_importances(study, target=None):
        print(f'{key}, {15-len(key)}, {val * 100}')

    


def main():
    cnn_loader = loader.CnnLoader()
    cnn_loader.download_dataset()
    loaders = cnn_loader.loaders()
    train_loader = loaders['train']
    test_loader = loaders['test']

    model = Network(10, 100, 0.1, 50)
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate,
    #                       momentum=momentum)
    optimizer = optim.Adam(model.parameters())
    optim

    for epoch in range(1, 5):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


if __name__ == '__main__':
    main()
    # optimize_with_optuna()  