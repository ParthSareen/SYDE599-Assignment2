import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import optuna
from optuna.trial import TrialState

import CnnLoader as loader


class Network(nn.Module):
    def __init__(self, num_conv1_channels, num_conv2_channels, conv2_drops, fc1_neurons):
        super(Network, self).__init__()
        self.dropout = nn.Dropout2d(p=conv2_drops)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_conv1_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=num_conv1_channels, out_channels=num_conv2_channels, kernel_size=5)
        # Number of input neurons to the fully connected layer = output neurons of conv2
        # output neurons of conv 1 = ((input size (28) - kernel size (5) + 1) / max pool size (2) )^2 * num channels
        # output neurons of conv 2 = ((input size (12) - kernel size (5) + 1) / max pool size (2) )^2 * num channels
        self.fc1_in_size = ((((28 - 5 + 1) / 2) - 5 + 1) / 2) ** 2 * num_conv2_channels
        self.fc1 = nn.Linear(int(self.fc1_in_size), fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, int(self.fc1_in_size))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, train_loader, optimizer, epoch):
    device = torch.device("cuda:0")
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
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
    device = torch.device("cuda:0")

    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
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

    return loss


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

    num_conv1_channels = trial.suggest_int("num_conv1_channels", 16, 128, 16)
    num_conv2_channels = trial.suggest_int("num_conv2_channels", 16, 128, 16)
    conv_drops = trial.suggest_float("conv2_drop", 0.05, 0.2)
    num_fc1_neurons = trial.suggest_int("fc1_neurons", 10, 200, 10)

    # create the model
    model = Network(num_conv1_channels, num_conv2_channels, conv_drops, num_fc1_neurons)

    device = torch.device("cuda:0")
    model.to(device)
    
    # get the optimizers
    select_optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.001, log=True)
    print(f"params: {trial.params}")
    optimizer = getattr(optim, select_optimizer)(model.parameters(), lr=learning_rate)

    # train the model
    epoch = 0
    test_loss = None
    best_test_loss = 69696969696969
    num_non_decreasing_loss = 0
    while num_non_decreasing_loss < 3 and epoch < 20:
        train(model, train_loader, optimizer, epoch)
        test_loss = test(model, test_loader)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_non_decreasing_loss = 0
        else:
            num_non_decreasing_loss += 1
        epoch += 1

    return best_test_loss


def optimize_with_optuna():

    # params
    optuna_trials = 100

    # create an optuna study to do optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(obj_func, n_trials=optuna_trials)

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
    for key, val in optuna.importance.get_param_importances(study, target=None).items():
        print(f'{key}, {val * 100}')


if __name__ == '__main__':
    optimize_with_optuna()
