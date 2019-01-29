from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

max_iters = 5
batch_size = 32
learning_rate = 1e-2

training_loss_list = []
# training_accuracy_list = []
validation_accuracy_list = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    total_loss = 0
    correct= 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()

        ## forward pass
        output = model(data)
        loss = loss_func(output, target)
        total_loss += loss.item()

        ## back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## accuracy
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # avg_loss = total_loss/ len(train_loader.dataset)
    avg_acc = 100. * correct / len(train_loader.dataset)
    # training_accuracy_list.append(avg_acc)
    training_loss_list.append(total_loss)

    print('\nTrain set: Total loss: {:.4f}, Train Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(train_loader.dataset), avg_acc))

def validation_accuracy(args, model, device, validation_loader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.float()

            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # avg_loss = total_loss / len(validation_loader.dataset)
    avg_acc = 100. * correct / len(validation_loader.dataset)
    validation_accuracy_list.append(avg_acc)

    print('\nValidation set: Total loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(validation_loader.dataset), avg_acc))

def test_accuracy(args, model, device, test_loader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()

            output = model(data)
            total_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(test_loader.dataset), avg_acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=max_iters, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_func =  F.nll_loss

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, loss_func, epoch)
        validation_accuracy(args, model, device, test_loader,  loss_func)

    test_accuracy(args, model, device, test_loader, loss_func)

    import matplotlib.pyplot as plt
    import  numpy as np
    figure = plt.figure()
    plt.plot(np.arange(max_iters) + 1, validation_accuracy_list)
    plt.title('Training accuracy over the epochs')
    plt.show()

    figure = plt.figure()
    plt.plot(np.arange(max_iters) + 1, training_loss_list)
    plt.title('Training loss over the epochs')
    plt.show()


if __name__ == '__main__':
    main()