from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io
import  numpy as np

max_iters = 35
batch_size = 32
learning_rate = 1e-2

training_loss_list = []
# training_accuracy_list = []
validation_accuracy_list = []


class NIST36(Dataset):
    def __init__(self, root, transform=None, preload=False, networkType='fc', dataType='train'):

        self.images = None
        self.labels = None
        self.root = root
        self.transform = transform

        data = scipy.io.loadmat(root)
        x, y = data[dataType+'_data'], data[ dataType+'_labels']

        # if preload dataset into memory
        if preload:
            self.labels = []
            self.images = []

            for row in range(x.shape[0]):
                if networkType == 'fc':
                    self.images.append(x[row])
                else:
                    self.images.append(x[row].reshape(1,32,32))
                self.labels.append(np.argmax(y[row]))

        self.len = x.shape[0]


    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]

        # May use transform function to transform samples
        # e.g., random crop, whitening
        # if self.transform is not None:
        #     image = self.transform(image)
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



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

    print('\nTrain set: Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(train_loader.dataset), avg_acc))

def validation_accuracy(args, model, device, validation_loader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()

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

    print('\nTest set: Total loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%)\n'.format(
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

    trainset = NIST36(root='../data/nist36_train.mat', preload=True, networkType='conv', dataType='train')
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = NIST36(root='../data/nist36_test.mat', preload=True, networkType='conv', dataType='test')
    testset_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_func =  F.nll_loss

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainset_loader, optimizer, loss_func, epoch)
        validation_accuracy(args, model, device, trainset_loader,  loss_func)

    test_accuracy(args, model, device, testset_loader, loss_func)

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