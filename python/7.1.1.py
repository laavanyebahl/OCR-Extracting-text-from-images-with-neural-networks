from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io
import  numpy as np

max_iters = 85
batch_size = 12
learning_rate = 1e-3
hidden_size = 64

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
                    self.transform = None
                else:
                    self.images.append(x[row].reshape(1, 32,32))
                    self.transform = None
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
        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1024, 64, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc1.bias.data.fill_(0)

        self.sigmoid = nn.Sigmoid()

        self.fc_out = nn.Linear(64, 36)
        # torch.nn.init.xavier_uniform_(self.fc_out.weight)
        # self.fc_out.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc_out(out)

        return out

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

    print('\nTest set: Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(test_loader.dataset), avg_acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='7.1.1')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=max_iters, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    trainset = NIST36( root='../data/nist36_train.mat', preload=True, networkType='fc', dataType='train',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ]) )
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = NIST36( root='../data/nist36_test.mat', preload=True, networkType='fc', dataType='test',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ]) )
    testset_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainset_loader, optimizer, criterion, epoch)
        validation_accuracy(args, model, device, testset_loader,  criterion)

    test_accuracy(args, model, device, testset_loader, criterion)

    import matplotlib.pyplot as plt
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