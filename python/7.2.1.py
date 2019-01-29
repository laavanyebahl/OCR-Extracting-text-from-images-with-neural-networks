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
import matplotlib.pyplot as plt
import torchvision

max_iters = 35
batch_size = 64
learning_rate = 2e-2

training_loss_list = []
training_loss_list2 = []
training_loss_list3 = []

# training_accuracy_list = []
validation_accuracy_list = []
# test_accuracy_list = []
validation_accuracy_list2 = []
validation_accuracy_list3 = []

finetuning_epoch = 7

global flag
flag =0

class Flower17(Dataset):
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


# Defining the network (LeNet-5)
class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1,  bias=True)
        # Max-pooling
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=True)
        # Max-pooling
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5,
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = nn.Linear(84, 17)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = nn.functional.relu(self.conv1(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)
        # convolve, then perform ReLU non-linearity
        x = nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16 * 5 * 5)
        # FC-1, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return x

def train(args, model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    total_loss = 0
    correct= 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # imshow(torchvision.utils.make_grid(data))

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

    global flag
    if flag ==2:
        training_loss_list2.append(total_loss)

    if flag ==3:
        training_loss_list3.append(total_loss)


    print('\nTrain set: Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(train_loader.dataset), avg_acc))

def validation_accuracy(args, model, device, validation_loader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
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

def validation_accuracy2(args, model, device, validation_loader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # avg_loss = total_loss / len(validation_loader.dataset)
    avg_acc = 100. * correct / len(validation_loader.dataset)
    validation_accuracy_list2.append(avg_acc)

    print('\nValidation set: Total loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(validation_loader.dataset), avg_acc))

def validation_accuracy3(args, model, device, validation_loader, loss_func):
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += loss_func(output, target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # avg_loss = total_loss / len(validation_loader.dataset)
        avg_acc = 100. * correct / len(validation_loader.dataset)
        validation_accuracy_list3.append(avg_acc)

        print('\nValidation set: Total loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
            total_loss, correct, len(validation_loader.dataset), avg_acc))

def test_accuracy(args, model, device, test_loader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = 100. * correct / len(test_loader.dataset)
    # test_accuracy_list.append(avg_acc)

    print('\nTest set: Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(test_loader.dataset), avg_acc))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Parser')
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
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    trainset = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform=data_transform)
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    validset = datasets.ImageFolder(root='../data/oxford-flowers17/val', transform=data_transform)
    validset_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = datasets.ImageFolder(root='../data/oxford-flowers17/test', transform=data_transform)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = torch.nn.CrossEntropyLoss()


    #### COMMENT OUT CODE FOR TRAINING , MODEL SAVED IN DATA

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainset_loader, optimizer, loss_func, epoch)
        validation_accuracy(args, model, device, validset_loader,  loss_func)

    test_accuracy(args, model, device, testset_loader, loss_func)

    torch.save(model, './q7_2_1')

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

    model = torch.load('./q7_2_1')
    model.eval()

    ## FINETUNINGGGG

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    data_transform =transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    trainset = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform=data_transform)
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    validset = datasets.ImageFolder(root='../data/oxford-flowers17/val', transform=data_transform)
    validset_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = datasets.ImageFolder(root='../data/oxford-flowers17/test', transform=data_transform)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1)


    dtype = torch.FloatTensor

    #We will first reinitialize the last layer of the
    # model, and train only the last layer for a few epochs. We will then finetune
    # the entire model on our dataset for a few more epochs.

    # First load the pretrained squeezenet model; this will download the model
    # weights from the web the first time you run it.
    model = torchvision.models.squeezenet1_1(pretrained=True)

    # Reinitialize the last layer of the model. Each pretrained model has a
    # slightly different structure, but from the ResNet class definition
    # we see that the final fully-connected layer is stored in model.fc:
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L111
    num_classes = len(trainset.classes)

    model.num_classes = num_classes
    model.classifier[1] = nn.Conv2d(512, model.num_classes, stride=(1,1), kernel_size=1)

    # Cast the model to the correct datatype, and create a loss function for
    # training the model.
    model.type(dtype)
    loss_func = nn.CrossEntropyLoss().type(dtype)

    # First we want to train only the reinitialized last layer for a few epochs.
    # During this phase we do not need to compute gradients with respect to the
    # other weights of the model, so we set the requires_grad flag to False for
    # all model parameters, then set requires_grad=True for the parameters in the
    # last layer only.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    # Construct an Optimizer object for updating the last layer only.
    optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=1e-3)

    # Update only the last layer for a few epochs.

    global flag
    flag =2

    for epoch in range(finetuning_epoch):
        # Run an epoch over the training data.
        print('Starting epoch %d / %d' % (epoch , finetuning_epoch))

        train(args, model, device, trainset_loader, optimizer, loss_func, epoch)
        validation_accuracy2(args, model, device, validset_loader,  loss_func)

    import matplotlib.pyplot as plt
    import  numpy as np
    figure = plt.figure()
    plt.plot(np.arange(finetuning_epoch) + 1, validation_accuracy_list2)
    plt.title('Training accuracy over the epochs')
    plt.show()

    figure = plt.figure()
    plt.plot(np.arange(finetuning_epoch) + 1, training_loss_list2)
    plt.title('Training loss over the epochs')
    plt.show()


    # Now we want to finetune the entire model for a few epochs. To do thise we
    # will need to compute gradients with respect to all model parameters, so
    # we flag all parameters as requiring gradients.
    for param in model.parameters():
        param.requires_grad = True

    # Construct a new Optimizer that will update all model parameters. Note the
    # small learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    # Train the entire model for a few more epochs, checking accuracy on the
    # train and validation sets after each epoch.

    global flag
    flag = 3
    for epoch in range(finetuning_epoch):
        print('Starting epoch %d / %d' % (epoch , finetuning_epoch))

        train(args, model, device, trainset_loader, optimizer, loss_func, epoch)
        validation_accuracy3(args, model, device, validset_loader, loss_func)

    test_accuracy(args, model, device, testset_loader, loss_func)


    import matplotlib.pyplot as plt
    import  numpy as np
    figure = plt.figure()
    plt.plot(np.arange(finetuning_epoch) + 1, validation_accuracy_list3)
    plt.title('Training accuracy over the epochs')
    plt.show()

    figure = plt.figure()
    plt.plot(np.arange(finetuning_epoch) + 1, training_loss_list3)
    plt.title('Training loss over the epochs')
    plt.show()


def imshow(img):
    npimg = img.numpy()
    # print(npimg)
    print('max: ', npimg.max())
    print('min: ', npimg.min())

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()