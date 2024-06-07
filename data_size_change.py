import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
import torch.optim as optim

import time


def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    # TODO: Implement rotate_img() - return the rotated img
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 1: # 90 degrees rotation
        return img.transpose(1, 2).flip(1)
    elif rot == 2: # 180 degrees rotation
        return img.flip(1).flip(2)
    elif rot == 3: # 270 degrees rotation
        return img.transpose(1, 2).flip(2)

    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class CIFAR10Rotation(torchvision.datasets.CIFAR10):

    def __init__(self, root, train, download, transform) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image, cls_label = super().__getitem__(index)

        # randomly select image rotation
        rotation_label = random.choice([0, 1, 2, 3])
        image_rotated = rotate_img(image, rotation_label)

        rotation_label = torch.tensor(rotation_label).long()
        return image, image_rotated, rotation_label, torch.tensor(cls_label).long()


def run_test(net, testloader, criterion, task, device):
    correct = 0
    total = 0
    avg_test_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, images_rotated, labels, cls_labels in testloader:
            #device = images.device
            if task == 'rotation':
              images, labels = images_rotated.to(device), labels.to(device)
            elif task == 'classification':
              images, labels = images.to(device), cls_labels.to(device)
            # TODO: Calculate outputs by running images through the network
            # The class with the highest energy is what we choose as prediction

            outputs = net(images)

            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # loss
            avg_test_loss += criterion(outputs, labels)  / len(testloader)
    print('TESTING:')
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
    print(f'Average loss on the 10000 test images: {avg_test_loss:.3f}')

def adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Both the self-supervised rotation task and supervised CIFAR10 classification are
# trained with the CrossEntropyLoss, so we can use the training loop code.

def train(net, criterion, optimizer, num_epochs, decay_epochs, init_lr, task, trainloader, testloader, device):

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        start_time = time.time()

        net.train()

        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(trainloader, 0):
            adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs)

            # TODO: Set the data to the correct device; Different task will use different inputs and labels
            #
            #device = imgs.device
            if task == 'rotation':
                inputs, labels = imgs_rotated.to(device), rotation_label.to(device)
            elif task == 'classification':
                inputs, labels = imgs.to(device), cls_label.to(device)

            # TODO: Zero the parameter gradients
            #
            optimizer.zero_grad()

            # TODO: forward + backward + optimize

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # TODO: Get predicted results
            predicted = None
            _, predicted = torch.max(outputs.data, 1)

            # print statistics
            print_freq = 100
            running_loss += loss.item()

            # calc acc
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if i % print_freq == (print_freq - 1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
                running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                start_time = time.time()

        # TODO: Run the run_test() function after each epoch; Set the model to the evaluation mode.

        net.eval()
        run_test(net, testloader, criterion, task, device)

    print('Finished Training')


def train_data_size(data_size, epochs=20):

    # main
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 128

    # Select a subset of the training set
    trainset = CIFAR10Rotation(root='./data', train=True,
                                            download=True, transform=transform_train)

    #########################################
    subset_size = 10000  # Change this value to experiment with different sizes
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    train_subset = Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = CIFAR10Rotation(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # TODO: Load the pre-trained ResNet18 model
    # Randomly initialize a ResNet18 model
    pretrain_net = resnet18(num_classes=4)

    # Load the pre-trained model from the local directory
    pretrained_path = 'final_project_1.pth'
    pretrain_net.load_state_dict(torch.load(pretrained_path))

    # Modify the final fully connected layer to match the number of CIFAR-10 classes
    num_ftrs = pretrain_net.fc.in_features
    pretrain_net.fc = nn.Linear(num_ftrs, 10)

    pretrain_net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pretrain_net.parameters(), lr=0.01, momentum=0.9)

    train(pretrain_net, criterion, optimizer, num_epochs=epochs, decay_epochs=10,
          init_lr=0.01, task='classification', trainloader=trainloader, testloader=testloader, device=device)
    torch.save(pretrain_net.state_dict(), f'pretrained_{data_size=}.pth')

    # Random Weight
    print("==================== RANDOM ========================")
    random_net = resnet18(num_classes=10)
    random_net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(random_net.parameters(), lr=0.01, momentum=0.9)

    train(random_net, criterion, optimizer, num_epochs=epochs, decay_epochs=10,
          init_lr=0.01, task='classification', trainloader=trainloader, testloader=testloader, device=device)
    torch.save(random_net.state_dict(), f'random_{data_size=}.pth')
