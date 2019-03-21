from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from layers import SVDConv2d
import matplotlib.pyplot as plt
import time

training_loss = []
testing_acc = []
k = 0.25

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SVDConv2d(1, 20, 5, k)
        self.conv2 = SVDConv2d(20, 50, 5, k)
        self.conv3 = SVDConv2d(50, 10, 4, k)
#        self.conv1 = nn.Conv2d(1, 20, 5, 1)
#        self.conv2 = nn.Conv2d(20, 50, 5, 1)
#        self.conv3 = nn.Conv2d(50, 10, 4, 1)

    def orth_reg(self):
        reg = 0
        for m in self.modules():
            if isinstance(m, SVDConv2d):
                reg += m.orth_reg()
        return reg

    def D_optimal_reg(self):
        reg = 0
        for m in self.modules():
            if isinstance(m, SVDConv2d):
                reg += m.spectral_reg()
        return reg

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss += 0.1*model.orth_reg()
        loss += 10*model.D_optimal_reg()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            training_loss.append(loss.item())

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    testing_acc.append(correct)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        curr = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        print("Time it took for this epoch is: " + str(time.time() - curr))
        test(args, model, device, test_loader)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    filename = f'train-loss-{k}'
    with open(filename, 'w') as fw:
        for train_loss in training_loss:
            fw.write('{}\n'.format(train_loss))

    filename = f'test-loss-{k}'
    with open(filename, 'w') as fw:
        for test_loss in testing_acc:
            fw.write('{}\n'.format(test_loss))

    if (args.save_model):
        torch.save(model.state_dict(),f"mnist_cnn-{k}.pt")
        
if __name__ == '__main__':
    main()
