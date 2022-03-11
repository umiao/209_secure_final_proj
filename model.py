import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.optim as optim
from partition_dataset import partition
import copy

class Net(nn.Module):
    n_epochs = 5
    batch_size_train = 32  # 64的不好 始终保持在32
    batch_size_test = 1000
    learning_rate = 0.001  # 0.01  持续5个batch 然后改为0.001再来5个
    momentum = 0.5
    log_interval = 10



    def __init__(self, retrieve_history=True, params=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        if not params:
            self.n_epochs = 5
            self.batch_size_train = 32  # 64的不好 始终保持在32
            self.batch_size_test = 1000
            self.learning_rate = 0.001  # 0.01  持续5个batch 然后改为0.001再来5个
            self.momentum = 0.5
            self.log_interval = 10
        else:
            self.n_epochs = params['n_epochs']
            self.batch_size_train = params['batch_size_train']
            self.batch_size_test = params['batch_size_test']
            self.learning_rate = params['learning_rate']
            self.momentum = params['momentum']
            self.log_interval = params['log_interval']
            self.train_set_num = params['train_set_num']
            self.class_distribution = params['class_distribution']


        MNIST_obj = torchvision.datasets.MNIST('./', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
        MNIST_obj = partition(MNIST_obj, self.class_distribution, self.train_set_num)

        self.train_loader = torch.utils.data.DataLoader(
            MNIST_obj,
            batch_size=self.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_test, shuffle=True)

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate,
                      momentum=self.momentum)

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]

        if retrieve_history:
            network_state_dict = torch.load('./results/model.pth')
            self.load_state_dict(network_state_dict)

            optimizer_state_dict = torch.load('./results/optimizer.pth')
            self.optimizer.load_state_dict(optimizer_state_dict)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def my_train(self, epoch):
        # would train a whole epoch, with training feedback
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append((batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
                torch.save(self.state_dict(), './results/model.pth')
                torch.save(self.optimizer.state_dict(), './results/optimizer.pth')

    def train_single_batch(self):
        # would train a single batch, without training feedback
        # would NOT update the local model
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self(data)
            old_named_parameters = {}
            for name, tensor_v in self.named_parameters():
                old_named_parameters[name] = copy.deepcopy(tensor_v)

            loss = F.nll_loss(output, target)
            # loss.backward(retain_graph=True)
            loss.backward()
            nn.utils.clip_grad_norm(self.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
            computed_gradient = {}
            for name, tensor_v in self.named_parameters():
                #computed_gradient.append(tensor_v.grad)
                computed_gradient[name] = tensor_v - old_named_parameters[name]
            return computed_gradient


    def test(self):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))