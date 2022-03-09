import matplotlib.pyplot as plt
import os
import torch
from model import Net

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

param_dict = {'n_epochs' : 1,
              'batch_size_train' : 32, # 64的不好 始终保持在32
              'batch_size_test' : 1000,
              'learning_rate' : 0.001, # 0.01  持续5个batch 然后改为0.001再来5个
              'momentum' : 0.5,
              'log_interval' : 10,
              'train_set_num' : 20000, # scale of training set possessed by the client
              'class_distribution' : [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
              # the distribution of the samples of each class
              }



network = Net(params = param_dict)
# network.test()
network.train_single_batch()

should_train = True
if should_train:
    for epoch in range(1, network.n_epochs + 1):
      network.my_train(epoch)
      network.test()

    fig = plt.figure()
    plt.plot(network.train_counter, network.train_losses, color='blue')
    plt.scatter(network.test_counter, network.test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
