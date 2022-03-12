import matplotlib.pyplot as plt
import os
import torch
from server import client
from model import Net
import numpy as np
import copy
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
np.random.seed(random_seed)

param_dict = {'n_epochs' : 1,
              'batch_size_train' : 32, # 64的不好 始终保持在32
              'batch_size_test' : 1000,
              'learning_rate' : 0.01, # 0.01  持续5个batch 然后改为0.001再来5个
              'momentum' : 0.5,
              'log_interval' : 10,
              'train_set_num' : 6000, # scale of training set possessed by the client
              'class_distribution' : [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
              # the distribution of the samples of each class
              }

#generate master model
Master = Net(params = None, retrieve_history=False)
torch.save(Master.state_dict(), './results/model.pth')
torch.save(Master.optimizer.state_dict(), './results/optimizer.pth')
master_model_dict = copy.deepcopy(Master.state_dict())

client_num = 10
epoch_size = 10000
distribution_lst = np.random.dirichlet(np.ones(10),size=client_num).tolist()
print("# of Clients: %d" % client_num)
print("# of Epochs: %d" % epoch_size)
print("Learning rate: %f" % param_dict['learning_rate'])
#print("Data distribution: ")
#for i in range(client_num):
#    print(distribution_lst[i])
client_lst = []

for c in range(0, client_num):
    param_dict['class_distribution'] = distribution_lst[c]
    client_tmp = client(params=param_dict, retrieve_history=False)
    client_lst.append(client_tmp)
#print(client_lst)
Master.test()

for epoch in range(epoch_size):
    # train each client and compute gradient
    grad_lst = []
    for client in client_lst:
        grad_lst.append(client.compute_gradient())
    #print(grad_lst)
    # compute the mean of all clients' gradient
    client_grad_mean = {}
    for param_name in grad_lst[0]:
        sum_tmp = torch.zeros(grad_lst[0][param_name].size())
        for grad in grad_lst:
            sum_tmp += grad[param_name]
        client_grad_mean[param_name] = (sum_tmp / len(grad_lst))
        nn.utils.clip_grad_norm(client_grad_mean[param_name], max_norm=20, norm_type=2)
    #print(client_grad_mean)
    # update to master's model
    for param_name in master_model_dict:
        # master_model_dict[param_name] += client_grad_mean[param_name]
        master_model_dict[param_name] += client_grad_mean[param_name]
    #print(master_model_dict)
    # update to client's model
    for client in client_lst:
        client.local_model.load_state_dict(master_model_dict)

    Master.load_state_dict(master_model_dict)
    if epoch % 20 == 0:
        print('Epoch: %d/%d' % (epoch, epoch_size))
        Master.test()
        torch.save(Master.state_dict(), './results/model.pth')
