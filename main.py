import matplotlib.pyplot as plt
import os
import torch
from server import client
from model import Net

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

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

#generate master model
Master = Net(params = param_dict)
# network.test()
Master.train_single_batch()

should_train = True
if should_train:
    for epoch in range(1, Master.n_epochs + 1):
        print("first train")
        Master.my_train(epoch)
        Master.test()

# generate 10 clients
distribution_lst = [[0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1],
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1],
                    [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
                    ]
client_num = 10
client_lst = []
#Master = client(params=param_dict, retrieve_history=False)

for c in range(0, client_num):
    param_dict['class_distribution'] = distribution_lst[c]
    client_tmp = client(retrieve_history=True, params=param_dict)
    client_lst.append(client_tmp)

#master_model_dict = Master.update_to_global_model()
#for client in client_lst:
#    client.update_to_local_model(master_model_dict)

# train each client and compute gradient
grad_lst = []
for client in client_lst:
    grad_lst.append(client.compute_gradient())

# compute the mean of all clients' gradient
client_grad_mean = {}
for param_name in grad_lst[0]:
    sum_tmp = torch.zeros(grad_lst[0][param_name].size())
    for grad in grad_lst:
        sum_tmp += grad[param_name]
    client_grad_mean[param_name] = (sum_tmp / len(grad_lst))

# update to master's model
master_model_dict = Master.state_dict()
for param_name in master_model_dict:
    master_model_dict[param_name] += client_grad_mean[param_name]
Master.load_state_dict(master_model_dict)

#train again
should_train = True
if should_train:
    for epoch in range(1, Master.n_epochs + 1):
        print("second train")
        Master.my_train(epoch)
        Master.test()
