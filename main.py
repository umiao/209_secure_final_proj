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
#.train_single_batch()
torch.save(Master.state_dict(), './results/model.pth')
torch.save(Master.optimizer.state_dict(), './results/optimizer.pth')
master_model_dict = Master.state_dict()

# generate 10 clients
#distribution_lst = [[0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
#                    [0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0],
#                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
#                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0],
#                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1],
#                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1],
#                    [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
#                    ]
distribution_lst = np.random.dirichlet(np.ones(10),size=10).tolist()
client_num = 10
client_lst = []
#Master = client(params=param_dict, retrieve_history=False)

for epoch in range(0,20):
    print("epoch:{}/20".format(epoch))
    # train each client and compute gradient
    grad_lst = []
    for client in client_lst:
        grad_lst.append(client.compute_gradient())
    print(grad_lst[0])
    # compute the mean of all clients' gradient
    client_grad_mean = {}
    for param_name in grad_lst[0]:
        sum_tmp = torch.zeros(grad_lst[0][param_name].size())
        for grad in grad_lst:
            sum_tmp += grad[param_name]
        client_grad_mean[param_name] = (sum_tmp / len(grad_lst))
    #print(client_grad_mean)
    # update to master's model
    for param_name in master_model_dict:
        master_model_dict[param_name] += client_grad_mean[param_name]
    #print(master_model_dict)
    # update to client's model
    for client in client_lst:
        client.local_model.load_state_dict(master_model_dict)

Master.load_state_dict(master_model_dict)
Master.test()

