# indir is MNIST dataset folder path

# setting is a matrix with 9 rows and n columns where n is the number of clients
# The ratio at (i,j) represents for MNIST class i, how many data client j have
# Ex: The first row is [0.2,0.2,0.2,0.2,0.2] -> For MNIST class 1, each client has 20% of data
# Ex: The thrid row is [1.0,0.0,0.0,0.0,0.0] -> For MNIST class 3, only cleint 1 has data
import random
import numpy as np
import torch
# outdir should contain n subfolder such that each subfolder has dataset for each client
def partition(data_obj, setting, nums):
    #todo
    data = data_obj.data.numpy()
    targets = data_obj.targets.numpy()

    ret_data = None
    ret_targets = []
    keep_prob = nums / 60000

    for i in range(len(targets)):
        if random.random() < keep_prob:
            c = targets[i]
            if random.random() < setting[c]:
                ret_targets.append(targets[i])
                if ret_data is None:
                    ret_data = data[i, :, :]
                    ret_data = ret_data[np.newaxis, :]
                else:
                    to_append = data[i, :, :]
                    ret_data = np.concatenate([ret_data, to_append[np.newaxis, :]], axis=0)
                # preserve this class
        else:
            continue
    data_obj.data = torch.tensor(ret_data)
    data_obj.targets = torch.tensor(ret_targets)
    return data_obj

if __name__ == "__main__":
    indir = '/MNIST'
    client_num = 5
    setting = [
        [0.2,0.2,0.2,0.2,0.2]
        [0.1,0.2,0.3,0.3,0.1]
        [1.0,0.0,0.0,0.0,0.0]
        [0.0,1.0,0.0,0.0,0.0]
        [0.0,0.0,1.0,0.0,0.0]
        [0.0,0.0,0.0,1.0,0.0]
        [0.0,0.0,0.0,0.0,1.0]
        [0.5,0.2,0.2,0.1,0.0]
        [0.1,0.2,0.3,0.2,0.2]
    ]
    outdir = '/MNIST_partition'
    partition_dataset(indir,setting,outdir)