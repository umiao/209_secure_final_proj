from unittest import result
import numpy as np
import torch

def quantization(data, wl=16, fl=14):
    #print(data)
    upper_bound = 2**(wl-fl-1)-2**(-fl)
    lower_bound = -2**(wl-fl-1)
    precision = 2**(-fl)
    data_q = torch.clamp(data,lower_bound,upper_bound)
    quotient = (data_q - lower_bound) // precision
    data_q = lower_bound + precision * quotient
    data_q = torch.where((data_q < 0) & (data_q < data), data_q+precision, data_q)
    #print(data_q)
    return data_q

def turn_to_2D(original_dim):
    total_dim = np.prod(original_dim)
    start = int(np.sqrt(total_dim))
    factor = total_dim / start
    while(int(factor)!=factor):
        start += 1
        factor = total_dim / start
    new_dim = (int(factor),start)
    return new_dim

# input matrix should be 2D
def low_rank_approximation(matrix, threshold = 0.99):
    U, S, V = torch.linalg.svd(matrix,full_matrices=True)
    var_explained = S**2/torch.sum(S**2)
    index = 0
    sum = torch.tensor(0)
    for i,var in enumerate(var_explained):
        sum = torch.add(sum,var)
        index = i
        if(sum >= torch.tensor(threshold)):
            break
    S = S[:index+1]
    U = U[:,:len(S)]
    V = V[:len(S),:]
    #print(np.prod(matrix.shape),"->",np.prod(U.shape)+np.prod(S.shape)+np.prod(V.shape))
    total_size = np.prod(matrix.shape)
    reduced_size = np.prod(matrix.shape) - (np.prod(U.shape)+np.prod(S.shape)+np.prod(V.shape))
    return U, S, V, reduced_size, total_size

def reconstruct(U,S,V):
    return U @ torch.diag(S) @ V

def zeroize(data, filter_ratio = 0.1):
    threshold = torch.min(torch.abs(data)) + (torch.max(torch.abs(data)) - torch.min(torch.abs(data)))*filter_ratio
    #print(threshold)
    #print(torch.count_nonzero(data))
    new_data = torch.zeros(data.shape)
    result = torch.where(abs(data)>=threshold,data,new_data)
    #print(torch.count_nonzero(result))
    return result

def sparse(data):
    dim = np.prod(data.shape)
    data = zeroize(data)
    data = data.to_sparse()
    new_dim = np.prod(data.coalesce().indices().shape) + np.prod(data.coalesce().values().shape)
    reduced_size = dim - new_dim
    return data, reduced_size, dim

def dense(data):
    result = data.to_dense()
    return result

if __name__ == "__main__":
    data = torch.rand(size=(2,2,2,2))
    print(data)
    dim = data.shape
    new_dim = turn_to_2D(dim)
    data = data.reshape(new_dim)
    U, S, V = low_rank_approximation(matrix=data)
    data2 = reconstruct(U,S,V)
    data2 = data2.reshape(2,2,2,2)
    print(data2)