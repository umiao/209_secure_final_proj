import numpy as np
import torch
#import pymf3

# "Non-negative matrix factorization"
# Given a data 'V' which is non-negative matrix with size m*n
# Return two matrices 'W'(base) and 'H'(coefficient) with size m*k and k*n respectively
# such that V ~ W*H
# num_bases decides k, and niter decides how many iterations to find solution 
#def low_rank_compression(data, num_bases=2, niter=10):
#    nmf_mdl = pymf3.NMF(data, num_bases=num_bases, niter=niter)
#    nmf_mdl.factorize()
#    return nmf_mdl.W, nmf_mdl.H

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