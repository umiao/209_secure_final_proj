import numpy as np
import pymf3

# "Non-negative matrix factorization"
# Given a data 'V' which is non-negative matrix with size m*n
# Return two matrices 'W'(base) and 'H'(coefficient) with size m*k and k*n respectively
# such that V ~ W*H
# num_bases decides k, and niter decides how many iterations to find solution 
def low_rank_compression(data, num_bases=2, niter=10):
    nmf_mdl = pymf3.NMF(data, num_bases=num_bases, niter=niter)
    nmf_mdl.factorize()
    return nmf_mdl.W, nmf_mdl.H

# data should be numpy array
def quantization(data, wl=16, fl=8):
    upper_bound = 2**(wl-fl-1)-2**(-fl)
    lower_bound = -2**(wl-fl-1)
    precision = 2**(-fl)
    value_q = np.clip(data,lower_bound,upper_bound)
    quotient = (value_q - lower_bound) // precision
    value_q = lower_bound + precision * quotient
    value_q = np.where((value_q < 0) & (value_q < data), value_q+precision, value_q)
    return value_q

if __name__ == "__main__":
    V = np.array([[1.0,0.0,2.0],[0.0,1.0,1.0]])
    print(V)
    W,H = low_rank_compression(V)
    V_approx = np.dot(W,H)
    print(V_approx)
