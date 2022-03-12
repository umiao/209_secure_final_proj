import random
import numpy as np
import torch

def partition(data_obj, setting, nums):
    data = data_obj.data.numpy()
    targets = data_obj.targets.numpy()

    ret_data = None
    ret_targets = []
    max_ratio = max(setting)
    amplify_r = max_ratio / 1
    # keep_prob = nums / 60000

    for i in range(len(targets)):
        # if random.random() < keep_prob:  # problematic now
        c = targets[i]
        if random.random() < setting[c] / amplify_r:
            ret_targets.append(targets[i])
            if ret_data is None:
                ret_data = data[i, :, :]
                ret_data = ret_data[np.newaxis, :]
            else:
                to_append = data[i, :, :]
                ret_data = np.concatenate([ret_data, to_append[np.newaxis, :]], axis=0)
                # preserve this class
        if len(ret_targets) >= nums:
            break

    data_obj.data = torch.tensor(ret_data)
    data_obj.targets = torch.tensor(ret_targets)
    return data_obj