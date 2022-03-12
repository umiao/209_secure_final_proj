from model import Net


class client:
    local_model = None
    local_data_set = None
    batch_size = 32


    def __init__(self, retrieve_history=True, params=None):
        self.local_model = Net(retrieve_history = retrieve_history, params = params)
        return


    def update_to_global_model(self):
        return

    def compute_gradient(self):
        return self.local_model.train_single_batch()


