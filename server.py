from model import Net


class client:
    local_model = None
    local_data_set = None
    batch_size = 32


    def __init__(self):
        self.local_model = Net()
        return


    def update_to_global_model(self):
        return

    def compute_gradient(self):
        return


