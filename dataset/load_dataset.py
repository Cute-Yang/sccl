import numpy as np
class Dataset:
    def __init__(self, train_x1: np.ndarray, train_x2: np.ndarray, train_x3: np.ndarray):
        assert len(train_x1) == len(train_x2) == len(
            train_x3), "size not match between pairs"
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_x3 = train_x3
        self.size = len(train_x1)

    def __len__(self):
        return len(self.train_x1)

    def __getitem__(self, idx):
        rs = {
            "train_x1": self.train_x1[idx],
            "train_x2": self.train_x2[idx],
            "train_x3": self.train_x3[idx]
        }
        return rs

    def generate(self, batch_size: int, epochs: int = None, drop_remain=True):
        """
        Args:
            batch_size:size of each iter

            epochs:epoch number
            drop_remain:if last iter < batch_size:drop it
        """
        epoch = 0
        start, end = 0, 0
        if epochs is None:
            flag = True
        else:
            flag = (epoch < epochs)

        # number of batchs for a epoch
        iters, last_iter = self.size % batch_size
        batchs = iters
        if last_iter != 0:
            if drop_remain:
                batchs = iters
            else:
                batchs = iters+1

        while flag:
            for _ in range(batchs):
                yield self.train_x1[start:end], self.train_x2[start:end], self.train_x3[start:end]
                start = start+batch_size
                end = end+batch_size
            # after a epoch,set the ptr to ori
            start, end = 0, batch_size

