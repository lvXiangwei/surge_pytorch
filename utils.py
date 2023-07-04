import random
import numpy as np
import torch

EOS = 1e-10
def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
   
    
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Config(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

class EarlyStopping:
    def __init__(self, patience=10, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.path = path

    def step(self, auc, model, epoch):
        score = auc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        es_str = f'{self.counter:02d}/{self.patience:02d} | BestVal={self.best_score:.4f}@E{self.best_epoch}'
        return self.early_stop, es_str

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), self.path)


def normalize(adj, mode="sym"):
    degrees = torch.sum(adj, dim=-1)
    D_inv_sqrt = torch.diag_embed(torch.pow(degrees, -0.5))
    norm_adj = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
    return norm_adj