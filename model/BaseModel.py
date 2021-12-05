##################################################
# Base model which inherits all functionalities models used within this repository should have.
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import torch.nn as nn
from utils import makedir


class BaseModel(nn.Module):

    def __init__(self, dataset, model, experiment='default'):
        super(BaseModel, self).__init__()

        self.model = model
        self.dataset = dataset
        self.experiment = experiment

        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)

    @property
    def path_checkpoints(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"