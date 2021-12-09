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

        self.path_checkpoints = f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"
        self.path_logs = f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"
        self.path_visuals = f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"

    @property
    def path_checkpoints(self):
        return self._path_checkpoints

    @path_checkpoints.setter
    def path_checkpoints(self, path):
        makedir(path)
        self._path_checkpoints = path

    @property
    def path_logs(self):
        return self._path_logs

    @path_logs.setter
    def path_logs(self, path):
        makedir(path)
        self._path_logs = path

    @property
    def path_visuals(self):
        return self._path_visuals

    @path_visuals.setter
    def path_visuals(self, path):
        makedir(path)
        self._path_visuals = path