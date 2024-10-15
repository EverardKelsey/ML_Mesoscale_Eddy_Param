import torch
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import xarray as xr
import sys
import warnings
warnings.filterwarnings('ignore')
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class simpleANN(nn.Module):
    def __init__(
        self, 
        in_feat = 8, 
        out_feat = 5, 
        hidden_dim=[128, 128, 128],
        init_weights = False,
        weight_mean = None,
        weight_std = None
    ):
        super().__init__()
        self.init_weights = init_weights
        # if init_weights:
        self.weight_mean = weight_mean
        self.weight_std = weight_std
        	# self.apply(self._init_weights)
        if isinstance(hidden_dim, int):
            self.forwardsel = 0
            self.fc1 = nn.Linear(in_feat, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_feat)
            
        else:
            self.forwardsel = 1
            layers = []
            for i in range(len(hidden_dim) + 1):
                if i == 0:
                    layers.append(nn.Linear(in_feat, hidden_dim[0]))
                elif i == len(hidden_dim):
                    layers.append(nn.Linear(hidden_dim[-1],out_feat))
                else:
                    layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            self.layers = nn.Sequential(*layers)    

        
        
    def _init_weights_fun(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.weight_mean, std=self.weight_std)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        if self.forwardsel == 0:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i < len(self.layers)-1:
                    x = F.relu(x)
        return x

    def compute_loss(self, x, ytrue):
        MSE = nn.MSELoss()(self.forward(x),ytrue)
        return {'loss':MSE}

class ANN_wrapper():
    """
    This wrapper function takes xarray dataset (data) as an input and applies an ANN (model) to the xarray dataset. The output will have the same dimensions as the input, except with the appropirate number of target variables (called output). The input xarray dataset must have a dimension called 'variable' which stores the input variables to the ANN
    """
    def __init__(
        self,
        data, 
        model
    ):
        self.model = model
        self.result = xr.apply_ufunc(
            self.apply_ANN,                     # Function to apply
            data.to_array().compute(),       # Input xarray object
            input_core_dims=[['variable']],  # Dimensions that the function expects as input
            output_core_dims=[['output']],   # Dimensions that the function returns
            vectorize=True,                  # Automatically vectorize the function          
            output_dtypes=[np.float32]       # Data type of the output
        )

    def apply_ANN(self, data):
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(data, dtype=torch.float32)
            output_tensor = self.model(input_tensor)
            return output_tensor.numpy()

class ANNimport(nn.Module):
    def __init__(self, layer_sizes=[7, 17, 27, 5]):
        super().__init__()
        
        self.layer_sizes = layer_sizes

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers)-1:
                x = nn.functional.relu(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def compute_loss(self, x, ytrue):
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}