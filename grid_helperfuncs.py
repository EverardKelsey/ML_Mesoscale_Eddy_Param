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
sys.path.append('/home/kae10022/PythonScripts/')
import warnings
warnings.filterwarnings('ignore')
# from dask.diagnostics import ProgressBar
# import matplotlib.pyplot as plt
# import pickle
from tqdm import tqdm
from Data_Preparation import * 

def prepare_grid(ds, path, **kwargs):
    # if 'grid_name' in kwargs.keys():
    #     print("yes")
    #     ds_grid = xr.open_dataset(kwargs['grid_name'])
    # else:
        # ds_grid = xr.open_dataset('output/ocean_geometry.nc' )
    ds_grid = xr.open_dataset(f'{path}ocean_geometry.nc' )
    ds_grid = ds_grid.rename({"lath": "yh", "lonh": "xh", "latq": "yq", "lonq": "xq"})

    if "zi" in ds:
        ds_grid["zi"] = ds["zi"]
    ds_grid["zl"] = ds["zl"]
    ds_grid["Time"] = ds["Time"]  # weirdly, need to add the time as a coord to the grid, otherwise time coord is lost after applying an xgcm operation
    grid = make_grid(ds_grid, symmetric=True, **kwargs)
   
    ds_grid_nonsym = ds_grid.isel(xq = slice(1,None), yq=slice(1,None))
    grid_nonsym = make_grid(ds_grid_nonsym, symmetric=False, **kwargs)

    return grid_nonsym, ds_grid_nonsym, grid, ds_grid

def make_grid(ds_grid, symmetric=True, include_time=True):
    
    if "zi" in ds_grid:
        z_coords = {'Z': {'center': 'zl', 'outer': 'zi'}}
    else:
        z_coords = {'Z': {'center': 'zl'}}

    if symmetric:
        coords = {
                'X': {'center': 'xh', 'outer': 'xq'},
                'Y': {'center': 'yh', 'outer': 'yq'}
                }
    else:
        coords = {
                'X': {'center': 'xh', 'right': 'xq'},
                'Y': {'center': 'yh', 'right': 'yq'}
                }
    if include_time:
    
        t_coords = {'Time': {'center': 'Time'}}
    else:
        t_coords = {}
        
    coords = {**coords, **z_coords, **t_coords}

    metrics = {
        ('X',):['dxCu','dxCv','dxT','dxBu'],
        ('Y',):['dyCu','dyCv','dyT','dyBu'],
    }

    grid = Grid(ds_grid, coords=coords, metrics=metrics)
    return grid

def grid_scale(ds_grid):
    Del2 = 2 * ds_grid.dxT**2 * ds_grid.dyT**2 / (ds_grid.dxT**2 + ds_grid.dyT**2)
    return Del2

def Coriolis_compute(grid, ds_grid):
    f = grid.interp(grid.interp(ds_grid.f,'X'),'Y')
    return f