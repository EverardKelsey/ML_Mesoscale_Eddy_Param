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
import xrft
sys.path.append('/home/kae10022/PythonScripts/')
import warnings
warnings.filterwarnings('ignore')
# from dask.diagnostics import ProgressBar
# import matplotlib.pyplot as plt
# import pickle
from tqdm import tqdm
from Data_classes import * 
from ANN_classes import * 

################ Import and export ANN functions  ################
def import_ANN(filename):
    ds = xr.open_dataset(filename)
    layer_sizes = ds['layer_sizes'].values

    ann = ANNimport(layer_sizes)
    
    for i in range(len(ann.layers)):
        # Naming convention for weights and dimensions
        matrix = torch.tensor(ds[f'A{i}'].T.values) # Transpose back: it is convention
        bias = torch.tensor(ds[f'b{i}'].values)
        ann.layers[i].weight.data = matrix
        ann.layers[i].bias.data = bias
        
    x_test = torch.tensor(ds['x_test'].values.reshape(1,-1))
    y_pred = ann(x_test).detach()
    y_test = ds['y_test'].values
    
    rel_error = float(np.abs(y_pred - y_test).max() / np.abs(y_test).max())
    #if rel_error > 1e-6:
    print(f'Test prediction using {filename}: {rel_error}')
    return ann

def export_ANN_KAE(ann, input_norms=None, output_norms=None, filename='ANN_test.nc'):
    ds = xr.Dataset()

    #ds['num_layers'] = xr.DataArray(len(ann.layer_sizes)).expand_dims('dummy_dimension') #Pavel's
    ds['num_layers'] = xr.DataArray(len(ann.layers)+1).expand_dims('dummy_dimension')
    #ds['layer_sizes'] = xr.DataArray(ann.layer_sizes, dims=['nlayers']) #Pavel's
    ds['layer_sizes'] = xr.DataArray(np.ones(len(ann.layers)+1), dims=['nlayers'])
    ds = ds.astype('int32') # MOM6 reads only int32 numbers
    
    ds.layer_sizes[0] = ann.layers[0].in_features
    for i in range(len(ann.layers)):
        # Naming convention for weights and dimensions
        matrix = f'A{i}'
        bias = f'b{i}'
        ncol = f'ncol{i}'
        nrow = f'nrow{i}'
        layer = ann.layers[i]

        ds.layer_sizes[i+1] = ann.layers[i].out_features
        
        # Transposed, because torch is row-major, while Fortran is column-major
        ds[matrix] = xr.DataArray(layer.weight.data.T, dims=[ncol, nrow])
        ds[bias] = xr.DataArray(layer.bias.data, dims=[nrow])

    
    # Save true answer for random vector for testing
    
    #x0 = torch.randn(ds.layer_sizes[0]) #Pavel's
    x0 = torch.randn(ann.layers[0].in_features)
    if input_norms is None or output_norms is None:
        input_norms = torch.ones(ann.layers[0].in_features)
        output_norms = torch.ones(ann.layers[-1].out_features)
        
    y0 = ann(x0 / input_norms) * output_norms
    nrow = f'nrow{len(ann.layers)-1}'
    
    ds['x_test'] = xr.DataArray(x0.data, dims=['ncol0'])
    ds['y_test'] = xr.DataArray(y0.data, dims=[nrow])
    
    ds['input_norms']  = xr.DataArray(input_norms.data, dims=['ncol0'])
    ds['output_norms'] = xr.DataArray(output_norms.data, dims=[nrow])

    
    # print('x_test = ', ds['x_test'].data)
    # print('y_test = ', ds['y_test'].data)
    
    if os.path.exists(filename):
        print(f'Rewrite {filename} ?')
        input()
        os.system(f'rm -f {filename}')
        print(f'{filename} is rewritten')
    
    ds.to_netcdf(filename)

################ Functions for preparing to train and training a model  ################
def define_train_val_test_files(files, 
                                train_frac = 0.8, 
                                val_frac = 0.1, 
                                test_frac = 0.1, 
                                shuffle_files = True, #should we shuffle files in time?
                                myseed = 123456789 #the seed to set to ensure file shuffling is reproducible
                               ):
    """
    Function which partitions out the number of snapshots used for testing and training. Saves a yaml file containing this information
    """
    totSamples = len(files)
    Ntrain = int(np.round(train_frac * totSamples))
    Nval = int(np.round(val_frac * totSamples))
    Ntest = totSamples - Ntrain - Nval

    if shuffle_files:
        files.sort() #sort the files first to ensure reproducible results
        rng = np.random.default_rng(myseed) #set the generator with given seed
        rng.shuffle(files)

    train_files = files[0:Ntrain]
    val_files = files[Ntrain:(Ntrain+Nval)]
    test_files = files[(Ntrain+Nval):]

    return {
        'train_files' : train_files,
        'val_files' : val_files,
        'test_files' : test_files,
        'shuffled' : shuffle_files,
        'set_seed' : myseed
    }

def validation_loss(NET, ds_val, **train_config):
    """
    Calculates the validation loss. For now, only works for training momentum flux and dual form stress components of EPF
    """
    device = train_config['device']
    criterion = nn.MSELoss()
    valMSE_loss = 0.
    valLEN = 0
    inputs = torch.squeeze(ds_val.x_data).to(device)
    targets = torch.squeeze(ds_val.y_data).to(device)
    unorm = torch.squeeze(ds_val.modu).to(device)
    bnorm = torch.squeeze(ds_val.modb).to(device)
    f = torch.squeeze(ds_val.f).to(device)
    gp = torch.squeeze(ds_val.gprime).to(device)
    gridsq = torch.squeeze(ds_val.gridsq).to(device)
    std_y = torch.tensor(train_config["std_reality"]).to(device)

    out = NET(inputs)
    if train_config["dimensional_loss"]:
        pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
        pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4)], dim=1)
        pred = torch.cat((pred_mom, pred_buoy), dim=1)
    else:
        pred_mom = out/std_y
            
    reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(train_config["std_reality"]))], dim=1)
    valMSE=criterion(pred, reality)
    valMSE_loss = valMSE.item()
    return valMSE_loss

def train_ANN(NET, ds_train, ds_val, 
              num_epochs = 100,
              learning_rate = 1e-2,
              batch_size = 2**16,
              std_reality = None,
              dimensional_loss = True,
              device = None,
              set_seed = True,
              myseed = 123456789, 
              log_epoch_info = False
             ):
    #### creating dictionary to hold all training info #### 
    train_dict = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'std_reality': std_reality,
        'dimensional_loss': dimensional_loss,
        'device': device,
        'set_seed': set_seed,
        'myseed': myseed,
        'log_epoch_info': log_epoch_info
    }
    #### Setting up the batches ####
    if set_seed:
        rng = np.random.default_rng(myseed)
        idx = np.arange(0,len(ds_train),1)
        rng.shuffle(idx)
        idx = torch.tensor(idx)
    else:
        idx=torch.randperm(len(ds_train))
    batches=[]
    for aa in range(0,len(ds_train),batch_size):
        batches.append(idx[aa:aa+batch_size])

    #### Setting the optimiser ####
    optimizer = optim.Adam(NET.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    log = {'MSE': [], 'valMSE': []}
    if log_epoch_info:
        log_epoch = {'MSE':[]}

    for epoch in tqdm(range(num_epochs)):
        MSE_loss = 0.
        LEN = 0.
        NET.train()

        for i, batch_idx in enumerate(batches):
            inputs, targets, unorm, bnorm, gridsq, f, gprime = ds_train[batch_idx]
            #forward pass
            inputs = torch.squeeze(inputs).to(device)
            targets = torch.squeeze(targets).to(device)
            unorm = torch.squeeze(unorm).to(device)
            bnorm = torch.squeeze(bnorm).to(device)
            gridsq = torch.squeeze(gridsq).to(device)
            f = torch.squeeze(f).to(device)
            gp = torch.squeeze(gprime).to(device)
            std_y = torch.tensor(std_reality).to(device)

            out = NET(inputs) #prediction
            if dimensional_loss:
                pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
                pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4)], dim=1)
                pred = torch.cat((pred_mom, pred_buoy), dim=1)
            else:
                pred = out / std_y
            reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(std_reality))], dim=1)
            MSE=criterion(pred, reality)
            #log_epoch['MSE'].append(MSE.item())
        
            # Backward and optimize
            optimizer.zero_grad()
            MSE.backward()
            optimizer.step()
            if log_epoch_info:
                log_epoch['MSE'].append(MSE.item())
            MSE_loss += MSE.item() * len(inputs)
            LEN += len(inputs)

        MSE_loss = MSE_loss / LEN
        NET.eval()
        with torch.no_grad():
            val_loss = validation_loss(NET, ds_val, **train_dict)

        log['MSE'].append(MSE_loss)
        log['valMSE'].append(val_loss)
    
        ## Print loss after every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {MSE_loss:.6f}, Validation Loss: {val_loss:.6f}")
    if log_epoch_info:
        return NET, log, log_epoch
    else:
        return NET, log
    





################ Functions for evaluating the skill of a model  ################
def build_prediction(NET, testdata, data_info):
    """
    Function which builds predictions for full EPF components from test data and an ANN
    """
    NET.eval()
    test_input = xr.Dataset()
    test_input['vort'] = testdata.vort #/ testdata.unorm
    test_input['stretch'] = testdata.stretch #/ testdata.unorm
    test_input['strain'] = testdata.strain #/ testdata.unorm

    if data_info['ml_data_choices']['buoy_inputs']['type'] == 'intfc_slopes':
        test_input['extop'] = testdata.extop #/ testdata.bnorm
        test_input['eytop'] = testdata.eytop #/ testdata.bnorm
        if data_info['ml_data_choices']['buoy_inputs']['method'] == 'components':
            test_input['exbot_avg'] = testdata.exbot_avg #/ testdata.bnorm
            test_input['eybot_avg'] = testdata.eybot_avg #/ testdata.bnorm
            test_input['exbot_dev'] = testdata.exbot_dev #/ testdata.bnorm
            test_input['eybot_dev'] = testdata.eybot_dev #/ testdata.bnorm
        else:
            test_input['exbot'] = testdata.exbot #/ testdata.bnorm
            test_input['eybot'] = testdata.eybot #/ testdata.bnorm
    else:
        print("sorry, comparison for thickness slopes as input not implemented yet")
    NETtest = ANN_wrapper(test_input, NET)
    Pred = xr.Dataset()
    Pred['Ruu'] = NETtest.result.isel(output = 0) * testdata.Delsq * testdata.unorm**2
    Pred['Ruv'] = NETtest.result.isel(output = 1) * testdata.Delsq * testdata.unorm**2
    Pred['Rvv'] = NETtest.result.isel(output = 2) * testdata.Delsq * testdata.unorm**2
    Pred['Formx'] = NETtest.result.isel(output = 3) * testdata.Delsq * testdata.unorm * testdata.f * testdata.bnorm
    Pred['Formy'] = NETtest.result.isel(output = 4) * testdata.Delsq * testdata.unorm * testdata.f * testdata.bnorm
    # dim_Pred = Pred * testdata.Delsq * testdata.momnorm**2
    return Pred
    
def R2(true, pred, var, dims=['Time', 'xh', 'yh', 'zl']):
    """
    Calculate the coefficient of determination (R-squared) between true and predicted values for a given variable.

    Parameters:
    - true (xr.DataArray): The true values.
    - pred (xr.DataArray): The predicted values.
    - var (str): The variable to calculate R-squared for.
    - dims (list): List of dimensions to average over (default: ['Time', 'xh', 'yh', 'zl']).

    Returns:
    - float: The R-squared value.
    """
    RSS = ((pred[var] - true[var]) ** 2).mean(dims)
    TSS = ((true[var]) ** 2).mean(dims)

    R2 = 1 - RSS / TSS

    return R2

def SGS_skill(pred, truth):
    '''
    This function computes:
    * 2D map of R-squared
    * 2D map of SGS dissipation
    * Power and energy transfer spectra
    in a few regions
    '''
    # grid = truth.grid
    # param = self.param
    SGSx = truth.y.Ruu
    SGSy = truth.y.Rvv
    ZB20u = pred.Ruu
    ZB20v = pred.Rvv

    ############# R-squared and correlation ##############
    # Here we define second moments
    def M2(x,y=None,centered=False,dims=None,exclude_dims='zl'):
        if dims is None and exclude_dims is not None:
            dims = []
            for dim in x.dims:
                if dim not in exclude_dims:
                    dims.append(dim)

        if y is None:
            y = x
        if centered:
            return (x*y).mean(dims) - x.mean(dims)*y.mean(dims)
        else:
            return (x*y).mean(dims)

    def M2u(x,y=None,centered=False,dims='Time'):
        return M2(x,y,centered,dims)
    def M2v(x,y=None,centered=False,dims='Time'):
        return M2(x,y,centered,dims)
            
    errx = SGSx - ZB20u
    erry = SGSy - ZB20v

    skill = xr.Dataset()
    ######## Simplest statistics ##########
    skill['SGSx_mean'] = SGSx.mean('Time')
    skill['SGSy_mean'] = SGSy.mean('Time')
    skill['ZB20u_mean'] = ZB20u.mean('Time')
    skill['ZB20v_mean'] = ZB20v.mean('Time')
    skill['SGSx_std']  = SGSx.std('Time')
    skill['SGSy_std']  = SGSy.std('Time')
    skill['ZB20u_std'] = ZB20u.std('Time')
    skill['ZB20v_std'] = ZB20v.std('Time')

    # These metrics are same as in GZ21 work
    # Note: eveything is uncentered
    skill['R2u_map'] = 1 - M2u(errx) / M2u(SGSx)
    skill['R2v_map'] = 1 - M2v(erry) / M2v(SGSy)
    skill['R2_map']  = 1 - (M2u(errx) + M2v(erry)) / (M2u(SGSx) + M2v(SGSy))

    # Here everything is centered according to definition of correlation
    skill['corru_map'] = M2u(SGSx,ZB20u,centered=True) / np.sqrt(M2u(SGSx,centered=True) * M2u(ZB20u,centered=True))
    skill['corrv_map'] = M2v(SGSy,ZB20v,centered=True) / np.sqrt(M2v(SGSy,centered=True) * M2v(ZB20v,centered=True))
    # It is complicated to derive a single true formula, so use simplest one
    skill['corr_map']  = (skill['corru_map'] + skill['corrv_map']) * 0.5

    # ########### Global metrics ############
    skill['R2u'] = 1 - M2(errx) / M2(SGSx)
    skill['R2v'] = 1 - M2(erry) / M2(SGSy)
    skill['R2'] = 1 - (M2(errx) + M2(erry)) / (M2(SGSx) + M2(SGSy))
    skill['corru'] = M2(SGSx,ZB20u,centered=True) \
            / np.sqrt(M2(SGSx,centered=True) * M2(ZB20u,centered=True))
    skill['corrv'] = M2(SGSy,ZB20v,centered=True) \
            / np.sqrt(M2(SGSy,centered=True) * M2(ZB20v,centered=True))
    skill['corr'] = (skill['corru'] + skill['corrv']) * 0.5
    skill['opt_scaling'] = (M2(SGSx,ZB20u) + M2(SGSy,ZB20v)) / (M2(ZB20u) + M2(ZB20v))

    # ############### Spectral analysis ##################
    # for region in ['NA', 'Pacific', 'Equator', 'ACC']:
    #     transfer, power, KE_spec, power_time, KE_time = self.state.transfer(SGSx, SGSy, region=region, additional_spectra=True)
    #     skill['transfer_'+region] = transfer.rename({'freq_r': 'freq_r_'+region})
    #     skill['power_'+region] = power.rename({'freq_r': 'freq_r_'+region})
    #     skill['KE_spec_'+region] = KE_spec.rename({'freq_r': 'freq_r_t'+region})
    #     skill['power_time_'+region] = power_time
    #     skill['KE_time_'+region] = KE_time
    #     transfer, power, KE_spec, power_time, KE_time = self.state.transfer(ZB20u, ZB20v, region=region, additional_spectra=True)
    #     skill['transfer_ZB_'+region] = transfer.rename({'freq_r': 'freq_r_'+region})
    #     skill['power_ZB_'+region] = power.rename({'freq_r': 'freq_r_'+region})
    #     skill['power_time_ZB_'+region] = power_time
    return skill.compute()

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
    
def globalR2(testdata, Pred, val, level):
    if R2(testdata.y,Pred,val,{'Time', 'yh','xh'}).isel(zl=level).values < 0:
        return 0.0
    else:
        return trunc(R2(testdata.y,Pred,val,{'Time', 'yh','xh'}).isel(zl=level).values,2)


def compute_isotropic_cospectrum(u_in, v_in, fu_in, fv_in, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=False, detrend='linear', window_correction=True):
    # Interpolate to the center of the cells
    u = u_in
    v = v_in
    fu = fu_in
    fv = fv_in

    # Select desired Lon-Lat square
    u = select_LatLon(u,Lat,Lon)
    v = select_LatLon(v,Lat,Lon)
    fu = select_LatLon(fu,Lat,Lon)
    fv = select_LatLon(fv,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(u.xh))
    y = dy*np.arange(len(u.yh))
    for variable in [u, v, fu, fv]:
        variable['xh'] = x
        variable['yh'] = y

    Eu = xrft.isotropic_cross_spectrum(u, fu, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_cross_spectrum(v, fv, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev)
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    
    return np.real(E)

def select_LatLon(array, Lat=(35,45), Lon=(5,15)):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    return array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq' or 'xh'
    as xarray
    '''
    try:
        coord = array.xq
    except:
        coord = array.xh
    return coord

def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh'
    as xarray
    '''
    try:
        coord = array.yq
    except:
        coord = array.yh
    return coord   

# ############### Most updated version of snapshot creation  ################
# class create_MLdataset(Dataset):
#     """
#     This class takes a list of single time snapshots (processed DG data), and compiles one large dataset (masking etc already done) that will then be sent to a torch dataset (eventually). It is assumed that Coriolis information already exists in the snapshot files being loaded. 
#     """
#     def __init__(self, data_files,  coarsen_fac = None, gprime = [0.98, 0.0098], one_data_mask = True, input_list = ['vort','stretch','strain','intfc_slopes'], output_list = ['Ruu','Ruv','Rvv','Formx','Formy'], no_reshape = False):
#         super().__init__()
#         self.data_files = data_files
#         self.inputs = input_list
#         self.outputs = output_list
#         print("opening all snapshot files as one xarray dataset")
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         x = xr.Dataset()
#         y = xr.Dataset()
#         print("Defining the data mask (based off of land values)")
#         if one_data_mask:
#             # if using one data mask, select the mask from the bottom-most layer
#             hmask = full_data.data_mask.isel(zl=-1).drop('zl')
#             hmask = hmask.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#             hmask["zl"] = full_data.Del_sq["zl"]
#         else:
#             hmask = full_data.data_mask

#         print("Defining the velocity gradient inputs")
#         if ('vort' in self.inputs) & ('stretch' in self.inputs) & ('strain' in self.inputs):
#             x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
#         else:
#             print('no alternative velocity inputs for now... assuming you want to train on velocity gradients, and so over-riding your choice to not include')
#             x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
            
            
#         if 'intfc_slopes' in self.inputs:
#             print("Defining the interface gradient inputs")
#             x['extop'], x['eytop'], x['exbot'], x['eybot'], self.buoynorm = self._egrad_input_comp(full_data*hmask)
#         elif 'hbar_slopes' in self.inputs:
#             print("Defining the thickness gradient inputs")
#             x['hx'], x['hy'], self.buoynorm = self._hgrad_input_comp(full_data*hmask)    
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms

#         for var in output_list:
#             y[var] = full_data[var] * hmask
            
#         Coriolis = full_data.Coriolis
#         self.Coriolis = Coriolis
        
#         self.Delsq = full_data.Del_sq * hmask  
#         self.hbar = full_data.hbar_coarse * hmask
#         zlevels = Coriolis["zl"]
#         gp = xr.DataArray(
#             data=gprime,
#             dims = ["zl"],
#             coords=dict(
#                 zl=("zl", Coriolis["zl"].data)
#             )
#         )
#         gp = gp.expand_dims(dim = {'yh': len(full_data.Del_sq['yh']),
#                               'xh': len(full_data.Del_sq['xh'])})
#         gp['yh'] = full_data.Del_sq["yh"]
#         gp['xh'] = full_data.Del_sq["xh"]
#         self.gprime = gp

#         data = xr.merge([x, y])
#         data['modu'] = self.momnorm
#         data['modb'] = self.buoynorm
#         data['Delsq'] = self.Delsq
#         data['f'] = self.Coriolis
#         data['gprime'] = self.gprime
#         if no_reshape:
#             self.mydata = data
#         else:
#             print("sending ds to numpy array for reshaping")
#             ds_np = data.to_array().to_numpy()
    
#             dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
#             dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
#             dimZ = np.shape(ds_np)[-3]   # Points in Z axis (layer)
#             dimT = np.shape(ds_np)[-4]   # Points in T axis (snapshot)
#             dimF = np.shape(ds_np)[0]   # total number of features/needed data in dataset
    
#             print("reshaping numpy array")
#             ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY * dimZ * dimT).transpose()
    
#             print("Now masking all 0 and nan samples")
#             ds_np_reshape[ds_np_reshape == 0.] = np.nan
#             nansum = np.sum(ds_np_reshape, axis = 1)
#             nan_inds = np.isnan(nansum)
    
#             masked_data = ds_np_reshape[~nan_inds]
    
#             print("Normalising the input datasets")
#             x_mom = torch.tensor(masked_data[:,0:3] / masked_data[:,-5][:,np.newaxis], dtype = torch.float32)
#             if 'intfc_slopes' in self.inputs:
#                 x_buoy = torch.tensor(masked_data[:,3:7] / masked_data[:,-4][:,np.newaxis], dtype = torch.float32)
#                 N_input = 7
#             elif 'hbar_slopes' in self.inputs:
#                 x_buoy = torch.tensor(masked_data[:,3:5] / masked_data[:,-4][:,np.newaxis], dtype = torch.float32)
#                 N_input = 5
#             print("Creating all needed torch tensors")
#             x_out = torch.cat((x_mom, x_buoy), dim=1)
#             y_out = torch.tensor(masked_data[:,N_input:(N_input+len(output_list))], dtype = torch.float32)
#             unorm = torch.tensor(masked_data[:,-5], dtype = torch.float32)
#             bnorm = torch.tensor(masked_data[:,-4], dtype = torch.float32)
#             gridsq = torch.tensor(masked_data[:,-3], dtype = torch.float32)
#             f = torch.tensor(masked_data[:,-2], dtype = torch.float32)
#             gprime = torch.tensor(masked_data[:,-1], dtype = torch.float32)
    
#             print("Saving all torch tensors as a list of tensors in self.mydata")
#             self.mydata = torch.utils.data.TensorDataset(x_out, y_out, unorm, bnorm, gridsq, f, gprime)

#     def _mom_input_comp(self, data):
#         """
#         Compute the normalisation for the momentum inputs
#         """
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _egrad_input_comp(self, data):
#         """
#         Compute the normalisation for the buoyancy inputs
#         """
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm

#     def _hgrad_input_comp(self, data):
#         """
#         Compute the normalisation for the buoyancy inputs (when using thickness gradients)
#         """
#         hx = data.hbarx
#         hy = data.hbary
#         norm = np.sqrt(hx**2 + hy**2)
#         return hx, hy, norm


# def define_train_val_test_files(files, train_frac = 0.8, val_frac = 0.1, test_frac = 0.1):
#     """
#     Function which partitions out the number of snapshots used for testing and training. Saves a yaml file containing this information
#     """
#     totSamples = len(files)
#     Ntrain = int(np.round(train_frac * totSamples))
#     Nval = int(np.round(val_frac * totSamples))
#     Ntest = totSamples - Ntrain - Nval

#     np.random.shuffle(files)
#     # print(files)
#     train_files = files[0:Ntrain]
#     val_files = files[Ntrain:(Ntrain+Nval)]
#     test_files = files[(Ntrain+Nval):]

#     return {
#         'train_files' : train_files,
#         'val_files' : val_files,
#         'test_files' :test_files
#     }

# class MLDataset(Dataset):
#     """
#     Dataset for use in training, validation, and testing of ANNs. Should be agnostic to number of input and output channels.
#     Sensitive to normalisation data defined, however. 
#     """
#     def __init__(self,data_tensor):
#         """
#         tensor containing data - assume that the inputs are already normalised
#         """
        
#         super().__init__()
#         self.x_data=data_tensor[0]
#         self.y_data=data_tensor[1]
#         self.modu = data_tensor[2]
#         self.modb = data_tensor[3]
#         self.gridsq = data_tensor[4]
#         self.f = data_tensor[5]
#         self.gprime = data_tensor[6]
#         # self.batch_size = batch_size
#         self.len = len(self.x_data)

#     def __len__(self):
#         return len(self.x_data)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         return self.x_data[idx], self.y_data[idx], self.modu[idx], self.modb[idx], self.gridsq[idx], self.f[idx], self.gprime[idx]
#     # def __getitem__(self, idx):
#     #     """ Return elements at each index specified by idx."""
#     #     idxs = np.arange((self.batch_size*idx),self.batch_size*(idx+1),1)
#     #     #try:
#     #     return self.x_data[idxs], self.y_data[idxs], self.modu[idxs], self.modb[idxs], self.gridsq[idxs], self.f[idxs], self.gprime[idxs]

#  # num_epochs = 100, learning_rate = 1e-3, batch_size = 2**10,
#  #                std_reality = [0.00176617, 0.00081482, 0.00218827, 5.0140246e-05, 3.0815412e-05], dimensional_loss = True
# def train_ANN(NET, ds_train, ds_val, device, **train_config):
#     #preparing the batches based off of batch size
#     num_epochs = train_config['num_epochs']
#     idx=torch.randperm(len(ds_train))
#     batches=[]
#     for aa in range(0,len(ds_train),train_config["batch_size"]):
#         batches.append(idx[aa:aa+train_config["batch_size"]])

#     optimizer = optim.Adam(NET.parameters(), lr=train_config["learning_rate"])
#     criterion=nn.MSELoss()
#     log = {'MSE': [], 'valMSE': []}
#     log_epoch = {'MSE': []}
#     for epoch in tqdm(range(train_config["num_epochs"])):
#         MSE_loss = 0.
#         LEN = 0
#         NET.train()

#         for i, batch_idx in enumerate(batches):
#             inputs, targets, unorm, bnorm, gridsq, f, gprime = ds_train[batch_idx]
#             # Forward pass
#             inputs = torch.squeeze(inputs).to(device)
#             targets = torch.squeeze(targets).to(device)
#             unorm = torch.squeeze(unorm).to(device)
#             bnorm = torch.squeeze(bnorm).to(device)
#             gridsq = torch.squeeze(gridsq).to(device)
#             f = torch.squeeze(f).to(device)
#             gp = torch.squeeze(gprime).to(device)
#             std_y = torch.tensor(train_config["std_reality"]).to(device)
            

#             out = NET(inputs)
#             if train_config['dimensional_loss']:
#                 pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
#                 pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4)], dim=1)
#                 pred = torch.cat((pred_mom, pred_buoy), dim=1)
#             else:
#                 pred = out / std_y
                
#             reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(train_config["std_reality"]))], dim=1)
            
#             MSE=criterion(pred, reality)
#             # if i%10==0:
#             #     #print(MSE.item())
#             #     perdone = (i/len(datloader))*100
#             #     print(f'{perdone}% done with epoch')
#             log_epoch['MSE'].append(MSE.item())
        
#             # Backward and optimize
#             optimizer.zero_grad()
#             MSE.backward()
#             optimizer.step()
#             log_epoch['MSE'].append(MSE.item())
#             MSE_loss += MSE.item() * len(inputs)
#             LEN += len(inputs)

#         MSE_loss = MSE_loss / LEN
#         NET.eval()
#         with torch.no_grad():
#             val_loss = validation_loss(NET, ds_val, device, **train_config)

#         log['MSE'].append(MSE_loss)
#         log['valMSE'].append(val_loss)
    
#         ## Print loss after every epoch
#         print(f"Epoch [{epoch+1}/{train_config['num_epochs']}], Loss: {MSE_loss:.6f}, Validation Loss: {val_loss:.6f}")
#     return NET, log, log_epoch
    

# def validation_loss(NET, ds_val, device, **train_config):
#     """
#     Computes the validation loss. 
#         -- dimensional_loss : indicates whether loss is calculated in physical dimensions (True) or if it will be done non-dimensionally (False). I.e., whether the non-dimensional ANN output is re-dimensionalised prior to calculating loss (comparison with targets) or whether the targets are non-dimensionalised prior to calculating loss (against non-dimensional ANN output). 
#     """
    
#     criterion=nn.MSELoss()
#     valMSE_loss = 0.
#     valLEN = 0
#     inputs = torch.squeeze(ds_val.x_data).to(device)
#     targets = torch.squeeze(ds_val.y_data).to(device)
#     unorm = torch.squeeze(ds_val.modu).to(device)
#     bnorm = torch.squeeze(ds_val.modb).to(device)
#     f = torch.squeeze(ds_val.f).to(device)
#     gp = torch.squeeze(ds_val.gprime).to(device)
#     gridsq = torch.squeeze(ds_val.gridsq).to(device)
#     std_y = torch.tensor(train_config["std_reality"]).to(device)

#     out = NET(inputs)
#     if train_config["dimensional_loss"]:
#         pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
#         pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4)], dim=1)
#         pred = torch.cat((pred_mom, pred_buoy), dim=1)
#     else:
#         pred_mom = out/std_y
            
#     reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(train_config["std_reality"]))], dim=1)
#     valMSE=criterion(pred, reality)
#     valMSE_loss = valMSE.item()
#     return valMSE_loss

    
    
# ############### Functions to use when not predicting sub-filter potential energy #######################
# class SnapshotDataset_no_subPE(Dataset):
#     def __init__(self, data_files, get_Coriolis = False, coarsen_fac = None, gprime = [0.98, 0.0098], one_data_mask = True):
#         super().__init__()
        
#         self.data_files = data_files
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         if get_Coriolis: #if the dataset doesn't already have Coriolis term defined, create it
#             exp = 'R32'
#             path = '/scratch/nl2631/mom6/double_gyre/'
#             snap_name = 'snap_prog__'
#             files = glob.glob(f"{path}{exp}/{snap_name}*") #list of all years of data to loop through
#             mydata = DoubleGyre_Snapshots(files[0], coarsen_fac = coarsen_fac, gen_path = path, exp = exp)
#             Coriolis = mydata.coarse_grid_nonsym.interp(mydata.coarse_grid_nonsym.interp(mydata.ds_coarse_grid_nonsym.Coriolis,'X'),'Y')
            
#             del mydata
#             del files
#         else:
#             Coriolis = full_data.Coriolis
            
#         self.Coriolis = Coriolis
        
#         x = xr.Dataset()
#         y = xr.Dataset()
#         if one_data_mask:
#             hmask = full_data.data_mask.isel(zl=-1).drop('zl')
#             hmask = hmask.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#             hmask["zl"] = full_data.Del_sq["zl"]
#         else:
#             hmask = full_data.data_mask
            
#         x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
#         x['extop'], x['eytop'], x['exbot'], x['eybot'], self.buoynorm = self._buoy_input_comp(full_data*hmask)
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms
        
#         y['Ruu'] = full_data.Ruu * hmask
#         y['Ruv'] = full_data.Ruv * hmask
#         y['Rvv'] = full_data.Rvv * hmask
#         y['Formx'] = full_data.Formx * hmask
#         y['Formy'] = full_data.Formy * hmask
#         self.Delsq = full_data.Del_sq * hmask  
#         # self.subPEold = full_data.subPE * hmask
#         self.hbar = full_data.hbar_coarse * hmask

#         zlevels = Coriolis["zl"]
#         gp = xr.DataArray(
#             data=gprime,
#             dims = ["zl"],
#             coords=dict(
#                 zl=("zl", Coriolis["zl"].data)
#             )
#         )
#         gp = gp.expand_dims(dim = {'yh': len(full_data.Del_sq['yh']),
#                               'xh': len(full_data.Del_sq['xh'])})
#         gp['yh'] = full_data.Del_sq["yh"]
#         gp['xh'] = full_data.Del_sq["xh"]
#         self.gprime = gp
        
#         self.x = x #torch.tensor(ds_np_reshape[:,:,0:3], dtype = torch.float32)
#         self.y = y #torch.tensor(ds_np_reshape[:,:,3:6], dtype = torch.float32)
        
#     def _mom_input_comp(self, data):
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _buoy_input_comp(self, data):
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm
        
#     def __len__(self):
#         return self.x.sizes['Time']

#     def __getitem__(self,idx):
#         xdat = self.x.isel(Time = idx)
#         ydat = self.y.isel(Time = idx)
#         modu = self.momnorm.isel(Time = idx)
#         modb = self.buoynorm.isel(Time = idx)
#         Delsq = self.Delsq.isel(Time = idx)
#         gprime = self.gprime
#         data = xr.merge([xdat,ydat])
#         data['modu'] = modu
#         data['modb'] = modb
#         data['Delsq'] = Delsq
#         data['f'] = self.Coriolis.isel(Time = idx)
#         data['gprime'] = gprime
#         ds_np = data.to_array().to_numpy()
    
#         dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
#         dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
#         dimZ = np.shape(ds_np)[-3]   # Points in Z axis (layer)
#         dimF = np.shape(ds_np)[0]   # total number of features in dataset
    
#         ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY * dimZ).transpose()

#         ds_np_reshape[ds_np_reshape == 0.] = np.nan
#         nansum = np.sum(ds_np_reshape, axis = 1)
#         nan_inds = np.isnan(nansum)

#         masked_data = ds_np_reshape[~nan_inds]

#         x_mom = torch.tensor(masked_data[:,0:3] / masked_data[:,12][:,np.newaxis], dtype = torch.float32)
#         x_buoy = torch.tensor(masked_data[:,3:7] / masked_data[:,13][:,np.newaxis], dtype = torch.float32)
#         x_out = torch.cat((x_mom, x_buoy), dim=1)
#         y_out = torch.tensor(masked_data[:,7:12], dtype = torch.float32)
#         unorm = torch.tensor(masked_data[:,12], dtype = torch.float32)
#         bnorm = torch.tensor(masked_data[:,13], dtype = torch.float32)
#         gridsq = torch.tensor(masked_data[:,14], dtype = torch.float32)
#         f = torch.tensor(masked_data[:,15], dtype = torch.float32)
#         gprime = torch.tensor(masked_data[:,16], dtype = torch.float32)
        
#         return x_out, y_out, unorm, bnorm, gridsq, f, gprime

# def validation_loss_no_subPE(NET, val_loader, device, std_reality):
#     criterion=nn.MSELoss()
#     valMSE_loss = 0.
#     valLEN = 0
#     for i, (inputs, targets, unorm, bnorm, gridsq, f, gprime) in enumerate(val_loader):
#         ### Inputs are vorticity, stretch and strain (normalised by modulus grad u)
#         ### vort_div_stretch and vort_div_strain are the standard deviations of vortivity divided by std of stretch and strain, respectively
#         inputs = torch.squeeze(inputs).to(device)
#         targets = torch.squeeze(targets).to(device)
#         unorm = torch.squeeze(unorm).to(device)
#         bnorm = torch.squeeze(bnorm).to(device)
#         gridsq = torch.squeeze(gridsq).to(device)
#         f = torch.squeeze(f).to(device)
#         gp = torch.squeeze(gprime).to(device)
#         std_y = torch.tensor(std_reality).to(device)

#         out = NET(inputs)
#         pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
#         pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4)], dim=1)
#         pred = torch.cat((pred_mom, pred_buoy), dim=1)
#         reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(std_reality))], dim=1)
#         valMSE=criterion(pred, reality)
#         valMSE_loss += valMSE.item() * len(inputs)
#         valLEN += len(inputs)
#     valMSE_loss = valMSE_loss / valLEN
#     return valMSE_loss
    
# def train_model_no_subPE(NET, datloader, val_loader, device, num_epochs = 50, learning_rate = 1e-2, 
#                 std_reality = [0.00176617, 0.00081482, 0.00218827, 5.0140246e-05, 3.0815412e-05]):
#     optimizer = optim.Adam(NET.parameters(), lr=learning_rate)
#     criterion=nn.MSELoss()
#     log = {'MSE': [], 'valMSE': []}
#     log_epoch = {'MSE': []}
#     for epoch in tqdm(range(num_epochs)):
#         MSE_loss = 0.
#         LEN = 0
#         NET.train()

#         for i, (inputs, targets, unorm, bnorm, gridsq, f, gprime) in enumerate(datloader):
#             # Forward pass
#             inputs = torch.squeeze(inputs).to(device)
#             targets = torch.squeeze(targets).to(device)
#             unorm = torch.squeeze(unorm).to(device)
#             bnorm = torch.squeeze(bnorm).to(device)
#             gridsq = torch.squeeze(gridsq).to(device)
#             f = torch.squeeze(f).to(device)
#             gp = torch.squeeze(gprime).to(device)
#             std_y = torch.tensor(std_reality).to(device)
            

#             out = NET(inputs)
#             pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
#             pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4)], dim=1)
#             pred = torch.cat((pred_mom, pred_buoy), dim=1)
#             reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(std_reality))], dim=1)
            
#             MSE=criterion(pred, reality)
#             if i%10==0:
#                 #print(MSE.item())
#                 perdone = (i/len(datloader))*100
#                 print(f'{perdone}% done with epoch')
#             log_epoch['MSE'].append(MSE.item())
        
#             # Backward and optimize
#             optimizer.zero_grad()
#             MSE.backward()
#             optimizer.step()
#             log_epoch['MSE'].append(MSE.item())
#             MSE_loss += MSE.item() * len(inputs)
#             LEN += len(inputs)

#         MSE_loss = MSE_loss / LEN
#         NET.eval()
#         with torch.no_grad():
#             val_loss = validation_loss_no_subPE(NET, val_loader, device, std_reality)

#         log['MSE'].append(MSE_loss)
#         log['valMSE'].append(val_loss)
    
#         ## Print loss after every epoch
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {MSE_loss:.6f}, Validation Loss: {val_loss:.6f}")
#     return NET, log, log_epoch

# ############### Functions to use when predicting all siz components of EPF tensor #######################
# class full_SnapshotDataset(Dataset):
#     def __init__(self, data_files, get_Coriolis = False, coarsen_fac = None, gprime = [0.98, 0.0098], one_data_mask = True):
#         super().__init__()
        
#         self.data_files = data_files
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         if get_Coriolis: #if the dataset doesn't already have Coriolis term defined, create it
#             exp = 'R32'
#             path = '/scratch/nl2631/mom6/double_gyre/'
#             snap_name = 'snap_prog__'
#             files = glob.glob(f"{path}{exp}/{snap_name}*") #list of all years of data to loop through
#             mydata = DoubleGyre_Snapshots(files[0], coarsen_fac = coarsen_fac, gen_path = path, exp = exp)
#             Coriolis = mydata.coarse_grid_nonsym.interp(mydata.coarse_grid_nonsym.interp(mydata.ds_coarse_grid_nonsym.Coriolis,'X'),'Y')
            
#             del mydata
#             del files
#         else:
#             Coriolis = full_data.Coriolis
            
#         Coriolis = Coriolis.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#         self.Coriolis = Coriolis
        
#         x = xr.Dataset()
#         y = xr.Dataset()
#         if one_data_mask:
#             hmask = full_data.data_mask.isel(zl=-1).drop('zl')
#             hmask = hmask.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#             hmask["zl"] = full_data.Del_sq["zl"]
#         else:
#             hmask = full_data.data_mask
            
#         x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
#         x['extop'], x['eytop'], x['exbot'], x['eybot'], self.buoynorm = self._buoy_input_comp(full_data*hmask)
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms
        
#         y['Ruu'] = full_data.Ruu * hmask
#         y['Ruv'] = full_data.Ruv * hmask
#         y['Rvv'] = full_data.Rvv * hmask
#         y['subPE'] = full_data.subPE * hmask * 2 * full_data.hbar_coarse
#         y['Formx'] = full_data.Formx * hmask
#         y['Formy'] = full_data.Formy * hmask
#         self.Delsq = full_data.Del_sq * hmask  
#         self.subPEold = full_data.subPE * hmask
#         self.hbar = full_data.hbar_coarse * hmask
        
#         self.Coriolis["zl"] = full_data.Del_sq["zl"]

        
#         zlevels = Coriolis["zl"]
#         gp = xr.DataArray(
#             data=gprime,
#             dims = ["zl"],
#             coords=dict(
#                 zl=("zl", Coriolis["zl"].data)
#             )
#         )
#         gp = gp.expand_dims(dim = {'Time': len(full_data.Del_sq['Time']), 
#                               'yh': len(full_data.Del_sq['yh']),
#                               'xh': len(full_data.Del_sq['xh'])})
#         gp['Time'] = full_data.Del_sq["Time"]
#         gp['yh'] = full_data.Del_sq["yh"]
#         gp['xh'] = full_data.Del_sq["xh"]
#         self.gprime = gp
        
#         self.x = x #torch.tensor(ds_np_reshape[:,:,0:3], dtype = torch.float32)
#         self.y = y #torch.tensor(ds_np_reshape[:,:,3:6], dtype = torch.float32)
        
#     def _mom_input_comp(self, data):
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _buoy_input_comp(self, data):
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm
        
#     def __len__(self):
#         return self.x.sizes['Time']

#     def __getitem__(self,idx):
#         xdat = self.x.isel(Time = idx)
#         ydat = self.y.isel(Time = idx)
#         modu = self.momnorm.isel(Time = idx)
#         modb = self.buoynorm.isel(Time = idx)
#         Delsq = self.Delsq.isel(Time = idx)
#         gprime = self.gprime.isel(Time = idx)
#         data = xr.merge([xdat,ydat])
#         data['modu'] = modu
#         data['modb'] = modb
#         data['Delsq'] = Delsq
#         data['f'] = self.Coriolis
#         data['gprime'] = gprime
#         ds_np = data.to_array().to_numpy()
    
#         dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
#         dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
#         dimZ = np.shape(ds_np)[-3]   # Points in Z axis (layer)
#         dimF = np.shape(ds_np)[0]   # total number of features in dataset
    
#         ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY * dimZ).transpose()

#         ds_np_reshape[ds_np_reshape == 0.] = np.nan
#         nansum = np.sum(ds_np_reshape, axis = 1)
#         nan_inds = np.isnan(nansum)

#         masked_data = ds_np_reshape[~nan_inds]

#         x_mom = torch.tensor(masked_data[:,0:3] / masked_data[:,13][:,np.newaxis], dtype = torch.float32)
#         x_buoy = torch.tensor(masked_data[:,3:7] / masked_data[:,14][:,np.newaxis], dtype = torch.float32)
#         x_out = torch.cat((x_mom, x_buoy), dim=1)
#         y_out = torch.tensor(masked_data[:,7:13], dtype = torch.float32)
#         unorm = torch.tensor(masked_data[:,13], dtype = torch.float32)
#         bnorm = torch.tensor(masked_data[:,14], dtype = torch.float32)
#         gridsq = torch.tensor(masked_data[:,15], dtype = torch.float32)
#         f = torch.tensor(masked_data[:,16], dtype = torch.float32)
#         gprime = torch.tensor(masked_data[:,17], dtype = torch.float32)
        
#         return x_out, y_out, unorm, bnorm, gridsq, f, gprime
    
# def validation_loss_full(NET, val_loader, device, std_reality):
#     criterion=nn.MSELoss()
#     valMSE_loss = 0.
#     valLEN = 0
#     for i, (inputs, targets, unorm, bnorm, gridsq, f, gprime) in enumerate(val_loader):
#         ### Inputs are vorticity, stretch and strain (normalised by modulus grad u)
#         ### vort_div_stretch and vort_div_strain are the standard deviations of vortivity divided by std of stretch and strain, respectively
#         inputs = torch.squeeze(inputs).to(device)
#         targets = torch.squeeze(targets).to(device)
#         unorm = torch.squeeze(unorm).to(device)
#         bnorm = torch.squeeze(bnorm).to(device)
#         gridsq = torch.squeeze(gridsq).to(device)
#         f = torch.squeeze(f).to(device)
#         gp = torch.squeeze(gprime).to(device)
#         std_y = torch.tensor(std_reality).to(device)

#         out = NET(inputs)
#         pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
#         pred_PE = out[:,3] * gridsq * gp * (bnorm**2)/std_y[3]
#         pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (4, 5)], dim=1)
#         pred = torch.cat((pred_mom, pred_PE[:,np.newaxis], pred_buoy), dim=1)
#         # pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (3, 4, 5)], dim=1)
#         # pred = torch.cat((pred_mom, pred_buoy), dim=1)
#         reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(std_reality))], dim=1)
#         valMSE=criterion(pred, reality)
#         valMSE_loss += valMSE.item() * len(inputs)
#         valLEN += len(inputs)
#     valMSE_loss = valMSE_loss / valLEN
#     return valMSE_loss

# def train_full_model(NET, datloader, val_loader, device, num_epochs = 50, learning_rate = 1e-2, 
#                 std_reality = [0.00176617, 0.00081482, 0.00218827, 2.078, 5.0140246e-05, 3.0815412e-05]):
#     optimizer = optim.Adam(NET.parameters(), lr=learning_rate)
#     criterion=nn.MSELoss()
#     log = {'MSE': [], 'valMSE': []}
#     log_epoch = {'MSE': []}
#     for epoch in tqdm(range(num_epochs)):
#         MSE_loss = 0.
#         LEN = 0
#         NET.train()

#         for i, (inputs, targets, unorm, bnorm, gridsq, f, gprime) in enumerate(datloader):
#             # Forward pass
#             inputs = torch.squeeze(inputs).to(device)
#             targets = torch.squeeze(targets).to(device)
#             unorm = torch.squeeze(unorm).to(device)
#             bnorm = torch.squeeze(bnorm).to(device)
#             gridsq = torch.squeeze(gridsq).to(device)
#             f = torch.squeeze(f).to(device)
#             gp = torch.squeeze(gprime).to(device)
#             std_y = torch.tensor(std_reality).to(device)
            

#             out = NET(inputs)
#             pred_mom = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in (0, 1, 2)], dim=1)
#             pred_PE = out[:,3] * gridsq * gp * (bnorm**2)/std_y[3]
#             pred_buoy = torch.stack([out[:,i] * gridsq * unorm * f * (bnorm)/std_y[i] for i in (4, 5)], dim=1)
#             pred = torch.cat((pred_mom, pred_PE[:,np.newaxis], pred_buoy), dim=1)
#             reality = torch.stack([targets[:,i]/std_y[i] for i in range(len(std_reality))], dim=1)
            
#             MSE=criterion(pred, reality)
#             if i%10==0:
#                 print(MSE.item())
#             log_epoch['MSE'].append(MSE.item())
        
#             # Backward and optimize
#             optimizer.zero_grad()
#             MSE.backward()
#             optimizer.step()
#             log_epoch['MSE'].append(MSE.item())
#             MSE_loss += MSE.item() * len(inputs)
#             LEN += len(inputs)

#         MSE_loss = MSE_loss / LEN
#         NET.eval()
#         with torch.no_grad():
#             val_loss = validation_loss_full(NET, val_loader, device, std_reality)

#         log['MSE'].append(MSE_loss)
#         log['valMSE'].append(val_loss)
    
#         ## Print loss after every epoch
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {MSE_loss:.6f}, Validation Loss: {val_loss:.6f}")
#     return NET, log, log_epoch

# ############### Functions to use when not predicting sub-filter potential energy and only one layer at a time #######################
# class single_layer_SnapshotDataset_no_subPE(Dataset):
#     def __init__(self, data_files, layer, get_Coriolis = False, coarsen_fac = None, gprime = [0.98, 0.0098], one_data_mask = True):
#         super().__init__()
        
#         self.layer = layer
#         self.data_files = data_files
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         if get_Coriolis: #if the dataset doesn't already have Coriolis term defined, create it
#             exp = 'R32'
#             path = '/scratch/nl2631/mom6/double_gyre/'
#             snap_name = 'snap_prog__'
#             files = glob.glob(f"{path}{exp}/{snap_name}*") #list of all years of data to loop through
#             mydata = DoubleGyre_Snapshots(files[0], coarsen_fac = coarsen_fac, gen_path = path, exp = exp)
#             Coriolis = mydata.coarse_grid_nonsym.interp(mydata.coarse_grid_nonsym.interp(mydata.ds_coarse_grid_nonsym.Coriolis,'X'),'Y')
#             self.Coriolis = Coriolis
#             del mydata
#             del files
#         else:
#             self.Coriolis = full_data.Coriolis

#         x = xr.Dataset()
#         y = xr.Dataset()
#         if one_data_mask:
#             hmask = full_data.data_mask.isel(zl=-1).drop('zl')
#             hmask = hmask.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#             hmask["zl"] = full_data.Del_sq["zl"]
#         else:
#             hmask = full_data.data_mask
            
#         x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
#         x['extop'], x['eytop'], x['exbot'], x['eybot'], self.buoynorm = self._buoy_input_comp(full_data*hmask)
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms
        
#         y['Ruu'] = full_data.Ruu * hmask
#         y['Ruv'] = full_data.Ruv * hmask
#         y['Rvv'] = full_data.Rvv * hmask
#         y['Formx'] = full_data.Formx * hmask
#         y['Formy'] = full_data.Formy * hmask
#         self.Delsq = full_data.Del_sq * hmask
        
        
        
#         gp = xr.DataArray(
#             data=gprime,
#             dims = ["zl"],
#             coords=dict(
#                 zl=("zl", full_data.Del_sq["zl"].data)
#             )
#         )
#         gp = gp.expand_dims(dim = {'Time': len(full_data.Del_sq['Time']), 
#                               'yh': len(full_data.Del_sq['yh']),
#                               'xh': len(full_data.Del_sq['xh'])})
#         gp['Time'] = full_data.Del_sq["Time"]
#         gp['yh'] = full_data.Del_sq["yh"]
#         gp['xh'] = full_data.Del_sq["xh"]
#         self.gprime = gp
        
#         self.x = x #torch.tensor(ds_np_reshape[:,:,0:3], dtype = torch.float32)
#         self.y = y #torch.tensor(ds_np_reshape[:,:,3:6], dtype = torch.float32)
        
#     def _mom_input_comp(self, data):
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _buoy_input_comp(self, data):
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm
        
#     def __len__(self):
#         return self.x.sizes['Time']

#     def __getitem__(self,idx):
#         xdat = self.x.isel(Time = idx, zl = self.layer)
#         ydat = self.y.isel(Time = idx, zl = self.layer)
#         modu = self.momnorm.isel(Time = idx, zl = self.layer)
#         modb = self.buoynorm.isel(Time = idx, zl = self.layer)
#         Delsq = self.Delsq.isel(Time = idx, zl = self.layer)
#         gprime = self.gprime.isel(Time = idx, zl = self.layer)
#         data = xr.merge([xdat,ydat])
#         data['modu'] = modu
#         data['modb'] = modb
#         data['Delsq'] = Delsq
#         data['f'] = self.Coriolis.isel(Time = idx, zl = self.layer)
#         data['gprime'] = gprime
#         ds_np = data.to_array().to_numpy()
    
#         dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
#         dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
#         dimF = np.shape(ds_np)[0]   # total number of features in dataset
    
#         ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY).transpose()

#         ds_np_reshape[ds_np_reshape == 0.] = np.nan
#         nansum = np.sum(ds_np_reshape, axis = 1)
#         nan_inds = np.isnan(nansum)

#         masked_data = ds_np_reshape[~nan_inds]

#         x_mom = torch.tensor(masked_data[:,0:3] / masked_data[:,13][:,np.newaxis], dtype = torch.float32)
#         x_buoy = torch.tensor(masked_data[:,3:7] / masked_data[:,14][:,np.newaxis], dtype = torch.float32)
#         x_out = torch.cat((x_mom, x_buoy), dim=1)
#         y_out = torch.tensor(masked_data[:,7:12], dtype = torch.float32)
#         unorm = torch.tensor(masked_data[:,12], dtype = torch.float32)
#         bnorm = torch.tensor(masked_data[:,13], dtype = torch.float32)
#         gridsq = torch.tensor(masked_data[:,14], dtype = torch.float32)
#         f = torch.tensor(masked_data[:,15], dtype = torch.float32)
#         gprime = torch.tensor(masked_data[:,16], dtype = torch.float32)
        
#         return x_out, y_out, unorm, bnorm, gridsq, f, gprime



# class single_layer_SnapshotDataset(Dataset):
#     def __init__(self, data_files, layer, get_Coriolis = False, coarsen_fac = None, gprime = [0.98, 0.0098], one_data_mask = True):
#         super().__init__()
        
#         self.layer = layer
#         self.data_files = data_files
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         if get_Coriolis: #if the dataset doesn't already have Coriolis term defined, create it
#             exp = 'R32'
#             path = '/scratch/nl2631/mom6/double_gyre/'
#             snap_name = 'snap_prog__'
#             files = glob.glob(f"{path}{exp}/{snap_name}*") #list of all years of data to loop through
#             mydata = DoubleGyre_Snapshots(files[0], coarsen_fac = coarsen_fac, gen_path = path, exp = exp)
#             Coriolis = mydata.coarse_grid_nonsym.interp(mydata.coarse_grid_nonsym.interp(mydata.ds_coarse_grid_nonsym.Coriolis,'X'),'Y')
#             self.Coriolis = Coriolis
#             del mydata
#             del files
#         else:
#             self.Coriolis = full_data.Coriolis

#         x = xr.Dataset()
#         y = xr.Dataset()
#         if one_data_mask:
#             hmask = full_data.data_mask.isel(zl=-1).drop('zl')
#             hmask = hmask.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#             hmask["zl"] = full_data.Del_sq["zl"]
#         else:
#             hmask = full_data.data_mask
            
#         x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
#         x['extop'], x['eytop'], x['exbot'], x['eybot'], self.buoynorm = self._buoy_input_comp(full_data*hmask)
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms
        
#         y['Ruu'] = full_data.Ruu * hmask
#         y['Ruv'] = full_data.Ruv * hmask
#         y['Rvv'] = full_data.Rvv * hmask
#         y['subPE'] = full_data.subPE * hmask * 2 * full_data.hbar_coarse
#         y['Formx'] = full_data.Formx * hmask
#         y['Formy'] = full_data.Formy * hmask
#         self.Delsq = full_data.Del_sq * hmask
#         self.subPEold = full_data.subPE * hmask
        
        
#         gp = xr.DataArray(
#             data=gprime,
#             dims = ["zl"],
#             coords=dict(
#                 zl=("zl", full_data.Del_sq["zl"].data)
#             )
#         )
#         gp = gp.expand_dims(dim = {'Time': len(full_data.Del_sq['Time']), 
#                               'yh': len(full_data.Del_sq['yh']),
#                               'xh': len(full_data.Del_sq['xh'])})
#         gp['Time'] = full_data.Del_sq["Time"]
#         gp['yh'] = full_data.Del_sq["yh"]
#         gp['xh'] = full_data.Del_sq["xh"]
#         self.gprime = gp
        
#         self.x = x #torch.tensor(ds_np_reshape[:,:,0:3], dtype = torch.float32)
#         self.y = y #torch.tensor(ds_np_reshape[:,:,3:6], dtype = torch.float32)
        
#     def _mom_input_comp(self, data):
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _buoy_input_comp(self, data):
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm
        
#     def __len__(self):
#         return self.x.sizes['Time']

#     def __getitem__(self,idx):
#         xdat = self.x.isel(Time = idx, zl = self.layer)
#         ydat = self.y.isel(Time = idx, zl = self.layer)
#         modu = self.momnorm.isel(Time = idx, zl = self.layer)
#         modb = self.buoynorm.isel(Time = idx, zl = self.layer)
#         Delsq = self.Delsq.isel(Time = idx, zl = self.layer)
#         gprime = self.gprime.isel(Time = idx, zl = self.layer)
#         data = xr.merge([xdat,ydat])
#         data['modu'] = modu
#         data['modb'] = modb
#         data['Delsq'] = Delsq
#         data['f'] = self.Coriolis.isel(Time = idx, zl = self.layer)
#         data['gprime'] = gprime
#         ds_np = data.to_array().to_numpy()
    
#         dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
#         dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
#         dimF = np.shape(ds_np)[0]   # total number of features in dataset
    
#         ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY).transpose()

#         ds_np_reshape[ds_np_reshape == 0.] = np.nan
#         nansum = np.sum(ds_np_reshape, axis = 1)
#         nan_inds = np.isnan(nansum)

#         masked_data = ds_np_reshape[~nan_inds]

#         x_mom = torch.tensor(masked_data[:,0:3] / masked_data[:,13][:,np.newaxis], dtype = torch.float32)
#         x_buoy = torch.tensor(masked_data[:,3:7] / masked_data[:,14][:,np.newaxis], dtype = torch.float32)
#         x_out = torch.cat((x_mom, x_buoy), dim=1)
#         y_out = torch.tensor(masked_data[:,7:13], dtype = torch.float32)
#         unorm = torch.tensor(masked_data[:,13], dtype = torch.float32)
#         bnorm = torch.tensor(masked_data[:,14], dtype = torch.float32)
#         gridsq = torch.tensor(masked_data[:,15], dtype = torch.float32)
#         f = torch.tensor(masked_data[:,16], dtype = torch.float32)
#         gprime = torch.tensor(masked_data[:,17], dtype = torch.float32)
        
#         return x_out, y_out, unorm, bnorm, gridsq, f, gprime
        

# def build_mom_prediction(NET, testdata):
#     """
#     This function takes the ANN and test dataset and re-dimensionalises the ANN output for comparison with test targets. This function is for momentum fluxes only (DO NOT USE WITH ANNs TRAINED FOR MORE FLUXES)
#     """
#     test_input = testdata.x / testdata.momnorm
#     NETtest = ANN_wrapper(test_input, NET)
#     Pred = testdata.y.copy() * 0
#     Pred['Ruu'] = NETtest.result.isel(output = 0)
#     Pred['Ruv'] = NETtest.result.isel(output = 1)
#     Pred['Rvv'] = NETtest.result.isel(output = 2) 
#     dim_Pred = Pred * testdata.Delsq * testdata.momnorm**2
#     return dim_Pred

# def build_EPF_prediction(NET, testdata):
#     """
#     Function which builds predictions for full EPF components from test data and an ANN
#     """
#     NET.eval()
#     test_input = testdata.x.copy(deep = True)
#     test_input['vort'] = testdata.x.vort / testdata.momnorm
#     test_input['stretch'] = testdata.x.stretch / testdata.momnorm
#     test_input['strain'] = testdata.x.strain / testdata.momnorm
    
#     test_input['extop'] = testdata.x.extop / testdata.buoynorm
#     test_input['eytop'] = testdata.x.eytop / testdata.buoynorm
#     test_input['exbot'] = testdata.x.exbot / testdata.buoynorm
#     test_input['eybot'] = testdata.x.eybot / testdata.buoynorm
#     NETtest = ANN_wrapper(test_input, NET)
#     Pred = testdata.y.copy(deep=True) * 0
#     Pred['Ruu'] = NETtest.result.isel(output = 0) * testdata.Delsq * testdata.momnorm**2
#     Pred['Ruv'] = NETtest.result.isel(output = 1) * testdata.Delsq * testdata.momnorm**2
#     Pred['Rvv'] = NETtest.result.isel(output = 2) * testdata.Delsq * testdata.momnorm**2
#     Pred['subPE'] = NETtest.result.isel(output = 3) * testdata.Delsq * (testdata.buoynorm**2) * testdata.gprime
#     Pred['Formx'] = NETtest.result.isel(output = 4) * testdata.Delsq * testdata.momnorm * testdata.Coriolis * testdata.buoynorm
#     Pred['Formy'] = NETtest.result.isel(output = 5) * testdata.Delsq * testdata.momnorm * testdata.Coriolis * testdata.buoynorm
#     # dim_Pred = Pred * testdata.Delsq * testdata.momnorm**2
#     return Pred

# def build_EPF_prediction_no_subPE(NET, testdata):
#     """
#     Function which builds predictions for full EPF components from test data and an ANN
#     """
#     print('hi')
#     NET.eval()
#     test_input = testdata.x.copy(deep = True)
#     test_input['vort'] = testdata.x.vort / testdata.momnorm
#     test_input['stretch'] = testdata.x.stretch / testdata.momnorm
#     test_input['strain'] = testdata.x.strain / testdata.momnorm
    
#     test_input['extop'] = testdata.x.extop / testdata.buoynorm
#     test_input['eytop'] = testdata.x.eytop / testdata.buoynorm
#     test_input['exbot'] = testdata.x.exbot / testdata.buoynorm
#     test_input['eybot'] = testdata.x.eybot / testdata.buoynorm
#     NETtest = ANN_wrapper(test_input, NET)
#     Pred = testdata.y.copy(deep=True) * 0
#     Pred['Ruu'] = NETtest.result.isel(output = 0) * testdata.Delsq * testdata.momnorm**2
#     Pred['Ruv'] = NETtest.result.isel(output = 1) * testdata.Delsq * testdata.momnorm**2
#     Pred['Rvv'] = NETtest.result.isel(output = 2) * testdata.Delsq * testdata.momnorm**2
#     Pred['Formx'] = NETtest.result.isel(output = 3) * testdata.Delsq * testdata.momnorm * testdata.Coriolis * testdata.buoynorm
#     Pred['Formy'] = NETtest.result.isel(output = 4) * testdata.Delsq * testdata.momnorm * testdata.Coriolis * testdata.buoynorm
#     # dim_Pred = Pred * testdata.Delsq * testdata.momnorm**2
#     return Pred

   
# class mom_only_SnapshotDataset(Dataset):
#     def __init__(self, data_files):
#         super().__init__()
#         self.data_files = data_files
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         x = xr.Dataset()
#         y = xr.Dataset()
#         norm_info = xr.Dataset()
#         x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*full_data.data_mask)
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms

#         y['Ruu'] = full_data.Ruu * full_data.data_mask
#         y['Ruv'] = full_data.Ruv * full_data.data_mask
#         y['Rvv'] = full_data.Rvv * full_data.data_mask

#         self.Delsq = full_data.Del_sq * full_data.data_mask

#         self.x = x #torch.tensor(ds_np_reshape[:,:,0:3], dtype = torch.float32)
#         self.y = y #torch.tensor(ds_np_reshape[:,:,3:6], dtype = torch.float32)
        
#     def _mom_input_comp(self, data):
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _buoy_input_comp(self, data):
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm
        
#     def __len__(self):
#         return self.x.sizes['Time']

#     def __getitem__(self, idx):
#         x_out = self.x.isel(Time = idx)
#         y_out = self.y.isel(Time = idx)
#         mod_u = self.momnorm.isel(Time = idx)
#         Delsq = self.Delsq.isel(Time = idx)
        
#         data = xr.merge([x_out,y_out])
#         data['mod_u'] = mod_u
#         data['Delsq'] = Delsq
#         ds_np = data.to_array().to_numpy()

#         dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
#         dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
#         dimZ = np.shape(ds_np)[-3]   # Points in Z axis (layer)
#         dimT = np.shape(ds_np)[-4]   # Points in Z axis (layer)
#         dimF = np.shape(ds_np)[0]   # total number of features in dataset

#         ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY * dimZ).transpose()

#         ds_np_reshape[ds_np_reshape == 0.] = np.nan
#         nansum = np.sum(ds_np_reshape, axis = 1)
#         nan_inds = np.isnan(nansum)

#         masked_data = ds_np_reshape[~nan_inds]
#         # The following should be approximately equal to the squareroot of 2
#         vort_div_stretch = masked_data[:,0].std() / masked_data[:,1].std()
#         vort_div_strain = masked_data[:,0].std() / masked_data[:,2].std()

#         # Now including the normalising information in to the input data
#         x_dat = torch.tensor(masked_data[:,0:3] / masked_data[:,6][:,np.newaxis], dtype = torch.float32)
#         y_dat = torch.tensor(masked_data[:,3:6], dtype = torch.float32)
#         unorm = torch.tensor(masked_data[:,6], dtype = torch.float32)
#         gridsq = torch.tensor(masked_data[:,7], dtype = torch.float32)
#         return x_dat, y_dat, unorm, gridsq, vort_div_stretch, vort_div_strain

# # def validation_loss(NET, val_loader, device, std_reality):
# #     criterion=nn.MSELoss()
# #     valMSE_loss = 0.
# #     valLEN = 0
# #     for i, (inputs, targets, unorm, gridsq, vort_div_stretch, vort_div_strain) in enumerate(val_loader):
# #         ### Inputs are vorticity, stretch and strain (normalised by modulus grad u)
# #         ### vort_div_stretch and vort_div_strain are the standard deviations of vortivity divided by std of stretch and strain, respectively
# #         x = torch.squeeze(inputs).to(device)
# #         y = torch.squeeze(targets).to(device)
# #         unorm = torch.squeeze(unorm)
# #         gridsq = torch.squeeze(gridsq)
# #         std_y = torch.tensor(std_reality).to(device)

# #         out = NET(x)
# #         pred = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in range(3)], dim=1)
# #         reality = torch.stack([y[:,i]/std_y[i] for i in range(3)], dim=1)
# #         valMSE=criterion(pred, reality)
# #         valMSE_loss += valMSE.item() * len(x)
# #         valLEN += len(x)
# #     valMSE_loss = valMSE_loss / valLEN
# #     return valMSE_loss

# # def train_model(NET, datloader, val_loader, device, num_epochs = 50, learning_rate = 1e-2, 
# #                 std_reality = [0.00194786, 0.00091312, 0.0024226]):
# #     optimizer = optim.Adam(NET.parameters(), lr=learning_rate)
# #     criterion=nn.MSELoss()
# #     log = {'MSE': [], 'valMSE': []}
# #     log_epoch = {'MSE': []}
# #     for epoch in tqdm(range(num_epochs)):
# #         MSE_loss = 0.
# #         LEN = 0
# #         NET.train()
# #         for i, (inputs, targets, unorm, gridsq, vort_div_stretch, vort_div_strain) in enumerate(datloader):
# #             # Forward pass
# #             x = torch.squeeze(inputs).to(device)
# #             y = torch.squeeze(targets).to(device)
# #             unorm = torch.squeeze(unorm).to(device)
# #             gridsq = torch.squeeze(gridsq).to(device)
# #             std_y = torch.tensor(std_reality).to(device)

# #             out = NET(x)
# #             pred = torch.stack([out[:,i]*gridsq*(unorm**2)/std_y[i] for i in range(3)], dim=1)
# #             reality = torch.stack([y[:,i]/std_y[i] for i in range(3)], dim=1)

# #             MSE=criterion(pred, reality)
        
# #             # Backward and optimize
# #             optimizer.zero_grad()
# #             MSE.backward()
# #             optimizer.step()
# #             log_epoch['MSE'].append(MSE.item())
# #             MSE_loss += MSE.item() * len(x)
# #             LEN += len(x)

# #         MSE_loss = MSE_loss / LEN
# #         NET.eval()
# #         with torch.no_grad():
# #             val_loss = validation_loss(NET, val_loader, device, std_reality)

# #         log['MSE'].append(MSE_loss)
# #         log['valMSE'].append(val_loss)
    
# #         ## Print loss after every epoch
# #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {MSE_loss:.6f}, Validation Loss: {val_loss:.6f}")
#     # return NET, log, log_epoch
    



# class SnapshotDataset_no_subPE_indvlayer(Dataset):
#     def __init__(self, data_files, ignore_bottom = False, get_Coriolis = False, coarsen_fac = None, gprime = [0.98, 0.0098], one_data_mask = True):
#         super().__init__()

#         self.data_files = data_files
#         full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
#         if get_Coriolis: #if the dataset doesn't already have Coriolis term defined, create it
#             exp = 'R32'
#             path = '/scratch/nl2631/mom6/double_gyre/'
#             snap_name = 'snap_prog__'
#             files = glob.glob(f"{path}{exp}/{snap_name}*") #list of all years of data to loop through
#             mydata = DoubleGyre_Snapshots(files[0], coarsen_fac = coarsen_fac, gen_path = path, exp = exp)
#             Coriolis = mydata.coarse_grid_nonsym.interp(mydata.coarse_grid_nonsym.interp(mydata.ds_coarse_grid_nonsym.Coriolis,'X'),'Y')

#             del mydata
#             del files
#         else:
#             Coriolis = full_data.Coriolis

#         self.Coriolis = Coriolis
#         self.ig_bot = ignore_bottom
#         x = xr.Dataset()
#         y = xr.Dataset()
#         if one_data_mask:
#             hmask = full_data.data_mask.isel(zl=-1).drop('zl')
#             hmask = hmask.expand_dims(dim = {'zl': len(full_data.data_mask['zl'])})
#             hmask["zl"] = full_data.Del_sq["zl"]
#         else:
#             hmask = full_data.data_mask
        
#         x['vort'], x['stretch'], x['strain'], self.momnorm = self._mom_input_comp(full_data*hmask)
#         x['extop'], x['eytop'], x['exbot'], x['eybot'], self.buoynorm = self._buoy_input_comp(full_data*hmask)
#         # x = x / self.momnorm #normalising the input by magnitude of all mom terms

#         y['Ruu'] = full_data.Ruu * hmask
#         y['Ruv'] = full_data.Ruv * hmask
#         y['Rvv'] = full_data.Rvv * hmask
#         y['Formx'] = full_data.Formx * hmask
#         y['Formy'] = full_data.Formy * hmask
#         self.Delsq = (full_data.Del_sq * hmask)
#         # self.subPEold = full_data.subPE * hmask
#         self.hbar = (full_data.hbar_coarse * hmask)

#         zlevels = Coriolis["zl"]
#         gp = xr.DataArray(
#             data=gprime,
#             dims = ["zl"],
#             coords=dict(
#                 zl=("zl", Coriolis["zl"].data)
#             )
#         )
#         gp = gp.expand_dims(dim = {'yh': len(full_data.Del_sq['yh']),
#                               'xh': len(full_data.Del_sq['xh'])})
#         gp['yh'] = full_data.Del_sq["yh"]
#         gp['xh'] = full_data.Del_sq["xh"]
#         self.gprime = gp

#         self.x = x #torch.tensor(ds_np_reshape[:,:,0:3], dtype = torch.float32)
#         self.y = y #torch.tensor(ds_np_reshape[:,:,3:6], dtype = torch.float32)
#         self.momnorm = self.momnorm
#         self.buoynorm = self.buoynorm

#     def _mom_input_comp(self, data):
#         vort = data.vhatx - data.uhaty
#         stretch = data.uhatx - data.vhaty
#         strain = data.uhaty + data.vhatx
#         norm = np.sqrt(vort**2 + stretch**2 + strain**2)
#         return vort, stretch, strain, norm

#     def _buoy_input_comp(self, data):
#         extop = data.ex_top
#         eytop = data.ey_top
#         exbot = data.ex_bot
#         eybot = data.ey_bot
#         #If we want to ignore the bottom slope, we set the last interface slopes at bottom to 0
#         if self.ig_bot:
#             exbot.isel(zl=-1).values[:] = 0
#             eybot.isel(zl=-1).values[:] = 0
#         norm = np.sqrt(extop**2 + eytop**2 + exbot**2 + eybot**2)
#         return extop, eytop, exbot, eybot, norm

#     def __len__(self):
#         return self.x.sizes['Time']

#     def __getitem__(self,idx,layer):
#         xdat = self.x.isel(Time = idx, zl = layer)
#         ydat = self.y.isel(Time = idx, zl = layer)
#         modu = self.momnorm.isel(Time = idx, zl = layer)
#         modb = self.buoynorm.isel(Time = idx, zl = layer)
#         Delsq = self.Delsq.isel(Time = idx, zl = layer)
#         gprime = self.gprime
#         data = xr.merge([xdat,ydat])
#         data['modu'] = modu
#         data['modb'] = modb
#         data['Delsq'] = Delsq
#         data['f'] = self.Coriolis.isel(Time = idx, zl = layer)
#         data['gprime'] = gprime.isel(zl = layer)
        
#         ds_np = data.to_array().to_numpy()

#         dimX = np.shape(ds_np)[2]   # Points in X axis (long)
#         dimY = np.shape(ds_np)[1]   # Points in Y axis (lat)
#         #dimZ = np.shape(ds_np)[-3]   # Points in Z axis (layer)
#         dimF = np.shape(ds_np)[0]   # total number of features in dataset

#         ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY).transpose()

#         ds_np_reshape[ds_np_reshape == 0.] = np.nan
#         nansum = np.sum(ds_np_reshape, axis = 1)
#         nan_inds = np.isnan(nansum)

#         masked_data = ds_np_reshape[~nan_inds]

#         x_mom = torch.tensor(masked_data[:,0:3] / masked_data[:,12][:,np.newaxis], dtype = torch.float32)
#         x_buoy = torch.tensor(masked_data[:,3:7] / masked_data[:,13][:,np.newaxis], dtype = torch.float32)
#         x_out = torch.cat((x_mom, x_buoy), dim=1)
#         y_out = torch.tensor(masked_data[:,7:12], dtype = torch.float32)
#         unorm = torch.tensor(masked_data[:,12], dtype = torch.float32)
#         bnorm = torch.tensor(masked_data[:,13], dtype = torch.float32)
#         gridsq = torch.tensor(masked_data[:,14], dtype = torch.float32)
#         f = torch.tensor(masked_data[:,15], dtype = torch.float32)
#         gprime = torch.tensor(masked_data[:,16], dtype = torch.float32)

#         return x_out, y_out, unorm, bnorm, gridsq, f, gprime




# def Coarsen_DoubleGyre_Snapshots(data, min_h):
#     """
#     Function which applies the various processing functions from the DoubleGyre_Snapshots class and outputs one coarenned xarray with all desired information for comparison with coarse res output
#     """
#     print("hi! I'm starting!")
#     ds_prog = data.calc_press_terms(data.data)
#     print("Pressure terms are calculated")

#     ds_interp = data.centre_interp(ds_prog)
#     print("Interpolation complete")
#     del ds_prog #no longer needed after this point

#     hbar = data.hbar_compute(ds_interp)
#     hat_terms = xr.Dataset()
#     hat_terms['uhat'], hat_terms['vhat'], hat_terms['uuhat'], hat_terms['uvhat'], hat_terms['vvhat'] = data.TWA_comp(ds_interp, hbar)
    
#     ##########################
#     # Coarsening output data #
#     ##########################
#     print("Now coarsening output data")
#     coarse_data = xr.Dataset()
#     coarse_data['hbar_coarse'] = data.coarsen_t(hbar) #coarsened filtered thickness
        
#     for var in list(hat_terms.keys()):
#         print(var)
#         coarse_data[var] = data.coarsen_t(hat_terms[var])
#     del hat_terms #no longer need this in memory
    
#     ###############################
#     # Compute masking information #
#     ###############################
#     print("computing mask information")
#     data_mask = data.mask(min_h)
#     data_mask = data.coarsen_t(data_mask) #coarsened mask
#     check = data_mask < 1
#     coarse_data['data_mask'] = np.abs(check - 1) #masking away any coarsened points with any contribution from the fine scale mask

#     return coarse_data