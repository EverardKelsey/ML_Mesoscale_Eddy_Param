import numpy as np
import glob
import xarray as xr
from xgcm import Grid
import sys
sys.path.append('/home/kae10022/PythonScripts/')
sys.path.insert(0, "../../gcm-filters/")
sys.path.insert(0, "../../../gcm-filters/")
import gcm_filters
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class SimulationData:
    """
    A class to read in MOM6 simulation datasets and apply a set of operations to pre-process the data for ML.
    This includes adding variables and non-dimensionalising variables. All concievable variables desired for training 
    Eliassen-Palm flux components are computed. Any inclusion of stencil information or data rotation/augmentation 
    is done in a separate class to the data which is generated within this class.
    This method reads in snapshot datasets saved directly from MOM6

    Attributes:
        - data_path    : The directory in which the snapshot data resides to be processed
        - coarsen_fac  : The coarsening factor to apply to the data. If more than one coarsening factor is given, 
            this class will subsample the higher resolution data so that 
    Default Attributes:
        - FGR          : filter to grid width ratio = 2.5 
        - pad_wet_mask : add zeros to the boundaries when defining the filter? Default = True
        - gpu          : perform filtering on GPU? Default = False
        - shape        : shape of the filter. Default = "gaussian"
        - avoid_bottom : Instead of filtering bottom topography, subsample on the coarse-grid scale. Default = True
        - GFS          : reduced gravity at the free surface. Default = 0.98 (why this isn't 9.8 is a mystery to me)
        - GINT         : reduced gravity at the layer interfaces. Default = 0.0098
    Private Methods:
        - _build_grids_and_filters() : prepare the original data grid and the coarsened data grid
        - _open_data()               : open the datafile as an xarray
    """

    def __init__(
        self,
        data_path,
        data_file,
        coarsen_fac,
        FGR = 2.5,
        nonsym_grid = True,
        pad_wet_mask = True,
        gpu = False,
        shape = "gaussian",
        GFS = 0.98,
        GINT = 0.0098,
        grid_args = {
            'include_time': True,
        },
        subsample_time = False,
        sampling_interval = None,
        preprocess = False,
        min_h = 100,
        simple_hbar = False,
        debug = True
    ):

        self.data_path = data_path
        self.data_file = data_file
        self.coarsen_fac = coarsen_fac
        self.FGR = FGR
        self.filter_fac = coarsen_fac * FGR
        self.nonsym_grid = nonsym_grid
        self.pad_wet_mask = pad_wet_mask
        self.gpu = gpu
        self.shape = shape
        self.GFS = GFS
        self.GINT = GINT
        self.grid_args = grid_args
        self.subsample_time = subsample_time
        self.sampling_interval = sampling_interval
        self.min_h = min_h
        self.simple_hbar = simple_hbar
        self.debug = debug

        # Build the data grids and open the data 
        self._open_data() #open the data
        if self.subsample_time:
            self.data = self.data.isel(Time = slice(0,len(self.data['Time']),self.sampling_interval))
        self._prepare_dsgrid() #prepare the symmetric and nonsymmetric grids
        self.ds_grid, self.grid = self._make_grid(self.ds_grid)
        self.ds_grid_nonsym, self.grid_nonsym = self._make_grid(self.ds_grid_nonsym, symmetric = False)
        self._coarsen_grid() #prepare the symmetric and nonsymmetric coarsened grids
        self._create_hfilter() #create the filter object

        # defining variables that recur throughout functions #
        self.N_layers = self.data.h.coords.sizes['zl'] #


    ### Private methods in order of application in class ###
    def _open_data(self):
        try:
            data = xr.open_dataset(self.data_file, decode_times=False).chunk({"Time":1})
            if self.nonsym_grid:
                self.data = data.isel(xq = slice(1,None), yq = slice(1,None))
        except Exception as e:
            print(f"Error reading dataset: {e}")

    def _prepare_dsgrid(self): 
        self.ds_grid = xr.open_dataset(f"{self.data_path}/static.nc").drop("Time")
        ds_grid2 = xr.open_dataset(f"{self.data_path}/ocean_geometry.nc")
        ds_grid2 = ds_grid2.rename({"lath": "yh", "lonh": "xh", "latq": "yq", "lonq": "xq"})
        assert self.ds_grid.dyCv.equals(ds_grid2.dyCv)
        assert self.ds_grid.dxCv.equals(ds_grid2.dxCv)
        assert self.ds_grid.dxCu.equals(ds_grid2.dxCu)
        assert self.ds_grid.dyCu.equals(ds_grid2.dyCu)
        assert self.ds_grid.area_t.equals(ds_grid2.Ah)
        assert self.ds_grid.wet.equals(ds_grid2.wet)
        self.ds_grid["dxT"] = ds_grid2["dxT"]
        self.ds_grid["dyT"] = ds_grid2["dyT"]
        self.ds_grid["dxBu"] = ds_grid2["dxBu"]
        self.ds_grid["dyBu"] = ds_grid2["dyBu"]

        # self.grid = self._make_grid() #makes the symmetric grid
        self.ds_grid_nonsym = self.ds_grid.isel(xq = slice(1,None), yq = slice(1,None))
        # self.grid_nonsym = self._make_grid(symmetric = False) #making the nonsymmetric grid

    def _make_grid(self, ds_grid, symmetric=True):
        if "zi" in self.data:
            ds_grid["zi"] = self.data["zi"]
            z_coords = {'Z': {'center': 'zl', 'outer': 'zi'}}
        else:
            z_coords = {'Z': {'center': 'zl'}}
        ds_grid["zl"] = self.data["zl"]
        ds_grid["Time"] = self.data["Time"]  # weirdly, need to add the time as a coord to the grid, otherwise time coord is lost after applying an xgcm operation

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
        if self.grid_args['include_time']:
            t_coords = {'Time': {'center': 'Time'}}
        else:
            t_coords = {}        
        coords = {**coords, **z_coords, **t_coords}
        metrics = {
            ('X',):['dxCu','dxCv','dxT','dxBu'],
            ('Y',):['dyCu','dyCv','dyT','dyBu'],
        }
        return ds_grid, Grid(ds_grid, coords=coords, metrics=metrics)

    def _coarsen_grid(self):
        "ds_grid is assumed to hold symmetric grid information, i.e., len(xq)=len(xh)+1 and len(yq)=len(yh)+1"
        self.ds_coarse_grid = xr.Dataset()
        var = 'dyCu'
        var_downsampled = self.ds_grid[var].isel(xq=slice(None, None, self.coarsen_fac))
        self.ds_coarse_grid[var] = var_downsampled.coarsen(yh=self.coarsen_fac).sum()
        var = 'dxCv'
        var_downsampled = self.ds_grid[var].isel(yq=slice(None, None, self.coarsen_fac))
        self.ds_coarse_grid[var] = var_downsampled.coarsen(xh=self.coarsen_fac).sum()
        var = 'dxT'
        tmp = self.ds_grid[var].coarsen(xh=self.coarsen_fac).sum()
        self.ds_coarse_grid[var] = tmp.coarsen(yh=self.coarsen_fac).mean()
        var = 'dyT'
        tmp = self.ds_grid[var].coarsen(yh=self.coarsen_fac).sum()
        self.ds_coarse_grid[var] = tmp.coarsen(xh=self.coarsen_fac).mean()
    
        # infer the remaining length scales from the already computed ones
        # for symmetric grids
        symcoords = {
            'X': {'center': 'xh', 'outer': 'xq'},
            'Y': {'center': 'yh', 'outer': 'yq'}
        }
        nonsymcoords = {
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'}
        }
    
        coarse_grid = Grid(self.ds_coarse_grid, coords=symcoords)
        ###########   
        self.ds_coarse_grid["dxCu"] = coarse_grid.interp(self.ds_coarse_grid["dxT"], 'X')
        self.ds_coarse_grid["dyCv"] = coarse_grid.interp(self.ds_coarse_grid["dyT"], 'Y')    
        self.ds_coarse_grid["dxBu"] = coarse_grid.interp(self.ds_coarse_grid["dxCv"], 'X')    
        self.ds_coarse_grid["dyBu"] = coarse_grid.interp(self.ds_coarse_grid["dyCu"], 'Y')
        var = 'Coriolis'
        self.ds_coarse_grid[var] = self.ds_grid[var].isel(xq=slice(None, None, self.coarsen_fac), yq=slice(None, None, self.coarsen_fac))
        var = 'depth_ocean'
        self.ds_coarse_grid[var] = self.ds_grid[var].coarsen(yh=self.coarsen_fac, xh=self.coarsen_fac).mean()
        varlist = ['zl', 'zi', 'Time']
        for var in varlist:
            if var in self.ds_grid:
                self.ds_coarse_grid[var] = self.ds_grid[var]

        if "zi" in self.ds_grid:
            z_coords = {'Z': {'center': 'zl', 'outer': 'zi'}}
        else:
            z_coords = {'Z': {'center': 'zl'}}

        if self.grid_args['include_time']:
            t_coords = {'Time': {'center': 'Time'}}
        else:
            t_coords = {}        
        metrics = {
            ('X',):['dxCu','dxCv','dxT','dxBu'],
            ('Y',):['dyCu','dyCv','dyT','dyBu'],
        }

        self.coarse_grid = Grid(self.ds_coarse_grid, coords={**symcoords, **z_coords, **t_coords}, metrics=metrics)
        self.ds_coarse_grid_nonsym = self.ds_coarse_grid.isel(xq = slice(1,None), yq=slice(1,None))
        self.coarse_grid_nonsym = Grid(self.ds_coarse_grid_nonsym, coords={**nonsymcoords, **z_coords, **t_coords}, metrics=metrics)

    def _create_hfilter(self):
        #Creating the filtering object associated with data grid - this filter is for the centre point
        if self.nonsym_grid:
            ds_grid = self.ds_grid_nonsym
        else:
            ds_grid = self.ds_grid
        
        wet_mask = ds_grid.wet
        if self.pad_wet_mask:
            wet_mask[0,:] = 0
            wet_mask[-1,:] = 0
            wet_mask[:,0] = 0
            wet_mask[:,-1] = 0
        area = ds_grid.area_t
        if self.shape.lower() == 'gaussian':
            filter_shape = gcm_filters.FilterShape.GAUSSIAN
        elif self.shape.lower() == 'taper':
            filter_shape = gcm_filters.FilterShape.TAPER
            
        if self.gpu:
            import cupy as cp
            wet_mask = wet_mask.chunk({'yh': -1,'xh': -1}) # 1 chunk
            _ = map_to_cupy(wet_mask)
            area = area.chunk({'yh': -1,'xh': -1}) # 1 chunk
            _ = map_to_cupy(area)
        self.hfilter = gcm_filters.Filter(
            filter_scale = self.filter_fac,
            dx_min = 1,
            filter_shape = filter_shape,
            grid_type = gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
            grid_vars = {'area' : area, 'wet_mask' : wet_mask}
        )

    ########################
    #### Public methods ####
    ########################
    
    #### Public methods which deal with grid related operations ####
    def centre_interp(self):
        ds_interp = xr.Dataset()
        for var_name in self.data:
            vardims = self.data[var_name].dims
            if 'xh' in vardims:
                if 'yh' in vardims: # already on xh yh grid
                   ds_interp[var_name] = self.data[var_name] 
                else: # originally on xh yq grid
                   print(f'only interpolating {var_name} to centre point in Y') 
                   ds_interp[var_name] = self.grid_nonsym.interp(self.data[var_name],"Y")
            else:
                if 'yh' in vardims: # originally on xq yh grid
                    print(f'only interpolating {var_name} to centre point in X') 
                    ds_interp[var_name] = self.grid_nonsym.interp(self.data[var_name],"X")
                else: # originally on xq yq grid
                    print(f'first interpolating {var_name} to centre point in Y')
                    ds_interp[var_name] = self.grid_nonsym.interp(self.data[var_name],"Y")
                    print(f'and now interpolating {var_name} to centre point in X')
                    ds_interp[var_name] = self.grid_nonsym.interp(self.data[var_name],"X")
        return ds_interp    

    def coarse_grid_scale(self):
        Del2 = 2 * self.ds_coarse_grid.dxT**2 * self.ds_coarse_grid.dyT**2 / (self.ds_coarse_grid.dxT**2 + self.ds_coarse_grid.dyT**2)
        return Del2
        
    def apply_hfilter(self, target):
        return self.hfilter.apply(target, dims = ['yh','xh'])

    def coarsen_t(self, var):
        if self.nonsym_grid:
            ds_grid = self.ds_grid_nonsym
        else:
            ds_grid = self.ds_grid

        var_coarse = (
            (var * ds_grid['area_t']).coarsen(xh=self.coarsen_fac, yh=self.coarsen_fac).sum()
        ) / ds_grid['area_t'].coarsen(xh=self.coarsen_fac, yh=self.coarsen_fac).sum()
        return var_coarse  

    def coarse_deriv(self, var, dim):
        temp = self.coarsen_t(var)
        temp = self.coarse_grid_nonsym.derivative(temp, dim)
        return self.coarse_grid_nonsym.interp(temp, dim)

    def Rd_compute(self, h, grid, ds_grid):
        # Computes deformation radius
        h1 = h.isel(zl = 0)
        h2 = h.isel(zl = 1)
        if 'xq' in ds_grid.Coriolis.dims:
            if 'yq' in ds_grid.Coriolis.dims:
                f = grid.interp(grid.interp(ds_grid.Coriolis,'X'),'Y')
            else:
                f = grid.interp(ds_grid.Coriolis,'X')
        else:
            if 'yq' in ds_grid.Coriolis.dims:
                f = grid.interp(ds_grid.Coriolis,'Y')
            else:
                f = ds_grid.Coriolis
        Rd = np.sqrt(self.GINT * h1 * h2 / (h1 + h2)) / f
        return Rd

    def centre_interp_Coriolis(self, grid, ds_grid):
        if 'xq' in ds_grid.Coriolis.dims:
            if 'yq' in ds_grid.Coriolis.dims:
                f = grid.interp(grid.interp(ds_grid.Coriolis,'X'),'Y')
            else:
                f = grid.interp(ds_grid.Coriolis,'X')
        else:
            if 'yq' in ds_grid.Coriolis.dims:
                f = grid.interp(ds_grid.Coriolis,'Y')
            else:
                f = ds_grid.Coriolis
        return f

    def mask(self,coarse = False, h = None):
        if coarse:
            return h > self.min_h
        else:
            return self.data.h > self.min_h

    #### Public methods that calculate new variables ####
    def calc_SSH(self):
        self.data['SSH'] = self.data.h.copy(deep=True) * 0
        for n in np.arange(self.N_layers):
            if n==0:
                self.data.SSH[:,n,:,:] = self.data.e.isel(zi=n)
            else:
                self.data.SSH[:,n,:,:] = self.data.e.isel(zi=n) - self.data.h.isel(zl=n-1)
    
    def calc_press_terms(self):
        press = self.data.h.copy(deep=True) * 0
        for n in np.arange(self.N_layers):
            if n==0:
                press[:,n,:,:] = self.GFS*self.data.e.isel(zi=n)        
            else:
                press[:,n,:,:] = np.sum(press,axis=1) + self.GINT*self.data.e.isel(zi=n)  

        self.data['press'] = press

        dp_dx = self.grid_nonsym.derivative(self.data.press,'X') #this is FD with derivs now on outside. We want centred diff, but for now will just interpolate to centre
        if self.debug:
            try:
                dp_dx.compute()
            except:
                print('Failure occurs after computing x pressure gradient')
        
        self.data['dpdx'] = dp_dx

        dp_dy = self.grid_nonsym.derivative(self.data.press,'Y') #this is FD with derivs now on outside. We want centred diff, but for now will just interpolate to centre
        if self.debug:
            try:
                dp_dy.compute()
            except:
                print('Failure occurs after computing y pressure gradient')
        self.data['dpdy'] = dp_dy 
        # return ds_prog

    def calc_pressure_terms(self):
        self.data['press'] = self.data.h.copy(deep=True) * 0
        for n in np.arange(self.N_layers):
            if n==0:
                self.data.press[:,n,:,:] = self.GFS*self.data.e.isel(zi=n)        
            else:
                self.data.press[:,n,:,:] = np.sum(self.data.press,axis=1) + self.GINT*self.data.e.isel(zi=n)       
        self.data['dpdx'] = self.grid_nonsym.derivative(self.data.press,'X') #this is FD with derivs now on outside. We want centred diff, but for now will just interpolate to centre
        self.data['dpdy'] = self.grid_nonsym.derivative(self.data.press,'Y') #this is FD with derivs now on outside. We want centred diff, but for now will just interpolate to centre

    def calc_hbar(self):
        hbar = self.data.h.copy(deep=True) * 0 #initialising structure of hbar
        if self.simple_hbar:
            hbar = self.apply_hfilter(self.data.h) 
        else:
            for n in np.arange(self.N_layers):
                if n == self.N_layers - 1:
                    hbar[:,n,:,:] = self.apply_hfilter(self.data.e.isel(zi = n)) - self.data.e.isel(zi = self.N_layers)
                else:
                    hbar[:,n,:,:] = self.apply_hfilter(self.data.h.isel(zl = n))
        self.data['hbar'] = hbar
        if self.debug:
            try:
                hbar.compute()
            except:
                print('hbar calculation fails')

    def varbar_comp(self, dsvarsel, return_TWA = False):
        varbar = self.data.h.copy(deep = True) * 0

        if self.simple_hbar:
            varbar = self.apply_hfilter((dsvarsel * self.data.h))
        else:
            for n in np.arange(self.N_layers):
                if n == self.N_layers - 1: # if in the bottom layer
                    varbar[:,n,:,:] = self.apply_hfilter(dsvarsel.isel(zl = n) * self.data.e.isel(zi=n)) - self.data.e.isel(zi=self.N_layers)*self.apply_hfilter(dsvarsel.isel(zl=n))
                else:
                    varbar[:,n,:,:] = self.apply_hfilter((dsvarsel * self.data.h).isel(zl=n))

        if return_TWA:
            return varbar, varbar / self.data.hbar
        else:
            return varbar
  
    def TWA_comp(self,varbar, hbar):
        return varbar / hbar

    ### Functions which are used after other operations have been performed ###
    def Reynolds_calc(self, ds):
        """ 
        Computes the Reynolds stress terms of the EPF components from thickness weighted variables
        """
        try:
            ds['Ruu'] = ds.uuhat - ds.uhat**2
            ds['Ruv'] = ds.uvhat - ds.uhat * ds.vhat
            ds['Rvv'] = ds.vvhat - ds.vhat**2
        except Exception as e:
            print(f'{e}')

        if self.debug:
            try:
                ds.Ruu.compute()
            except:
                print("computation of Ruu fails")
                
    def compute_intfc_var(self):
        """ 
        Computes the interface variance (used in subPE term)
        """
        S_eta = self.data.hbar.copy(deep=True) * 0
        for n in np.arange(self.N_layers):
            ebar = self.apply_hfilter((self.data.e).isel(zi=n))
            eebar = self.apply_hfilter((self.data.e**2).isel(zi=n))
            S_eta[:,n,:,:] = (eebar - (ebar**2))
        return S_eta

    def compute_dual_form(self, ds):
        """ 
        Computes the dual form stresses for each vertical interface in the domain (number of layers + 1)
        """
        S_eta_x = self.data.e.copy(deep=True) * 0
        S_eta_y = self.data.e.copy(deep=True) * 0
        N = self.data.e.coords.sizes['zi'] #number of interfaces
        for n in np.arange(N):
            if n == 0:
                #Assuming that the atmosphere is taken as pressure = 0, then there is no eddy form stress on free surface
                continue
            elif n == (N - 1):
                #Assuming that the bottom topography isn't changing (eta' = 0), there is no eddy form stress at bottom boundary
                continue
            else:
                barsqx = self.apply_hfilter(self.data.e.isel(zi=n) * ds.dpdx.isel(zl=n-1))
                barsqy = self.apply_hfilter(self.data.e.isel(zi=n) * ds.dpdy.isel(zl=n-1))
                barxx = self.apply_hfilter(self.data.e.isel(zi=n)) * self.apply_hfilter(ds.dpdx.isel(zl=n-1))
                baryy = self.apply_hfilter(self.data.e.isel(zi=n)) * self.apply_hfilter(ds.dpdy.isel(zl=n-1))
                S_eta_x[:,n,:,:] = barsqx - barxx
                S_eta_y[:,n,:,:] = barsqy - baryy
        return S_eta_x, S_eta_y    

def process_SimulationData(SimData):
    SimData.calc_pressure_terms() #calculate pressure terms
    SimData.calc_hbar() #calculate filtered layer thicknesses both with and without filtered bottom topography
    ds_interp = SimData.centre_interp() #interpolate data to centrepoint
    
    # Calculating filtered variables of interest along with TWA quantities of interest
    ds_interp['ubar'] = SimData.apply_hfilter(ds_interp.u)
    ds_interp['vbar'] = SimData.apply_hfilter(ds_interp.v)
    ds_interp['uhbar'], ds_interp['uhat'] = SimData.varbar_comp(ds_interp.u, return_TWA = True)
    ds_interp['vhbar'], ds_interp['vhat'] = SimData.varbar_comp(ds_interp.v, return_TWA = True)
    ds_interp['uuhbar'], ds_interp['uuhat'] = SimData.varbar_comp(ds_interp.u**2, return_TWA = True)
    ds_interp['uvhbar'], ds_interp['uvhat'] = SimData.varbar_comp(ds_interp.u*ds_interp.v, return_TWA = True)
    ds_interp['vvhbar'], ds_interp['vvhat'] = SimData.varbar_comp(ds_interp.v**2, return_TWA = True)
    
#     # Adding the Reynolds stress terms, the subfilter PE correction terms, and the dual form stress terms
    SimData.Reynolds_calc(ds_interp)
    ds_interp['intfc_var'] = SimData.compute_intfc_var()
    ds_interp['S_eta_x'],ds_interp['S_eta_y'] = SimData.compute_dual_form(ds_interp)
    ds_interp['Formx'] = ds_interp.Ruu.copy(deep = True) * 0
    ds_interp['Formy'] = ds_interp.Ruu.copy(deep = True) * 0
    for n in np.arange(SimData.N_layers):
        ds_interp.Formx[:,n,:,:] = (ds_interp.S_eta_x.isel(zi = n) - ds_interp.S_eta_x.isel(zi = n+1))
        ds_interp.Formy[:,n,:,:] = (ds_interp.S_eta_y.isel(zi = n) - ds_interp.S_eta_y.isel(zi = n+1))
    #SimData.data['subPE'] = SimData.compute_intfc_var()
    ds_interp['ebar'] = SimData.apply_hfilter(ds_interp.e)
    ds_interp['hmask'] = SimData.mask()
    #return ds_interp
    ### Now coarsening the data
    ds_coarse = xr.Dataset()
    for var in list(ds_interp.keys()):
        # print(var)
        ds_coarse[var] = SimData.coarsen_t(ds_interp[var])
    
    ### Now calculating coarse derivative variables
    ds_coarse['hbarx'] = SimData.coarse_deriv(ds_interp.hbar, 'X')
    ds_coarse['hbary'] = SimData.coarse_deriv(ds_interp.hbar, 'Y')
    ds_coarse['uhatx'] = SimData.coarse_deriv(ds_interp.uhat, 'X')
    ds_coarse['uhaty'] = SimData.coarse_deriv(ds_interp.uhat, 'Y')
    ds_coarse['vhatx'] = SimData.coarse_deriv(ds_interp.vhat, 'X')
    ds_coarse['vhaty'] = SimData.coarse_deriv(ds_interp.vhat, 'Y')
    ds_coarse['ubarx'] = SimData.coarse_deriv(ds_interp.ubar, 'X')
    ds_coarse['ubary'] = SimData.coarse_deriv(ds_interp.ubar, 'Y')
    ds_coarse['vbarx'] = SimData.coarse_deriv(ds_interp.vbar, 'X')
    ds_coarse['vbary'] = SimData.coarse_deriv(ds_interp.vbar, 'Y')

    ds_coarse['ebarx'] = SimData.coarse_deriv(ds_interp.ebar,'X')
    ds_coarse['ebary'] = SimData.coarse_deriv(ds_interp.ebar,'Y')

    #Computing additional terms for normalisation of targets etc
    ds_coarse['Rd'] = SimData.Rd_compute(ds_coarse.hbar, SimData.coarse_grid, SimData.ds_coarse_grid)
    ds_coarse['Delsq'] = SimData.coarse_grid_scale()

    ### Finally, computing the coarse grid data mask
    check = ds_coarse['hmask'] < 1
    ds_coarse['hmask'] = np.abs(check - 1)
    ds_coarse['coarse_mask'] = SimData.mask(coarse=True, h=ds_coarse.hbar) 
    ds_coarse['f'] = SimData.centre_interp_Coriolis(SimData.coarse_grid, SimData.ds_coarse_grid)
    if SimData.debug:
        for idx, var in enumerate(list(ds_coarse.keys())):
            print(idx)
            try:
                ds_coarse[var].compute()
            except:
                print(f"computation of {var} fails")
    return ds_interp, ds_coarse

class create_MLdataset:
    """
    This class takes a list of single time snapshots (processed DG data), and compiles one large dataset (masking etc already done) that will then be sent to a torch dataset (eventually). It is assumed that Coriolis information already exists in the snapshot files being loaded. 
    """
    def __init__(self, 
                 data_files,  
                 coarsen_fac = None, 
                 gprime = [0.98, 0.0098], 
                 one_data_mask = True, 
                 mask_name = 'coarse_mask',
                 vel_inputs = ['vort','stretch','strain'],
                 buoy_inputs = {'type': 'intfc_slopes', 'method': 'full'}, 
                 output_list = ['Ruu','Ruv','Rvv','Formx','Formy'], 
                 N_mom_out = 3,
                 N_buoy_out = 2,
                 no_reshape = False,
                 all_layers = True,
                 layer_sel = None, 
                 apply_input_norm = False,
                 apply_land_mask = True,
                 mask_zeros = True,
                 include_print_statements = True
                ):
        super().__init__()
        self.data_files = data_files
        self.vel_inputs = vel_inputs
        self.buoy_inputs = buoy_inputs
        self.outputs = output_list
        self.all_layers = all_layers
        
        if include_print_statements:
            print("opening all snapshot files as one xarray dataset")
        full_data = xr.open_mfdataset(data_files, combine='nested', concat_dim = 'Time')
        if include_print_statements:
            print("Defining the data mask (based off of land values)")
        if one_data_mask:
            # if using one data mask, select the mask from the bottom-most layer
            hmask = full_data[mask_name].isel(zl=-1).drop('zl')
            #hmask = hmask.expand_dims(dim = {'zl': len(full_data[mask_name]['zl'])})
            #hmask['zl'] = full_data[mask_name]['zl']
        else:
            hmask = full_data[mask_name]
        self.hmask = hmask
        if apply_land_mask:
            full_data = full_data * self.hmask
        
        if include_print_statements:
            print("Defining the velocity gradient inputs")
        full_data['vort'], full_data['stretch'], full_data['strain'] = self._mom_input_comp(full_data)
        # now, defining the momentum norm - using L2 norm
        self.momnorm = np.sqrt((full_data[vel_inputs]**2).to_array().sum("variable"))

        if include_print_statements:
            print("Defining buoyancy related inputs")     
        if buoy_inputs['type']=='intfc_slopes':
            if include_print_statements:
                print("Defining the interface gradient inputs")
            temp = self._egrad_input_comp(full_data)

        else:
            if include_print_statements:
                print("Defining the thickness gradient inputs")
            temp = self._hgrad_input_comp(full_data)
        # now, defining the buoyancy norm - using L2 norm
        self.buoynorm = np.sqrt((temp**2).to_array().sum("variable"))

        if apply_input_norm:
            x = xr.merge([full_data[vel_inputs] / self.momnorm, temp / self.buoynorm])
        else:
            x = xr.merge([full_data[vel_inputs], temp])
        y = full_data[output_list]
        self.data = xr.merge([x, y])

        self.N_inputs = len(x)
        self.input_idxs = np.arange(0,self.N_inputs,1)
        self.N_outputs = len(y)
        self.output_idxs = np.arange(self.N_inputs,self.N_inputs + self.N_outputs,1)
        
        self.mom_in_idx = np.arange(0,len(self.vel_inputs),1)
        self.buoy_in_idx = np.arange(len(self.vel_inputs), len(temp) + len(self.vel_inputs), 1)
        self.mom_out_idx = np.arange(self.buoy_in_idx[-1] + 1, self.buoy_in_idx[-1] + 1 + N_mom_out, 1)
        self.buoy_out_idx = np.arange(self.mom_out_idx[-1] + 1, self.mom_out_idx[-1] + 1 + N_buoy_out, 1)
        
        self.data['unorm'] = self.momnorm
        self.unorm_idx = self.buoy_out_idx[-1] + 1
        self.data['bnorm'] = self.buoynorm
        self.bnorm_idx = self.unorm_idx + 1
        self.data['f'] = self._add_dimension(full_data.f, ['zl'])
        self.f_idx = self.bnorm_idx + 1
        self.data['Rd'] = self._add_dimension(full_data.Rd, ['zl'])
        self.Rd_idx = self.f_idx + 1
        self.data['Delsq'] = self._add_dimension(full_data.Delsq, ['zl'])
        self.Delsq_idx = self.Rd_idx + 1        
        self.data['gp'] = xr.DataArray(data=gprime,
                                       dims = ['zl'],
                                       coords=dict(
                                           zl=('zl', self.data['zl'].data)
                                       )
                                      )
        self.data['gp'] = self._add_dimension(self.data['gp'],['Time', 'yh', 'xh'])
        self.gp_idx = self.Delsq_idx + 1
        self.data = self.data.transpose(*self.data[vel_inputs[0]].dims)
        

        if no_reshape:
            if include_print_statements:
                print("you can find the processed data in self.data")
        else:
            if include_print_statements:
                print("sending ds to numpy array for reshaping")
            if self.all_layers:
                if include_print_statements:
                    print("reshaping numpy array to include all layers")
                ds_np = self.data.to_array().to_numpy()
        
                dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
                dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
                dimZ = np.shape(ds_np)[-3]   # Points in Z axis (layer)
                dimT = np.shape(ds_np)[-4]   # Points in T axis (snapshot)
                dimF = np.shape(ds_np)[0]   # total number of features/needed data in dataset
                ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY * dimZ * dimT).transpose()
            else:
                self.layer_sel = layer_sel
                ds_np = self.data.isel(zl = layer_sel).to_array().to_numpy()
        
                dimX = np.shape(ds_np)[-1]   # Points in X axis (long)
                dimY = np.shape(ds_np)[-2]   # Points in Y axis (lat)
                dimT = np.shape(ds_np)[-3]   # Points in T axis (snapshot)
                dimF = np.shape(ds_np)[0]   # total number of features/needed data in dataset
                ds_np_reshape = ds_np.astype("float32").reshape(dimF, dimX * dimY * dimT).transpose()
            self.ds_np = ds_np

            
            if mask_zeros:
                if include_print_statements:
                    print("Now masking all 0 and nan samples")
                ds_np_reshape[ds_np_reshape == 0.] = np.nan
            nansum = np.sum(ds_np_reshape, axis = 1)
            nan_inds = np.isnan(nansum)
    
            masked_data = ds_np_reshape[~nan_inds]
    
            if include_print_statements:
                print("Creating all needed torch tensors")
            x_out = torch.tensor(masked_data[:,self.input_idxs], dtype = torch.float32)
            y_out = torch.tensor(masked_data[:,self.output_idxs], dtype = torch.float32)
            unorm = torch.tensor(masked_data[:,self.unorm_idx], dtype = torch.float32)
            bnorm = torch.tensor(masked_data[:,self.bnorm_idx], dtype = torch.float32)
            gridsq = torch.tensor(masked_data[:,self.Delsq_idx], dtype = torch.float32)
            f = torch.tensor(masked_data[:,self.f_idx], dtype = torch.float32)
            gprime = torch.tensor(masked_data[:,self.gp_idx], dtype = torch.float32)

            if include_print_statements:
                print("Sending all torch tensors as a list of tensors in self.torch_data")
            self.torch_data = torch.utils.data.TensorDataset(x_out, y_out, unorm, bnorm, gridsq, f, gprime)

    def _mom_input_comp(self, data):
        """
        Compute the normalisation for the momentum inputs
        """
        vort = data.vhatx - data.uhaty
        stretch = data.uhatx - data.vhaty
        strain = data.uhaty + data.vhatx
        return vort, stretch, strain

    def _egrad_input_comp(self, data):
        """
        Compute the normalisation for the buoyancy inputs
        """
        top = xr.Dataset()
        bot = xr.Dataset()
    
        top['extop'] = data.ebarx.isel(zi = slice(0,-1,1))
        top['eytop'] = data.ebary.isel(zi = slice(0,-1,1))
        try:
            top = top.rename({'zi':'zl'})
            top['zl'] = data['zl']
        except:
            top = top.drop('zl').rename({'zi':'zl'})
            top['zl'] = data['zl']

        if self.buoy_inputs['method'] == 'full':
            bot['exbot'] = data.ebarx.isel(zi = slice(1,len(data['zi']),1))
            bot['eybot'] = data.ebary.isel(zi = slice(1,len(data['zi']),1))
        else:
            the_bot = data['zi'] == data['zi'][-1] #only 1 for bottom interface
            not_bot = data['zi'] != data['zi'][-1] #only 0 for bottom interface
            bot['exbot_avg'] = (data.ebarx*the_bot).isel(zi = slice(1,len(data['zi']),1))
            bot['eybot_avg'] = (data.ebary*the_bot).isel(zi = slice(1,len(data['zi']),1))
            bot['exbot_dev'] = (data.ebarx*not_bot).isel(zi = slice(1,len(data['zi']),1))
            bot['eybot_dev'] = (data.ebary*not_bot).isel(zi = slice(1,len(data['zi']),1))
        try:    
            bot = bot.rename({'zi':'zl'})
            bot['zl'] = data['zl']
        except:
            bot = bot.drop('zl').rename({'zi':'zl'})
            bot['zl'] = data['zl']
        
        return xr.merge([top, bot])
        
    def _hgrad_input_comp(self, data):
        """
        Compute the normalisation for the buoyancy inputs (when using thickness gradients)
        """
        temp = xr.Dataset()
        if self.buoy_inputs['method'] == 'full':
            temp['hx'] = data.hbarx
            temp['hy'] = data.hbary
        else:
            print('error: this method for thickness gradient inputs is not done yet')
            temp['hx_avg'] = data.hbarx
            temp['hy_avg'] = data.hbary
            temp['hx_dev'] = data.hbarx
            temp['hy_dev'] = data.hbary
            the_bot = data['zl'] == data['zl'][-1] #only 1 for bottom layer
            not_bot = data['zil'] != data['zl'][-1] #only 0 for bottom layer
        return temp
    def _add_dimension(self, var, dim):
        """
        This function just cleans up the process of adding coriolis, gprime, and delsq to dataset. 
        In each case the layer information is missing (time, xh, and yh for gprime), and these dimensions need 
        to be added. target_dims gives the taret organisation of the dimensions
        """
        for i, d in enumerate(dim):
            var = var.expand_dims(dim = {d: len(self.data[d])})#.transpose(*target_dims)
        return var

class MLDataset(Dataset):
    """
    Dataset for use in training, validation, and testing of ANNs. Should be agnostic to number of input and output channels.
    Sensitive to normalisation data defined, however. 
    """
    def __init__(self,data_tensor):
        """
        tensor containing data - assume that the inputs are already normalised
        """
        
        super().__init__()
        self.x_data=data_tensor[0]
        self.y_data=data_tensor[1]
        self.modu = data_tensor[2]
        self.modb = data_tensor[3]
        self.gridsq = data_tensor[4]
        self.f = data_tensor[5]
        self.gprime = data_tensor[6]
        # self.batch_size = batch_size
        self.len = len(self.x_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx], self.y_data[idx], self.modu[idx], self.modb[idx], self.gridsq[idx], self.f[idx], self.gprime[idx]
    
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
