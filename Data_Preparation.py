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







class DoubleGyre_Snapshots():
    def __init__(
        self, 
        data_path,
        coarsen_fac,
        FGR = 2.5,
        pad_wet_mask = True,
        gpu = False,
        shape = "gaussian",
        avoid_bottom = True,
        gaf_kwargs = None,
        gen_path = None,
        exp = None,
        GFS = 0.98,
        GINT = 0.0098,
        subsample_time = False,
        time_subsampling = None
    ):

        self.data_path = data_path
        self.coarsen_fac = coarsen_fac
        self.FGR = FGR
        self.filter_fac = coarsen_fac * FGR
        self.pad_wet_mask = pad_wet_mask
        self.gpu = gpu
        self.shape = shape
        self.avoid_bottom = avoid_bottom
        self.path = gen_path
        self.exp = exp
        self.GFS = GFS
        self.GINT = GINT
        self.subsample_time = subsample_time
        self.time_subsampling = time_subsampling
        
        
        #### Running routines ####
        if gaf_kwargs is None:
            print("building the grid information")
            self._build_grids_and_filters() #build regular and coarsened grids
        else:
            print("unpacking the grid information")
            self._unpack_grids_and_filters(gaf_kwargs)
            
        self.data = self._open_data(data_path)
        if self.subsample_time:
            self.data = self.data.isel(Time = slice(0,len(self.data['Time']),self.time_subsampling))

    def _unpack_grids_and_filters(self, gaf_kwargs):
        self.grid = gaf_kwargs['grid']
        self.ds_grid = gaf_kwargs['ds_grid']
        self.grid_nonsym = gaf_kwargs['grid_nonsym']
        self.ds_grid_nonsym = gaf_kwargs['ds_grid_nonsym']
        self.coarse_grid = gaf_kwargs['coarse_grid']
        self.ds_coarse_grid = gaf_kwargs['ds_coarse_grid']
        self.coarse_grid_nonsym = gaf_kwargs['coarse_grid_nonsym']
        self.ds_coarse_grid_nonsym = gaf_kwargs['ds_coarse_grid_nonsym']
        self.hfilter = gaf_kwargs['hfilter']
   
    def mask(self, min_h):
        return self.data.h > min_h
        
    def Reynolds_calc(self, uhat, vhat, uuhat, uvhat, vvhat):
        """ 
        Computes the Reynolds stress terms of the EPF components from thickness weighted variables
        """
        Ruu = uuhat - uhat**2
        Ruv = uvhat - uhat*vhat
        Rvv = vvhat - vhat**2
        return Ruu, Ruv, Rvv

    
    # def compute_PE_correction(self, ds_interp, hbar):
    #     """ 
    #     Computes the sub-grid potential energy term for computation of the EPF components
    #     """
    #     S_eta = hbar.copy()
    #     N = hbar.coords.sizes['zl'] #number of layers
    #     for n in np.arange(N):
    #         if n == 0:
    #             gp = self.GFS
    #         else:
    #             gp = self.GINT
    #         ebar = self._apply_filter((ds_interp.e).isel(zi=n))
    #         eebar = self._apply_filter((ds_interp.e**2).isel(zi=n))
    #         S_eta[:,n,:,:] = (1/(2*hbar.isel(zl=n)))*gp*(eebar - (ebar**2))
    #     return S_eta

    def compute_PE_correction(self, ds_interp, hbar):
        """ 
        Computes the interface variance (used in subPE term)
        """
        S_eta = hbar.copy()
        N = hbar.coords.sizes['zl'] #number of layers
        for n in np.arange(N):
            ebar = self._apply_filter((ds_interp.e).isel(zi=n))
            eebar = self._apply_filter((ds_interp.e**2).isel(zi=n))
            S_eta[:,n,:,:] = (eebar - (ebar**2))
        return S_eta

    
    def compute_dual_form(self, ds_interp):
        """ 
        Computes the dual form stresses for each vertical interface in the domain (number of layers + 1)
        """
        S_eta_x = ds_interp.e.copy() * 0
        S_eta_y = ds_interp.e.copy() * 0
        N = ds_interp.e.coords.sizes['zi'] #number of interfaces
        for n in np.arange(N):
            if n == 0:
                #Assuming that the atmosphere is taken as pressure = 0, then there is no eddy form stress on free surface
                continue
            elif n == (N - 1):
                #Assuming that the bottom topography isn't changing (eta' = 0), there is no eddy form stress at bottom boundary
                continue
            else:
                barsqx = self._apply_filter(ds_interp.e.isel(zi=n) * ds_interp.dpdx.isel(zl=n-1))
                barsqy = self._apply_filter(ds_interp.e.isel(zi=n) * ds_interp.dpdy.isel(zl=n-1))
                barxx = self._apply_filter(ds_interp.e.isel(zi=n)) * self._apply_filter(ds_interp.dpdx.isel(zl=n-1))
                baryy = self._apply_filter(ds_interp.e.isel(zi=n)) * self._apply_filter(ds_interp.dpdy.isel(zl=n-1))
                S_eta_x[:,n,:,:] = barsqx - barxx
                S_eta_y[:,n,:,:] = barsqy - baryy
        return S_eta_x, S_eta_y
        
    def TWA_comp(self, ds_interp, hbar):
        """ 
        Computes the thickness weighted variables needed for computation of EPF components
        """
        uhat = self.hat_comp(ds_interp.u, ds_interp, hbar)
        vhat = self.hat_comp(ds_interp.v, ds_interp, hbar)
        uuhat = self.hat_comp(ds_interp.u**2, ds_interp, hbar)
        uvhat = self.hat_comp(ds_interp.u * ds_interp.v, ds_interp, hbar)
        vvhat = self.hat_comp(ds_interp.v**2, ds_interp, hbar)
        return uhat, vhat, uuhat, uvhat, vvhat
        
    def hbar_compute(self, ds_interp):
        hbar = ds_interp.h * 0 #initialising structure of hbar
        if self.avoid_bottom:
            N = hbar.coords.sizes['zl'] #number of layers
            for n in np.arange(N):
                if n == N - 1:
                    hbar[:,n,:,:] = self._apply_filter(ds_interp.e.isel(zi = n)) - ds_interp.e.isel(zi = N)
                else:
                    hbar[:,n,:,:] = self._apply_filter(ds_interp.h.isel(zl = n))
        else:
            hbar = self._apply_filter(ds_interp.h) 
        return hbar
        
    def hat_comp(self, dsvarsel, ds_interp, hbar):
        varbar = ds_interp.h.copy()
        N = hbar.coords.sizes['zl'] #number of layers
        if self.avoid_bottom:
            for n in np.arange(N):
                if n == N - 1: # if in the bottom layer
                    varbar[:,n,:,:] = self._apply_filter(dsvarsel.isel(zl = n) * ds_interp.e.isel(zi=n)) - ds_interp.e.isel(zi=N)*self._apply_filter(dsvarsel.isel(zl=n))
                else:
                    varbar[:,n,:,:] = self._apply_filter((dsvarsel * ds_interp.h).isel(zl=n))
        else:
            varbar = self._apply_filter((dsvarsel * ds_interp.h))
        varhat = varbar / hbar
        return varhat    
        
    def Rd_compute(self, hbar_coarse):
        Omega = 7.2921*10**-5 #radians per second
        lat = self.ds_coarse_grid.yh
        h1 = hbar_coarse.isel(zl = 0)
        h2 = hbar_coarse.isel(zl = 1)
        f = 2*Omega*np.sin(np.pi*lat/180)
        Rd = np.sqrt(self.GINT * h1 * h2 / (h1 + h2)) / f
        return Rd

    def Coriolis_compute(self):
        f = self.coarse_grid_nonsym.interp(self.coarse_grid_nonsym.interp(self.ds_coarse_grid_nonsym.Coriolis,'X'),'Y')
        return f

    def SSH_compute(self, ds_prog):
        SSH = ds_prog.h.copy(deep=True) * 0
        N = ds_prog.h.coords.sizes['zl']
        for n in np.arange(N):
            if n==0:
                SSH[:,n,:,:] = ds_prog.e.isel(zi=n)
            else:
                SSH[:,n,:,:] = ds_prog.e.isel(zi=n) - ds_prog.h.isel(zl=n-1)
        return SSH
  
    def calc_press_terms(self, ds_prog):
        press = ds_prog.h * 0
        N = ds_prog.h.coords.sizes['zl'] #number of layers
        for n in np.arange(N):
            if n==0:
                press[:,n,:,:] = self.GFS*ds_prog.e.isel(zi=n)        
            else:
                press[:,n,:,:] = np.sum(press,axis=1) + self.GINT*ds_prog.e.isel(zi=n)       
        ds_prog['press'] = press

        dp_dx = self.grid_nonsym.derivative(ds_prog.press,'X') #this is FD with derivs now on outside. We want centred diff, but for now will just interpolate to centre
        ds_prog['dpdx'] = dp_dx

        dp_dy = self.grid_nonsym.derivative(ds_prog.press,'Y') #this is FD with derivs now on outside. We want centred diff, but for now will just interpolate to centre
        ds_prog['dpdy'] = dp_dy 
        return ds_prog
        
    def centre_interp(self,ds_prog):
        #ds_interp = self.ds_prog.copy()
        ds_interp = xr.Dataset()
        for var_name in ds_prog:
            vardims = ds_prog[var_name].dims
            if 'xh' in vardims:
                if 'yh' in vardims: # already on xh yh grid
                   ds_interp[var_name] = ds_prog[var_name] 
                else: # originally on xh yq grid
                   print(f'only interpolating {var_name} to centre point in Y') 
                   ds_interp[var_name] = self.grid_nonsym.interp(ds_prog[var_name],"Y")
            else:
                if 'yh' in vardims: # originally on xq yh grid
                    print(f'only interpolating {var_name} to centre point in X') 
                    ds_interp[var_name] = self.grid_nonsym.interp(ds_prog[var_name],"X")
                else: # originally on xq yq grid
                    print(f'first interpolating {var_name} to centre point in Y')
                    ds_interp[var_name] = self.grid_nonsym.interp(ds_prog[var_name],"Y")
                    print(f'and now interpolating {var_name} to centre point in X')
                    ds_interp[var_name] = self.grid_nonsym.interp(ds_interp[var_name],"X")
        return ds_interp    
        
    def _build_grids_and_filters(self):
        ds_prog = self._open_data(self.data_path)
        if self.subsample_time:
            ds_prog = ds_prog.isel(Time = slice(0,len(ds_prog['Time']),self.time_subsampling))
        self.grid_nonsym, self.ds_grid_nonsym, self.grid, self.ds_grid = self.prepare_grid(ds_prog)
        self.ds_coarse_grid, self.coarse_grid, self.ds_coarse_grid_nonsym, self.coarse_grid_nonsym = self.coarsen_grid(self.ds_grid, ds_prog)
        self.hfilter = self.create_filter(self.ds_grid) #create the filter object
        
    def _open_data(self, file):
        year = file[-7:-3]
        try:
            ds_prog = xr.open_dataset(file, decode_times=False).chunk({"Time": 1})
            return ds_prog.isel(xq = slice(1,None), yq=slice(1,None))
        except Exception as e:
            print(f"Error reading dataset: {e}")

    def prepare_grid(self, ds, kwargs={}):   
        ds_grid = xr.open_dataset(f"{self.path}{self.exp}/static.nc").drop("Time")
        ds_grid2 = xr.open_dataset(f"{self.path}{self.exp}/ocean_geometry.nc")
        ds_grid2 = ds_grid2.rename({"lath": "yh", "lonh": "xh", "latq": "yq", "lonq": "xq"})
        assert ds_grid.dyCv.equals(ds_grid2.dyCv)
        assert ds_grid.dxCv.equals(ds_grid2.dxCv)
        assert ds_grid.dxCu.equals(ds_grid2.dxCu)
        assert ds_grid.dyCu.equals(ds_grid2.dyCu)
        assert ds_grid.area_t.equals(ds_grid2.Ah)
        assert ds_grid.wet.equals(ds_grid2.wet)
        ds_grid["dxT"] = ds_grid2["dxT"]
        ds_grid["dyT"] = ds_grid2["dyT"]
        ds_grid["dxBu"] = ds_grid2["dxBu"]
        ds_grid["dyBu"] = ds_grid2["dyBu"]

        if "zi" in ds:
            ds_grid["zi"] = ds["zi"]
        ds_grid["zl"] = ds["zl"]
        ds_grid["Time"] = ds["Time"]  # weirdly, need to add the time as a coord to the grid, otherwise time coord is lost after applying an xgcm operation
        grid = self._make_grid(ds_grid, data = ds, symmetric=True, **kwargs)
   
        ds_grid_nonsym = ds_grid.isel(xq = slice(1,None), yq=slice(1,None))
        grid_nonsym = self._make_grid(ds_grid_nonsym, data = ds, symmetric=False, **kwargs)
        return grid_nonsym, ds_grid_nonsym, grid, ds_grid

    def coarsen_grid(self, ds_grid, ds):
        "ds_grid is assumed to hold symmetric grid information, i.e., len(xq)=len(xh)+1 and len(yq)=len(yh)+1"
        factor = self.coarsen_fac
        ds_grid_coarsened = xr.Dataset()
        var = 'dyCu'
        var_downsampled = ds_grid[var].isel(xq=slice(None, None, factor))
        ds_grid_coarsened[var] = var_downsampled.coarsen(yh=factor).sum()
        var = 'dxCv'
        var_downsampled = ds_grid[var].isel(yq=slice(None, None, factor))
        ds_grid_coarsened[var] = var_downsampled.coarsen(xh=factor).sum()
        var = 'dxT'
        tmp = ds_grid[var].coarsen(xh=factor).sum()
        ds_grid_coarsened[var] = tmp.coarsen(yh=factor).mean()
        var = 'dyT'
        tmp = ds_grid[var].coarsen(yh=factor).sum()
        ds_grid_coarsened[var] = tmp.coarsen(xh=factor).mean()
        
        # infer the remaining length scales from the already computed ones
        # for symmetric grids
        coords = {
            'X': {'center': 'xh', 'outer': 'xq'},
            'Y': {'center': 'yh', 'outer': 'yq'}
        }
    
        coarse_grid = Grid(ds_grid_coarsened, coords=coords)
        ###########   
        ds_grid_coarsened["dxCu"] = coarse_grid.interp(ds_grid_coarsened["dxT"], 'X')
        ds_grid_coarsened["dyCv"] = coarse_grid.interp(ds_grid_coarsened["dyT"], 'Y')    
        ds_grid_coarsened["dxBu"] = coarse_grid.interp(ds_grid_coarsened["dxCv"], 'X')    
        ds_grid_coarsened["dyBu"] = coarse_grid.interp(ds_grid_coarsened["dyCu"], 'Y')
        var = 'Coriolis'
        ds_grid_coarsened[var] = ds_grid[var].isel(xq=slice(None, None, factor), yq=slice(None, None, factor))
        var = 'depth_ocean'
        ds_grid_coarsened[var] = ds_grid[var].coarsen(yh=factor, xh=factor).mean()
        varlist = ['zl', 'zi', 'Time']
        for var in varlist:
            if var in ds_grid:
                ds_grid_coarsened[var] = ds_grid[var]
    
        coarse_grid = self._make_grid(ds_grid_coarsened, data = ds, symmetric=True)
        ds_grid_coarsened_nonsym = ds_grid_coarsened.isel(xq = slice(1,None), yq=slice(1,None))
        coarse_grid_nonsym = self._make_grid(ds_grid_coarsened_nonsym, data = ds, symmetric=False)    
        return ds_grid_coarsened, coarse_grid, ds_grid_coarsened_nonsym, coarse_grid_nonsym


    def _make_grid(self, ds_grid, data, symmetric=True, include_time=True):
        if "zi" in data:
            ds_grid["zi"] = data["zi"]
            z_coords = {'Z': {'center': 'zl', 'outer': 'zi'}}
        else:
            z_coords = {'Z': {'center': 'zl'}}
        ds_grid["zl"] = data["zl"]
        ds_grid["Time"] = data["Time"]  # weirdly, need to add the time as a coord to the grid, otherwise time coord is lost after applying an xgcm operation

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

    def create_filter(self, ds_grid):
        #Creating the filtering object associated with data grid
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
        hfilter = gcm_filters.Filter(
            filter_scale = self.filter_fac,
            dx_min = 1,
            filter_shape = filter_shape,
            grid_type = gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
            grid_vars = {'area' : area, 'wet_mask' : wet_mask}
        )
        return hfilter
    
    def _apply_filter(self, target):
            return self.hfilter.apply(target, dims = ['yh','xh'])

    def coarsen_t(self, var):
        factor = self.coarsen_fac
        var_coarse = (
            (var * self.ds_grid['area_t']).coarsen(xh=factor, yh=factor).sum()
        ) / self.ds_grid['area_t'].coarsen(xh=factor, yh=factor).sum()
        return var_coarse  

    def coarse_deriv(self, var, dim):
        temp = self.coarsen_t(var)
        temp = self.coarse_grid_nonsym.derivative(temp, dim)
        return self.coarse_grid_nonsym.interp(temp, dim)

    def coarse_grid_scale(self):
        Del2 = 2 * self.ds_coarse_grid.dxT**2 * self.ds_coarse_grid.dyT**2 / (self.ds_coarse_grid.dxT**2 + self.ds_coarse_grid.dyT**2)
        return Del2


def Process_DoubleGyre_Snapshots(data, min_h):
    """
    Function which applies the various processing functions from the DoubleGyre_Snapshots class and outputs one coarenned xarray with all desired information for ML
    """
    print("hi! I'm starting!")
    ds_prog = data.calc_press_terms(data.data)
    print("Pressure terms are calculated")

    ds_prog['SSH'] = data.SSH_compute(data.data)
    print("SSH/interface deflection terms are calculated")

    ds_interp = data.centre_interp(ds_prog)
    print("Interpolation complete")
    del ds_prog #no longer needed after this point

    hbar = data.hbar_compute(ds_interp)

    uhat, vhat, uuhat, uvhat, vvhat = data.TWA_comp(ds_interp, hbar)

    S_eta_x, S_eta_y = data.compute_dual_form(ds_interp)
    print("variables for EPF calculated")
    
    EPF_terms = xr.Dataset()
    EPF_terms['Ruu'], EPF_terms['Ruv'], EPF_terms['Rvv'] = data.Reynolds_calc(uhat, vhat, uuhat, uvhat, vvhat)
    print("Reynolds stress calculated")
    
    EPF_terms['subPE'] = data.compute_PE_correction(ds_interp, hbar)
    print("top interface variance(ish) calculated")
    
    EPF_terms['Formx'] = EPF_terms.Rvv.copy() * 0
    EPF_terms['Formy'] = EPF_terms.Rvv.copy() * 0
    N = hbar.coords.sizes['zl'] #number of layers... should only be 2
    for n in np.arange(N):
        if n == 0: # top layer
            EPF_terms.Formx[:,n,:,:] = - (S_eta_x.isel(zi = n+1))
            EPF_terms.Formy[:,n,:,:] = - (S_eta_y.isel(zi = n+1))                   
        else:
            EPF_terms.Formx[:,n,:,:] = (S_eta_x.isel(zi = n)) #- EPF_components.Formx.isel(zi = n+1))
            EPF_terms.Formy[:,n,:,:] = (S_eta_y.isel(zi = n)) #- EPF_components.Formy.isel(zi = n+1))
    print("EPF computation complete")
    ##########################
    # Coarsening output data #
    ##########################
    print("Now coarsening output data")
    coarse_data = xr.Dataset()
    coarse_data['hbar_coarse'] = data.coarsen_t(hbar) #coarsened filtered thickness
    coarse_data['ubar'] = data.coarsen_t(data._apply_filter(ds_interp.u))
    coarse_data['vbar'] = data.coarsen_t(data._apply_filter(ds_interp.u))
        
    for var in list(EPF_terms.keys()):
        print(var)
        coarse_data[var] = data.coarsen_t(EPF_terms[var])
    del EPF_terms #no longer need this in memory
    
    ######################################
    # Calculating data needed for inputs #
    ######################################
    print("Moving on to input data")
    Rd = data.Rd_compute(coarse_data.hbar_coarse)
    Rd = Rd.expand_dims(dim = {'zl': len(data.ds_grid['zl'])})
    Rd["zl"] = coarse_data.hbar_coarse["zl"]
    Coriolis = data.Coriolis_compute()
    Coriolis = Coriolis.expand_dims(dim = {'Time': len(data.ds_grid['Time']), 'zl' : len(data.ds_grid['zl'])})
    Coriolis["zl"] = coarse_data.hbar_coarse["zl"]
    Coriolis["Time"] = coarse_data.hbar_coarse["Time"]
    coarse_data['Coriolis'] = Coriolis
    del Rd, Coriolis #no longer need this hanging around either

    ebar_top = hbar.copy() * 0
    ebar_bot = hbar.copy() * 0
    N = hbar.coords.sizes['zl'] #number of layers
    for n in np.arange(N):
        ebar_top[:,n,:,:] = data._apply_filter((ds_interp.e).isel(zi=n))
        ebar_bot[:,n,:,:] = data._apply_filter((ds_interp.e).isel(zi=n+1))    

    coarse_data['ebar'] = data.coarsen_t(data._apply_filter(ds_interp.e))
    coarse_data['ex_top'] = data.coarse_deriv(ebar_top, 'X')
    coarse_data['ey_top'] = data.coarse_deriv(ebar_top, 'Y')
    coarse_data['ex_bot'] = data.coarse_deriv(ebar_bot, 'X')
    coarse_data['ey_bot'] = data.coarse_deriv(ebar_bot, 'Y')
    coarse_data['SSH'] = data.coarsen_t(data._apply_filter(ds_interp.SSH))
    del ebar_top, ebar_bot #no longer need this in memory

    coarse_data['hbarx'] = data.coarse_deriv(hbar, 'X')
    coarse_data['hbary'] = data.coarse_deriv(hbar, 'Y')
    coarse_data['uhatx'] = data.coarse_deriv(uhat, 'X')
    coarse_data['uhaty'] = data.coarse_deriv(uhat, 'Y')
    coarse_data['vhatx'] = data.coarse_deriv(vhat, 'X')
    coarse_data['vhaty'] = data.coarse_deriv(vhat, 'Y')
    coarse_data['uhat'] = data.coarsen_t(uhat)
    coarse_data['vhat'] = data.coarsen_t(vhat)
    coarse_data['uuhat'] = data.coarsen_t(uuhat)
    coarse_data['uvhat'] = data.coarsen_t(uvhat)
    coarse_data['vvhat'] = data.coarsen_t(vvhat)
    del uhat, vhat, uuhat, uvhat, vvhat #no longer need this in memory

    Del_sq = data.coarse_grid_scale() #harmonic average of gridscale squared
    Del_sq = Del_sq.expand_dims(dim = {'Time': len(data.ds_grid['Time']), 'zl' : len(data.ds_grid['zl'])})
    Del_sq['zl'] = coarse_data.hbar_coarse['zl']
    Del_sq['Time'] = coarse_data.hbar_coarse['Time']
    coarse_data['Del_sq'] = Del_sq
    del Del_sq #no longer need this in memory
    
    ###############################
    # Compute masking information #
    ###############################
    print("computing mask information")
    data_mask = data.mask(min_h)
    data_mask = data.coarsen_t(data_mask) #coarsened mask
    check = data_mask < 1
    coarse_data['data_mask'] = np.abs(check - 1) #masking away any coarsened points with any contribution from the fine scale mask

    return coarse_data

# def Coarsen_DoubleGyre_Snapshots(data, min_h):
#     """
#     Function which applies the various processing functions from the DoubleGyre_Snapshots class and outputs one coarenned xarray with all desired information for comparison with coarse res output
#     """
#     print("hi! I'm starting!")
#     ds_prog = data.calc_press_terms(data.data)
#     print("Pressure terms are calculated")

#     ds_prog['SSH'] = data.SSH_compute(ds_prog)
#     print("SSH/interface deflection terms are calculated")
    
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
