import sys
sys.path.append('/home/kae10022/PythonScripts/')
from ml_helperfuncs import *
import numpy as np
import xarray as xr
import sys
import xrft
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def sub_plot_histogram(data, axis, nbins = 20):
    counts, bins = np.histogram(data, bins = nbins)
    axis.stairs(counts, bins)

def R2_figure_10panels_2rows(testdata, Pred, save_name = None, show = True):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6), width_ratios = [5,5,5,5,6])
    xloc = 15
    yloc = 48
    R2(testdata.y,Pred,'Ruu','Time').isel(zl=0).plot(ax=axs[0][0], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][0].set_title("Ruu");
    axs[0][0].set_xlabel("");
    axs[0][0].set_xticklabels([]);
    axs[0][0].set_ylabel('Latitude ($^{\circ}$N)');
    axs[0][0].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruu',0)))
    R2(testdata.y,Pred,'Ruv','Time').isel(zl=0).plot(ax=axs[0][1], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][1].set_title("Ruv");
    axs[0][1].set_xlabel("");
    axs[0][1].set_xticklabels([]);
    axs[0][1].set_ylabel('');
    axs[0][1].set_yticklabels([]);
    axs[0][1].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruv',0)))
    R2(testdata.y,Pred,'Rvv','Time').isel(zl=0).plot(ax=axs[0][2], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][2].set_title("Rvv");
    axs[0][2].set_xlabel("");
    axs[0][2].set_xticklabels([]);
    axs[0][2].set_ylabel('');
    axs[0][2].set_yticklabels([]);
    axs[0][2].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Rvv',0)))
    R2(testdata.y,Pred,'Formx','Time').isel(zl=0).plot(ax=axs[0][3], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][3].set_title("Formx");
    axs[0][3].set_xlabel("");
    axs[0][3].set_xticklabels([]);
    axs[0][3].set_ylabel('');
    axs[0][3].set_yticklabels([]);
    axs[0][3].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formx',0)))
    R2(testdata.y,Pred,'Formy','Time').isel(zl=0).plot(ax=axs[0][4], vmin = -1, vmax = 1, cmap = "RdBu_r",cbar_kwargs={"label": "R$^2$ for top layer"})
    axs[0][4].set_title("Formy");
    axs[0][4].set_xlabel("");
    axs[0][4].set_xticklabels([]);
    axs[0][4].set_ylabel("");
    axs[0][4].set_yticklabels([]);
    axs[0][4].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formy',0)))
        
    R2(testdata.y,Pred,'Ruu','Time').isel(zl=1).plot(ax=axs[1][0], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][0].set_title("");
    axs[1][0].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][0].set_ylabel('Latitude ($^{\circ}$N)');
    axs[1][0].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruu',1)))
    R2(testdata.y,Pred,'Ruv','Time').isel(zl=1).plot(ax=axs[1][1], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][1].set_title("");
    axs[1][1].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][1].set_ylabel('');
    axs[1][1].set_yticklabels([]);
    axs[1][1].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruv',1)))
    R2(testdata.y,Pred,'Rvv','Time').isel(zl=1).plot(ax=axs[1][2], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][2].set_title("");
    axs[1][2].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][2].set_ylabel('');
    axs[1][2].set_yticklabels([]);
    axs[1][2].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Rvv',1)))
    R2(testdata.y,Pred,'Formx','Time').isel(zl=1).plot(ax=axs[1][3], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][3].set_title("");
    axs[1][3].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][3].set_ylabel('');
    axs[1][3].set_yticklabels([]);
    axs[1][3].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formx',1)))
    R2(testdata.y,Pred,'Formy','Time').isel(zl=1).plot(ax=axs[1][4], vmin = -1, vmax = 1, cmap = "RdBu_r",cbar_kwargs={"label": "R$^2$ for bottom layer"})
    axs[1][4].set_title("");
    axs[1][4].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][4].set_ylabel("");
    axs[1][4].set_yticklabels([]);
    axs[1][4].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formy',1)))
    
    plt.tight_layout()
        
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
    if not show:
        plt.close()




def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
    
def globalR2(testdata, Pred, val, level):
    if R2(testdata.y,Pred,val,{'Time', 'yh','xh'}).isel(zl=level).values < 0:
        return 0.0
    else:
        return trunc(R2(testdata.y,Pred,val,{'Time', 'yh','xh'}).isel(zl=level).values,2)





def meanflow_comp_figure_6panels_2rows(testdata, Pred, save_name = None, show = True):
    fig, axs = plt.subplots(2, 3, figsize=(15, 6), width_ratios = [5,5,5,5,6])
    xloc = 15
    yloc = 48
    R2(testdata.y,Pred,'Ruu','Time').isel(zl=0).plot(ax=axs[0][0], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][0].set_title("Ruu");
    axs[0][0].set_xlabel("");
    axs[0][0].set_xticklabels([]);
    axs[0][0].set_ylabel('Latitude ($^{\circ}$N)');
    axs[0][0].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruu',0)))
    R2(testdata.y,Pred,'Ruv','Time').isel(zl=0).plot(ax=axs[0][1], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][1].set_title("Ruv");
    axs[0][1].set_xlabel("");
    axs[0][1].set_xticklabels([]);
    axs[0][1].set_ylabel('');
    axs[0][1].set_yticklabels([]);
    axs[0][1].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruv',0)))
    R2(testdata.y,Pred,'Rvv','Time').isel(zl=0).plot(ax=axs[0][2], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][2].set_title("Rvv");
    axs[0][2].set_xlabel("");
    axs[0][2].set_xticklabels([]);
    axs[0][2].set_ylabel('');
    axs[0][2].set_yticklabels([]);
    axs[0][2].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Rvv',0)))
    R2(testdata.y,Pred,'Formx','Time').isel(zl=0).plot(ax=axs[0][3], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[0][3].set_title("Formx");
    axs[0][3].set_xlabel("");
    axs[0][3].set_xticklabels([]);
    axs[0][3].set_ylabel('');
    axs[0][3].set_yticklabels([]);
    axs[0][3].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formx',0)))
    R2(testdata.y,Pred,'Formy','Time').isel(zl=0).plot(ax=axs[0][4], vmin = -1, vmax = 1, cmap = "RdBu_r",cbar_kwargs={"label": "R$^2$ for top layer"})
    axs[0][4].set_title("Formy");
    axs[0][4].set_xlabel("");
    axs[0][4].set_xticklabels([]);
    axs[0][4].set_ylabel("");
    axs[0][4].set_yticklabels([]);
    axs[0][4].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formy',0)))
        
    R2(testdata.y,Pred,'Ruu','Time').isel(zl=1).plot(ax=axs[1][0], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][0].set_title("");
    axs[1][0].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][0].set_ylabel('Latitude ($^{\circ}$N)');
    axs[1][0].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruu',1)))
    R2(testdata.y,Pred,'Ruv','Time').isel(zl=1).plot(ax=axs[1][1], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][1].set_title("");
    axs[1][1].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][1].set_ylabel('');
    axs[1][1].set_yticklabels([]);
    axs[1][1].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Ruv',1)))
    R2(testdata.y,Pred,'Rvv','Time').isel(zl=1).plot(ax=axs[1][2], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][2].set_title("");
    axs[1][2].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][2].set_ylabel('');
    axs[1][2].set_yticklabels([]);
    axs[1][2].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Rvv',1)))
    R2(testdata.y,Pred,'Formx','Time').isel(zl=1).plot(ax=axs[1][3], vmin = -1, vmax = 1, cmap = "RdBu_r", add_colorbar = False)
    axs[1][3].set_title("");
    axs[1][3].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][3].set_ylabel('');
    axs[1][3].set_yticklabels([]);
    axs[1][3].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formx',1)))
    R2(testdata.y,Pred,'Formy','Time').isel(zl=1).plot(ax=axs[1][4], vmin = -1, vmax = 1, cmap = "RdBu_r",cbar_kwargs={"label": "R$^2$ for bottom layer"})
    axs[1][4].set_title("");
    axs[1][4].set_xlabel("Longitude ($^{\circ}$E)");
    axs[1][4].set_ylabel("");
    axs[1][4].set_yticklabels([]);
    axs[1][4].text(xloc, yloc, 'R$^2 =$ {0:.2f}'.format(globalR2(testdata,Pred,'Formy',1)))
    
    plt.tight_layout()
        
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
    if not show:
        plt.close()


def meanflow_comp_fig(**kwargs):
    nrow = len(kwargs['data'].keys())
    ncol = len(kwargs['data'][list(kwargs['data'].keys())[-1]].keys())
    hratios = 5*np.ones(nrow)
    hratios[0] += 1
    fig, axs = plt.subplots(nrow, ncol, figsize=kwargs['figsize'], height_ratios = hratios)
    if nrow == 1:
        print(list(kwargs['data'][list(kwargs['data'].keys())[0]].keys()))
        for col, key in enumerate(list(kwargs['data'][list(kwargs['data'].keys())[0]].keys())):
            data =  kwargs['data'][list(kwargs['data'].keys())[0]][key] 

            if 'cmap_override' in list(kwargs.keys()):
                colourpalette = kwargs['cmap_override'][col]
            else:
                if np.min(data) < 0:
                    colourpalette = kwargs['cmaps']['diverging']
                else:
                    colourpalette = kwargs['cmaps']['mono']
            data.plot(ax=axs[col],
                        vmin = kwargs['limits'][col][0],
                        vmax = kwargs['limits'][col][1],
                        cmap = colourpalette,
                        cbar_kwargs={"label": kwargs['cbar_titles'][col],"location": 'top'}
                        )
            axs[col].set_title("");
            axs[col].text(**kwargs['pltlabellocs'], s=kwargs['pltlabels'][col])
            if col > 0:
                axs[col].set_ylabel("");
                axs[col].set_yticklabels([]);
            else:
                axs[col].set_ylabel("Latitude ($^{\circ}$N)");
            axs[col].set_xlabel("Longitude ($^{\circ}$E)");
    else:        
        for row in range(nrow):
            for col, key in enumerate(list(kwargs['data'][list(kwargs['data'].keys())[row]].keys())):
                data =  kwargs['data'][list(kwargs['data'].keys())[row]][key] 
    
                if 'cmap_override' in list(kwargs.keys()):
                    colourpalette = kwargs['cmap_override'][col]
                else:
                    if np.min(data) < 0:
                        colourpalette = kwargs['cmaps']['diverging']
                    else:
                        colourpalette = kwargs['cmaps']['mono']
                
                if row > 0:
                    data.plot(ax=axs[row][col],
                              vmin = kwargs['limits'][col][0],
                              vmax = kwargs['limits'][col][1],
                              cmap = colourpalette,
                              add_colorbar = False
                             )      
                else:
                    data.plot(ax=axs[row][col],
                              vmin = kwargs['limits'][col][0],
                              vmax = kwargs['limits'][col][1],
                              cmap = colourpalette,
                              cbar_kwargs={"label": kwargs['cbar_titles'][col],"location": 'top'}
                             )
                axs[row][col].set_title("");
                axs[row][col].text(**kwargs['pltlabellocs'], s=kwargs['pltlabels'][row][col])
                if col > 0:
                    axs[row][col].set_ylabel("");
                    axs[row][col].set_yticklabels([]);
                else:
                    axs[row][col].set_ylabel("Latitude ($^{\circ}$N)");
                if row == nrow-1:
                    axs[row][col].set_xlabel("Longitude ($^{\circ}$E)");
                else:
                    axs[row][col].set_xlabel("");
                    axs[row][col].set_xticklabels([]);

    if 'fig_name' in list(kwargs.keys()):
        fig.savefig(kwargs['fig_name'], bbox_inches='tight')
                


