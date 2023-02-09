"""
Calculates vh,uh and h binned into density bins in Southern Ocean for 1 month
"""

# Load modules

# Standard modules
import cosima_cookbook as cc
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import numpy as np
from dask.distributed import Client
import cftime
import glob
import dask.array as dsa
from cosima_cookbook import distributed as ccd
import climtas.nci
from xhistogram.xarray import histogram
import xgcm


# Ignore warnings
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

def bin_fields(year):

    expt = '01deg_jra55v13_ryf9091'

    start_time = year + '-01-01'
    end_time= year + '-12-31'

    # reference density value:
    rho_0 = 1035.0
    g = 9.81

    # Restrict to Southern Ocean latitudes
    lat_range = slice(-85,-35.05)

    #load coordinates
    yt_ocean = cc.querying.getvar(expt,'yt_ocean',session,n=1)
    yt_ocean = yt_ocean.sel(yt_ocean=lat_range)

    yu_ocean = cc.querying.getvar(expt,'yu_ocean',session,n=-1)
    yu_ocean = yu_ocean.sel(yu_ocean=lat_range)

    xt_ocean = cc.querying.getvar(expt,'xt_ocean',session,n=1)

    xu_ocean = cc.querying.getvar(expt,'xu_ocean',session,n=-1)


    for run_count in range(1,13):
        if run_count<10:
            month = '0'+str(run_count)
        elif run_count ==10:
            month = '10'
        elif run_count ==11:
            month = '11'
        elif run_count ==12:
            month = '12'
        else:
            raise ValueError('run_count set incorrectly: {0}'.format(run_count))
        # load time slices


        # time_slice = slice(start_time, end_time)
        time_slice= year + '-' + month


        #load pot_rho_1
        pot_rho_1 = cc.querying.getvar(expt,'pot_rho_1',session,start_time=start_time, end_time=end_time,ncfile='%daily%')
        pot_rho_1 = pot_rho_1.sel(yt_ocean=lat_range).sel(time=time_slice)
        time = pot_rho_1.time

        #load dzt
        dzt = cc.querying.getvar(expt,'dzt',session,start_time=start_time, end_time=end_time,ncfile='%daily%')
        dzt = dzt.sel(yt_ocean=lat_range).sel(time=time_slice)

        #load vhrho and uhrho
        # Note vhrho_nt is v*dz*1035 and is positioned on north centre edge of t-cell.
        uhrho = cc.querying.getvar(expt,'uhrho_et',session,start_time=start_time, end_time=end_time)
        uhrho = uhrho.sel(yt_ocean=lat_range).sel(time=time_slice)

        vhrho = cc.querying.getvar(expt,'vhrho_nt',session,start_time=start_time, end_time=end_time)
        vhrho = vhrho.sel(yt_ocean=lat_range).sel(time=time_slice)


        # swap coordinates for vhrho and uhrho, since they are defined on the faces, not the cell centre.
        uhrho = uhrho.assign_coords(xt_ocean=xu_ocean.values)
        uhrho = uhrho.rename({'xt_ocean':'xu_ocean'})
        # now bring the metadata
        uhrho = uhrho.assign_coords(xu_ocean=xu_ocean)

        vhrho = vhrho.assign_coords(yt_ocean=yu_ocean.values)
        vhrho = vhrho.rename({'yt_ocean':'yu_ocean'})
        # now bring the metadata
        vhrho = vhrho.assign_coords(yu_ocean=yu_ocean)


        ## define isopycnal bins
        isopycnal_bins_sigma1 = 1000+ np.array([1,28,29,30,31,31.5,31.9,32,32.1,32.2,32.25,
                                                    32.3,32.35,32.4,32.42,32.44,32.46,32.48,32.50,32.51,
                                                    32.52,32.53,32.54,32.55,32.56,32.58,32.6,32.8,33,34,45])

        # make a merged dataset for xgcm grid object
        ds = xr.merge([uhrho, vhrho, pot_rho_1, dzt])

        # instantiate xgcm grid object
        metrics = {
            # ('X',): ['dxu', 'dxt'], # X distances
            # ('Y',): ['dyu', 'dyt'], # Y distances
            ('Z',): ['dzt'], # Z distances (varies in time because of MOM's vetical coordinate)
            # ('X', 'Y'): ['area_t', 'area_u'] # Areas
        }

        grid = xgcm.Grid(ds, coords={'X':{'center':'xt_ocean', 'right':'xu_ocean'},
                                     'Y':{'center':'yt_ocean', 'right':'yu_ocean'},
                                     'Z':{'center':'st_ocean',}},
                         periodic = ['X'], metrics=metrics)


        # move uhrho and vhrho to tracer point
        vhrho_T = grid.interp(ds['vhrho_nt'], 'Y')
        uhrho_T = grid.interp(ds['uhrho_et'], 'X')



        uh_binned = histogram(pot_rho_1,
                                  bins = [isopycnal_bins_sigma1],
                                  dim = ['st_ocean'],
                                  weights = uhrho_T).rename({pot_rho_1.name + '_bin': 'potrho',})
                                                             # 'xt_ocean': 'grid_xt_ocean',
                                                             # 'yt_ocean': 'grid_yt_ocean'})

        vh_binned = histogram(pot_rho_1,
                                  bins = [isopycnal_bins_sigma1],
                                  dim = ['st_ocean'],
                                  weights = vhrho_T).rename({pot_rho_1.name + '_bin': 'potrho',})
                                                             # 'xt_ocean': 'grid_xt_ocean',
                                                             # 'yt_ocean': 'grid_yt_ocean'})


        h_binned = histogram(pot_rho_1,
                                  bins = [isopycnal_bins_sigma1],
                                  dim = ['st_ocean'],
                                  weights = dzt).rename({pot_rho_1.name + '_bin': 'potrho',})
                                                             # 'xt_ocean': 'grid_xt_ocean',
                                                             # 'yt_ocean': 'grid_yt_ocean'})

        #save data
        save_dir = '/g/data/jk72/ed7737/SO-channel_embayment/ACESS-OM2/01deg_jra55v13_ryf9091/data/'



        merid_transport = xr.Dataset({'vh_binned': vh_binned})
        merid_transport.to_netcdf(save_dir+'vh_binned_'+year+'-'+month+'.nc',
                                                     encoding={'vh_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

        zonal_transport = xr.Dataset({'uh_binned': uh_binned})
        zonal_transport.to_netcdf(save_dir+'uh_binned_'+year+'-'+month+'.nc',
                                                     encoding={'uh_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

        thickness = xr.Dataset({'h_binned': h_binned})
        thickness.to_netcdf(save_dir+'h_binned_'+year+'-'+month+'.nc',
                                                     encoding={'h_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    return

if __name__ == '__main__':

    # Start a dask cluster with multiple cores
    climtas.nci.GadiClient()

    # client = Client(local_directory='/scratch/jk72/ed7737/')
    # Load database
    session = cc.database.create_session()

    # This script calculates binned quantities for a single year.
        #### get year argument that was passed to python script ####

    import sys

    year = str(sys.argv[1])

    bin_fields(year)

