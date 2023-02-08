"""
Compute mean tracer and transport fields
"""

# Load modules as normal

import matplotlib.pyplot as plt
import xarray as xr
from xgcm import Grid
import xgcm
import numpy as np
import pandas as pd
import cftime
from xhistogram.xarray import histogram
import cosima_cookbook as cc

import cmocean
import gsw

import IPython.display
import cartopy.crs as ccrs
import cartopy.feature as cft

import sys, os
import warnings
warnings.simplefilter("ignore")
from dask.distributed import Client

import climtas.nci

def calc_psi(year, month, time_slice):
  """
  Calculate overturning stream function in density coordinates.
  """


  # Load data
  sw_ocean = cc.querying.getvar(expt,'sw_ocean', session, n=-1)
  sw_edges_ocean = cc.querying.getvar(expt,'sw_edges_ocean', session, n=-1)
  st_ocean = cc.querying.getvar(expt,'st_ocean', session, n=-1)
  st_edges_ocean= cc.querying.getvar(expt,'st_edges_ocean', session, n=-1)

  dstF = xr.DataArray(st_edges_ocean.values[1:] - st_edges_ocean.values[:-1],
                     coords={'st_ocean':st_ocean},
                     dims={'st_ocean':st_ocean})


  # ty_trans = cc.querying.getvar(expt,'ty_trans', session, start_time = start_time, end_time = end_time)
  # ty_trans = ty_trans.sel(time=time_slice).sel(yu_ocean=lat_range)


  vhrho_nt = cc.querying.getvar(expt,'vhrho_nt', session, start_time = start_time, end_time = end_time)
  vhrho_nt = vhrho_nt.sel(time=time_slice).sel(yt_ocean=lat_range)
  vhrho_nt = vhrho_nt.chunk([1,-1,838,800])

  pot_rho_1 = cc.querying.getvar(expt,'pot_rho_1', session, start_time = start_time, end_time = end_time)
  pot_rho_1 = pot_rho_1.sel(time=time_slice).sel(yt_ocean=lat_range)
  pot_rho_1 = pot_rho_1.chunk([1,-1,838,800])

  potrho_edges = cc.querying.getvar(expt,'potrho_edges', session, n=1,frequency = '1 monthly' )

  pot_rho_1_midpoints = (potrho_edges[1:] + potrho_edges[:-1])/2



  # Define grid and metrics

  ds = xr.Dataset({'pot_rho_1':pot_rho_1,
                   'sw_ocean':sw_ocean, 'st_edges_ocean':st_edges_ocean})
  grid = xgcm.Grid(ds, coords={'X':{'center':'xt_ocean'},
                               'Y':{'center':'yt_ocean'},
                               'Z':{'center':'st_ocean', 'outer':'st_edges_ocean', 'right':'sw_ocean'}},
                               periodic = ['X'])

  # Calculate psi

  psi = histogram(pot_rho_1,
                        bins=[potrho_edges.values],
                        dim = ['st_ocean'],
                        weights=vhrho_nt).rename({'pot_rho_1_bin':'potrho'}).rename('psi')

  psi.load()

  ##saving
  save_dir = 'g/data/jk72/ed7737/SO-channel_embayment/ACESS-OM2/01deg_jra55v13_ryf9091/data/'
  ds = xr.Dataset({'psi': psi})
  ds.to_netcdf(save_dir+'psi_01deg_jra55v13_ryf9091_year_'+str(year)+'_month_'+str(month)+'.nc',
               encoding={'psi': {'shuffle': True, 'zlib': True, 'complevel': 5}})





if __name__ == '__main__':

    climtas.nci.GadiClient()
    session = cc.database.create_session()

    # Constants
    # reference density value:
    rho_0 = 1035.0
    g = 9.81

    # Restrict to Southern Ocean latitudes
    lat_range = slice(-90,-35.5)

    #### get year argument that was passed to python script ####
    import sys
    year = int(sys.argv[1])

    # RYF run with daily outputs
    #   for years 2170-2179, there is global daily temp, salt, pot_rho_1, uhrho_et, vhrho_nt, u, v, wt, dzt.
    expt = '01deg_jra55v13_ryf9091'
    # start_time = '2170-01-01'
    # end_time = '2179-12-31'

    # Run through the four quarters for this year

    # start in Jan (J)
    month = '01'
    start_time = str(year)+'-01-01'
    end_time = str(year)+'-01-31'
    time_slice = slice(start_time, end_time)
    calc_psi(year, month, time_slice)

    # start in April (AMJ)
    month = '04'
    start_time = str(year)+'-04-01'
    end_time = str(year)+'-06-30'
    time_slice = slice(start_time, end_time)
 #   calc_psi(year, month, time_slice)

    # Start in July (JAS)
    month = '07'
    start_time = str(year)+'-07-01'
    end_time = str(year)+'-09-30'
    time_slice = slice(start_time, end_time)
 #   calc_psi(year, month, time_slice)

    # Start in Oct (OND)
    month = '10'
    start_time = str(year)+'-10-01'
    end_time = str(year)+'-12-31'
    time_slice = slice(start_time, end_time)
 #   calc_psi(year, month, time_slice)
