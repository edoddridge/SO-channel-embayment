"""
Calculates Temperature and salt binned into density bins in Southern Ocean for 1 month
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

# Ignore warnings
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

def bin_fields(run_count, year):

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

    expt = '01deg_jra55v13_ryf9091'

    time_slice= year + '-' + month
    start_time = year + '-01-01'
    end_time= year + '-12-31'

    # reference density value:
    rho_0 = 1035.0
    # Note: change this range, so it matches the size of your contour arrays:
    ## FULL SO ##
    # lat_range = slice(-90,-34.99)
    # lat_range_big =  slice(-90.0,-34.90)

    lat_range = slice(-85,-34.99)
    lat_range_big =  slice(-85.05,-34.90)

    #load coordinates
    yt_ocean = cc.querying.getvar(expt,'yt_ocean',session,n=1)
    yt_ocean = yt_ocean.sel(yt_ocean=lat_range)
    xt_ocean = cc.querying.getvar(expt,'xt_ocean',session,n=1)

    #load temperature and salt
    temperature = cc.querying.getvar(expt,'temp',session,start_time=start_time, end_time=end_time, frequency='1 daily')
    salt = cc.querying.getvar(expt,'salt',session,start_time=start_time, end_time=end_time, frequency='1 daily')

    # select latitude range and this month:
    temperature = temperature.sel(yt_ocean=lat_range_big).sel(time=time_slice)
    salt = salt.sel(yt_ocean=lat_range).sel(time=time_slice)

    #load pot_rho_1
    pot_rho_1 = cc.querying.getvar(expt,'pot_rho_1',session,start_time=start_time, end_time=end_time,ncfile='%daily%')
    pot_rho_1 = pot_rho_1.sel(yt_ocean=lat_range).sel(time=time_slice)
    time = pot_rho_1.time

    #load dzt
    dzt = cc.querying.getvar(expt,'dzt',session,start_time=start_time, end_time=end_time,ncfile='%daily%')
    dzt = dzt.sel(yt_ocean=lat_range_big).sel(time=time_slice)

    ## define isopycnal bins
    isopycnal_bins_sigma1 = 1000+ np.array([1,28,29,30,31,31.5,31.9,32,32.1,32.2,32.25,
                                                32.3,32.35,32.4,32.42,32.44,32.46,32.48,32.50,32.51,
                                                32.52,32.53,32.54,32.55,32.56,32.58,32.6,32.8,33,34,45])
    ## intialise empty transport along contour in density bins array
    temperature_binned = xr.DataArray(np.zeros((len(time),len(isopycnal_bins_sigma1),len(yt_ocean),len(xt_ocean))),
                                                   coords = [time,isopycnal_bins_sigma1, yt_ocean, xt_ocean],
                                                   dims = ['time','isopycnal_bins', 'yt_ocean','xt_ocean'],
                                                   name = 'temperature_binned')
    salt_binned = xr.DataArray(np.zeros((len(time),len(isopycnal_bins_sigma1),len(yt_ocean),len(xt_ocean))),
                                                   coords = [time,isopycnal_bins_sigma1, yt_ocean, xt_ocean],
                                                   dims = ['time','isopycnal_bins', 'yt_ocean','xt_ocean'],
                                                   name = 'salt_binned')

    # loop over time for that month
    for day in range(len(time)):
        print('day '+str(day))

        temperature_j = temperature[day,...]
        temperature_j = temperature_j.fillna(0)
        temperature_j = temperature_j.load()
        salt_j = salt[day,...]
        salt_j = salt_j.fillna(0)
        salt_j = salt_j.load()


        #load dzt and pot_rho_1 for the "week"
        dzt_j = dzt[day,...]
        dzt_j = dzt_j.fillna(0)
        dzt_j = dzt_j.load()

        pot_rho_1_j = pot_rho_1[day,...]
        pot_rho_1_j = pot_rho_1_j.fillna(0)
        pot_rho_1_j = pot_rho_1_j.load()

        # now bin into density bins
        for i in range(len(isopycnal_bins_sigma1)-1):
            print(i)
            #create masks for isopycnal binnning, that are 1 where the density that day is between two bin values, and 0 elsewhere
            bin_mask = pot_rho_1_j.where(pot_rho_1_j<=isopycnal_bins_sigma1[i+1]).where(pot_rho_1_j>isopycnal_bins_sigma1[i])*0+1
            # create a fractional value that splits the transport between each bin based on which bin it is closer to
            bin_fractions = (isopycnal_bins_sigma1[i+1]-pot_rho_1_j * bin_mask)/(isopycnal_bins_sigma1[i+1]-isopycnal_bins_sigma1[i])

            ## temperature
            temperature_in_sigmalower_bin = ( temperature_j * bin_mask * bin_fractions).sum(dim = 'st_ocean')
            temperature_binned[day,i,:,:] += temperature_in_sigmalower_bin.fillna(0)
            del temperature_in_sigmalower_bin
            temperature_in_sigmaupper_bin = ( temperature_j * bin_mask * (1-bin_fractions)).sum(dim = 'st_ocean')
            temperature_binned[day,i+1,:,:] += temperature_in_sigmaupper_bin.fillna(0)
            del temperature_in_sigmaupper_bin

            ## salt
            salt_in_sigmalower_bin = ( salt_j * bin_mask * bin_fractions).sum(dim = 'st_ocean')
            salt_binned[day,i,:,:] += salt_in_sigmalower_bin.fillna(0)
            del salt_in_sigmalower_bin
            salt_in_sigmaupper_bin = ( salt_j * bin_mask * (1-bin_fractions)).sum(dim = 'st_ocean')
            salt_binned[day,i+1,:,:] += salt_in_sigmaupper_bin.fillna(0)
            del salt_in_sigmaupper_bin


        del pot_rho_1_j, dzt_j

    #save monthly data
    save_dir = '/g/data/jk72/ed7737/SO-channel_embayment/ACESS-OM2/01deg_jra55v13_ryf9091/data/'

    ds_temperature_binned = xr.Dataset({'temperature_binned': temperature_binned})
    ds_temperature_binned.to_netcdf(save_dir+'temperature_binned_'+year+'-'+month+'.nc',
                                                 encoding={'temperature_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    ds_salt_binned = xr.Dataset({'salt_binned': salt_binned})
    ds_salt_binned.to_netcdf(save_dir+'salt_binned_'+year+'-'+month+'.nc',
                                                 encoding={'salt_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

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
    month = 1
    # for run_count in range(1,13):
    bin_fields(month, year)

