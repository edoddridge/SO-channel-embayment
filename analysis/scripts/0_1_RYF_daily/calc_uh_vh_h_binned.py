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
    lat_range = slice(-70,-34.99)
    lat_range_big =  slice(-70.05,-34.90)

    #load coordinates
    yt_ocean = cc.querying.getvar(expt,'yt_ocean',session,n=1)
    yt_ocean = yt_ocean.sel(yt_ocean=lat_range)
    xt_ocean = cc.querying.getvar(expt,'xt_ocean',session,n=1)

    #load vhrho and uhrho
    # Note vhrho_nt is v*dz*1035 and is positioned on north centre edge of t-cell.
    vhrho = cc.querying.getvar(expt,'vhrho_nt',session,start_time=start_time, end_time=end_time)
    uhrho = cc.querying.getvar(expt,'uhrho_et',session,start_time=start_time, end_time=end_time)

    # select latitude range and this month:
    vhrho = vhrho.sel(yt_ocean=lat_range_big).sel(time=time_slice)
    uhrho = uhrho.sel(yt_ocean=lat_range).sel(time=time_slice)

    uhrho_E = uhrho[:,:,:,3599]
    uhrho_W = uhrho[:,:,:,0]
    uhrho_E.xt_ocean.values = -280.05
    uhrho_W.xt_ocean.values = 80.05
    uhrho = xr.concat([uhrho_E, uhrho, uhrho_W], dim = 'xt_ocean')

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
    vh_binned = xr.DataArray(np.zeros((len(time),len(isopycnal_bins_sigma1),len(yt_ocean),len(xt_ocean))),
                                                   coords = [time,isopycnal_bins_sigma1, yt_ocean, xt_ocean],
                                                   dims = ['time','isopycnal_bins', 'yt_ocean','xt_ocean'],
                                                   name = 'vh_binned')
    uh_binned = xr.DataArray(np.zeros((len(time),len(isopycnal_bins_sigma1),len(yt_ocean),len(xt_ocean))),
                                                   coords = [time,isopycnal_bins_sigma1, yt_ocean, xt_ocean],
                                                   dims = ['time','isopycnal_bins', 'yt_ocean','xt_ocean'],
                                                   name = 'uh_binned')
    h_binned = xr.DataArray(np.zeros((len(time),len(isopycnal_bins_sigma1),len(yt_ocean),len(xt_ocean))),
                                                   coords = [time,isopycnal_bins_sigma1, yt_ocean, xt_ocean],
                                                   dims = ['time','isopycnal_bins', 'yt_ocean','xt_ocean'],
                                                   name = 'h_binned')
    # loop over time for that month
    for day in range(len(time)):
        print('day '+str(day))

        uhrho_j = uhrho[day,...]
        uhrho_j = uhrho_j.fillna(0)
        uhrho_j = uhrho_j.load()
        vhrho_j = vhrho[day,...]
        vhrho_j = vhrho_j.fillna(0)
        vhrho_j = vhrho_j.load()

        #move uhrho and vhrho to t grid since uhrho is eastern side of cell and vhrho on northern
        uhrho_t = 0.5*(uhrho_j+uhrho_j.roll(xt_ocean=1, roll_coords = False)) #this takes average of adjacent cells
        uhrho_t = uhrho_t[:,:,1:-1]

        vhrho_t = 0.5*(vhrho_j+vhrho_j.roll(yt_ocean=1, roll_coords = False))
        vhrho_t = vhrho_t[:,1:-1,:]
        #load dzt and pot_rho_1 for the "week"
        dzt_j = dzt[day,...]
        dzt_j = dzt_j.fillna(0)
        dzt_j = dzt_j.load()

        #vhrho and uhrho grids are from BAY(dzu) and BAX(dzu) NOT dzt: find these
        dzt_j_right = dzt_j.roll(xt_ocean = -1, roll_coords = False)
        dzt_j_up = dzt_j.roll(yt_ocean = -1,roll_coords = False)
        dzt_j_up_right=dzt_j.roll(yt_ocean = -1,roll_coords = False).roll(xt_ocean = -1, roll_coords = False)
        dzu = np.fmin(np.fmin(np.fmin(dzt_j,dzt_j_right),dzt_j_up),dzt_j_up_right)
        #now the xgrid needs BAY(dzu) while ygrid needs BAX(dzu) so that they are on uhrho and vhrho grids. we only need to do one,
        #as they are equivalent when interpolatred to t-grid. I chose to find BAX because I took dzt to have bigger lon range
        dzu_e = dzu.copy()
        dzu_w = dzu.roll(xt_ocean = 1, roll_coords=False)
        BAX_dzu = (dzu_w+dzu_e)/2

        #as with vhrho these need to be moved to t-grid in the same way:
        dzt_j = 0.5*(BAX_dzu+BAX_dzu.roll(yt_ocean=1, roll_coords = False))
        dzt_j = dzt_j[:,1:-1,:]

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

            ## vh - splits transport between the two bins and saves into vh_binned array
            transport_across_contour_in_sigmalower_bin = ( vhrho_t * bin_mask * bin_fractions).sum(dim = 'st_ocean')
            vh_binned[day,i,:,:] += transport_across_contour_in_sigmalower_bin.fillna(0)
            del transport_across_contour_in_sigmalower_bin
            transport_across_contour_in_sigmaupper_bin = ( vhrho_t * bin_mask * (1-bin_fractions)).sum(dim = 'st_ocean')
            vh_binned[day,i+1,:,:] += transport_across_contour_in_sigmaupper_bin.fillna(0)
            del transport_across_contour_in_sigmaupper_bin

            ## uh
            transport_across_contour_in_sigmalower_bin = ( uhrho_t * bin_mask * bin_fractions).sum(dim = 'st_ocean')
            uh_binned[day,i,:,:] += transport_across_contour_in_sigmalower_bin.fillna(0)
            del transport_across_contour_in_sigmalower_bin
            transport_across_contour_in_sigmaupper_bin = ( uhrho_t * bin_mask * (1-bin_fractions)).sum(dim = 'st_ocean')
            uh_binned[day,i+1,:,:] += transport_across_contour_in_sigmaupper_bin.fillna(0)
            del transport_across_contour_in_sigmaupper_bin

            ## h
            transport_across_contour_in_sigmalower_bin = (dzt_j * bin_mask * bin_fractions).sum(dim = 'st_ocean')
            h_binned[day,i,:,:] += transport_across_contour_in_sigmalower_bin.fillna(0)
            del transport_across_contour_in_sigmalower_bin
            transport_across_contour_in_sigmaupper_bin = (dzt_j * bin_mask * (1-bin_fractions)).sum(dim = 'st_ocean')
            h_binned[day,i+1,:,:] += transport_across_contour_in_sigmaupper_bin.fillna(0)
            del bin_mask, bin_fractions, transport_across_contour_in_sigmaupper_bin

        del pot_rho_1_j, dzt_j, uhrho_j, vhrho_j, uhrho_t, vhrho_t, dzt_j_right,dzt_j_up, dzt_j_up_right, BAX_dzu, dzu_e, dzu_w

    #save monthly data
    save_dir = '/g/data/jk72/ed7737/SO-channel_embayment/ACESS-OM2/01deg_jra55v13_ryf9091/data/'



    ds_vol_trans_across_contour_binned = xr.Dataset({'vh_binned': vh_binned})
    ds_vol_trans_across_contour_binned.to_netcdf(save_dir+'vh_binned_'+year+'-'+month+'.nc',
                                                 encoding={'vh_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    ds_vol_trans_across_contour_binned = xr.Dataset({'uh_binned': uh_binned})
    ds_vol_trans_across_contour_binned.to_netcdf(save_dir+'uh_binned_'+year+'-'+month+'.nc',
                                                 encoding={'uh_binned': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    ds_vol_trans_across_contour_binned = xr.Dataset({'h_binned': h_binned})
    ds_vol_trans_across_contour_binned.to_netcdf(save_dir+'h_binned_'+year+'-'+month+'.nc',
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
    month = 10

    # for run_count in range(1,13):
    bin_fields(month, year)

