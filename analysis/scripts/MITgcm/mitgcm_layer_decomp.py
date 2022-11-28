"""
Calculate layerwise diagnostics from MITgcm simulations.
"""

import xarray as xr
import xmitgcm
import xgcm
import numpy as np
import ipywidgets
import dask
import cmocean
import pandas as pd
from xhistogram.xarray import histogram
import os
import gsw

import climtas.nci

# Ignore warnings
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

def cumsum_from_bottom (transposrts, dim='sigma2'):
    cumsum= (transposrts.cumsum(dim)-transposrts.sum(dim))
    return cumsum

def bin_fields(model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                output_dir='sigma_space_output/',
                Tref=0):
    """
    Convert z-coordinate fields to isopycnal coordinate.
    """


    # make sure the output_dir exists
    os.makedirs(os.path.join(model_dir, output_dir), exist_ok=True)

    # load desired iterations from iters.txt file
    iters = np.fromfile(os.path.join(model_dir, 'iters.txt'), dtype=int, sep=',')
    iters = iters.tolist()
    # to use only the last 10 years
    iters = iters[-720:]

    # load model data
    ds_2d = xmitgcm.open_mdsdataset(data_dir=os.path.join(model_dir, 'Diags'), grid_dir=model_dir,
                                prefix=['2D_diags'], delta_t=500, calendar='360_day',
                                ref_date='2000-1-1 0:0:0', geometry='Cartesian',
                               iters=iters)

    ds_seaice = xmitgcm.open_mdsdataset(data_dir=os.path.join(model_dir, 'Diags'), grid_dir=model_dir,
                                    prefix=['seaIceDiag'], delta_t=500, calendar='360_day',
                                    ref_date='2000-1-1 0:0:0', geometry='Cartesian',
                               iters=iters)

    ds_heat = xmitgcm.open_mdsdataset(data_dir=os.path.join(model_dir, 'Diags'), grid_dir=model_dir,
                                      prefix=['heat_3D'], delta_t=500, calendar='360_day',
                                      ref_date='2000-1-1 0:0:0', geometry='Cartesian',
                               iters=iters)

    ds_state = xmitgcm.open_mdsdataset(data_dir=os.path.join(model_dir, 'Diags'), grid_dir=model_dir,
                                   prefix=['state'], delta_t=500, calendar='360_day',
                                   ref_date='2000-1-1 0:0:0', geometry='Cartesian',
                               iters=iters)

    # define xgcm grid object
    ds_state['drW'] = ds_state.hFacW * ds_state.drF #vertical cell size at u point
    ds_state['drS'] = ds_state.hFacS * ds_state.drF #vertical cell size at v point
    ds_state['drC'] = ds_state.hFacC * ds_state.drF #vertical cell size at tracer point

    metrics = {
        ('X',): ['dxC', 'dxG'], # X distances
        ('Y',): ['dyC', 'dyG'], # Y distances
        ('Z',): ['drW', 'drS', 'drC'], # Z distances
        ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
    }

    grid = xgcm.Grid(ds_state, periodic=['X'], metrics=metrics)


    # Calculate layerwise fluxes using 5 day outputs
    time_range = (ds_state['time'][-720], ds_state['time'][-1])

    # masks to select only wet cells
    # tracer points
    mask_TP = xr.where(ds_state['hFacC']>0, 1, 0)
    # meridional velocity points
    mask_VP = xr.where(ds_state['hFacS']>0, 1, 0)

    # For TEOS-80 models, McDougall et al. (2021) recommends scaling model salinity by
    # ups to convert it to preformed salinity. And then just use model variables as inputs to
    # the gsw calculations.

    # calc potential density
    # Should possibly be sigma1, since the dynamics of interest happen at about that depth.
    sigma2 = gsw.density.sigma2(ds_state['SALT'].where(mask_TP)*gsw.gibbs.constants.uPS,
                            ds_state['THETA'].where(mask_TP))
    sigma2.name = 'sigma2'

    #move to required location for interpolation
    sigma2_yp1 = grid.interp(sigma2, 'Y', boundary='extend')
    sigma2_yp1_zp1 = grid.interp(sigma2_yp1, 'Z', boundary='extend')#, to='outer')

    sigma2_yp1_bar = sigma2_yp1.sel(time=slice(time_range[0], time_range[1])).mean(dim='time').compute()
    sigma2_yp1_zp1_bar = grid.interp(sigma2_yp1_bar, 'Z', boundary='extend', to='outer')

    # sigma layer bounds
    sigma2_layer_bounds = np.append(np.linspace(31.5,36.8, 53, endpoint=False), np.linspace(36.8, 36.95,3, endpoint=False))
    sigma2_layer_bounds = np.append(sigma2_layer_bounds, np.linspace(36.95, 37.1, 6, endpoint=False))
    sigma2_layer_bounds = np.append(sigma2_layer_bounds, np.linspace(37.1, 38, 10))

    sigma2_layer_midpoints = (sigma2_layer_bounds[1:] + sigma2_layer_bounds[:-1])/2

    # define time mean vvel at constant depth
    vbar = ds_state['VVEL'].sel(time=slice(time_range[0], time_range[1])).mean(dim=['time']).compute()

    meridional_volume_flux = ds_state['VVEL']*ds_state['drF']*ds_state['dxG']*ds_state['hFacS']
    heat_content = grid.interp(ds_state['THETA']*ds_state['drF']*ds_state['hFacC'], 'Y', boundary='extend')
    vertical_coordinate = xr.ones_like(ds_state['VVEL'])*ds_state['drF']

    #
    layerwise_merid_vol_flux = histogram(sigma2,
                              bins=[sigma2_layer_bounds],
                              dim = ['Z'],
                              weights=meridional_volume_flux).rename({'sigma2_bin':'sigma2'})

    psi = layerwise_merid_vol_flux.mean(dim='time').compute().rename('psi')

    # This is the traditional way of computing the Eulerian-mean overturning
    # Can also be done by computing the layerwise velocity and then averaging.
    psi_bar = histogram(sigma2_bar,
                          bins=[sigma2_layer_bounds],
                          dim = ['Z'],
                          weights=vbar*ds_state['drF']*ds_state['dxG']*
                                    ds_state['hFacS']).rename({'sigma2_bin':'sigma2'}).rename('psi_bar')

    # Eddy overturning
    psi_star = psi - psi_bar

    # Eulerian-mean layer thickness.
    hbar = histogram(sigma2_bar,
                          bins=[sigma2_layer_bounds],
                          dim = ['Z'],
                          weights=xr.ones_like(vbar)*ds_state['drF']).rename({'sigma2_bin':'sigma2'}).rename('hbar')

    X_layered_plotting = np.tile(ds_state['YG'], (len(sigma2_layer_bounds)-1,1)).T
    Y_layered_plotting = -hbar.mean(dim=['XC']).cumsum(dim='sigma2').compute()

    if plotting:
        # isopycnal space
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        levels=np.linspace(-3,3,30,)*1e6

        cumsum_from_bottom(psi.sum(dim='XC')).plot.contourf(ax=ax[0], y='sigma2', yincrease=False, robust=True, levels=levels)
        ax[0].set_title('$\Psi$')

        cumsum_from_bottom(psi_bar.sum(dim='XC')).plot.contourf(ax=ax[1], y='sigma2', yincrease=False, robust=True, levels=levels)
        ax[1].set_title('$\overline{\Psi}$')

        cumsum_from_bottom(psi_star.sum(dim='XC')).plot.contourf(ax=ax[2], y='sigma2', yincrease=False, robust=True, levels=levels)
        ax[2].set_title('$\Psi^{*}$')

        fig.savefig(output_dir+'/psi_eddy_mean_sigma_space.png', dpi=200, bbox_inches='tight')

        # depth space
        fig, ax = plt.subplots(1,3, figsize=(15,4))

        im = ax[0].contourf(X_layered_plotting, Y_layered_plotting,
                    cumsum_from_bottom(psi.sum(dim='XC')), cmap='RdBu_r', vmin=-3.5e6, vmax=3.5e6)
        ax[0].set_title('$\Psi$')
        plt.colorbar(im, ax=ax[0])

        im2 = ax[1].contourf(X_layered_plotting, Y_layered_plotting,
                    cumsum_from_bottom(psi_bar.sum(dim='XC')), cmap='RdBu_r', vmin=-3.5e6, vmax=3.5e6)
        ax[1].set_title('$\overline{\Psi}$')
        plt.colorbar(im2, ax=ax[1])

        im3 = ax[2].contourf(X_layered_plotting, Y_layered_plotting,
                    cumsum_from_bottom(psi_star.sum(dim='XC')), cmap='RdBu_r', vmin=-3.5e6, vmax=3.5e6)
        ax[2].set_title('$\Psi^{*}$')
        plt.colorbar(im3, ax=ax[2])

        # blue rotates counter-clockwise
        fig.savefig(output_dir+'/psi_eddy_mean_depth_space.png', dpi=200, bbox_inches='tight')


    # Zonal-mean and zonal perturbation streamfunctions
    psi_zm = xr.ones_like(psi_bar)*histogram(sigma2_bar.mean(dim='XC'),
                                          bins=[sigma2_layer_bounds],
                                          dim = ['Z'],
                                          weights=(vbar*ds_state['drF']*ds_state['dxG']
                                                   *ds_state['hFacS']).mean(dim='XC')).rename(
                                                {'sigma2_bin':'sigma2'}).rename('psi_zm')

    psi_zp = psi_bar - psi_zm


    if plotting:
        # isopycnal space
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        cumsum_from_bottom(psi_bar.sum(dim='XC')).plot(ax=ax[0], y='sigma2', yincrease=False, robust=True)
        ax[0].set_title('$\overline{\Psi}$')

        cumsum_from_bottom(psi_zm.sum(dim='XC')).plot(ax=ax[1], y='sigma2', yincrease=False, robust=True)
        ax[1].set_title('$\Psi_{zm}$')


        cumsum_from_bottom(psi_zp.sum(dim='XC')).plot(ax=ax[2], y='sigma2', yincrease=False, robust=True)
        ax[2].set_title('$\Psi_{zp}$ - zonal perturbation')

        fig.savefig(output_dir+'/psi_bar_decompostion_sigma_space.png', dpi=200, bbox_inches='tight')

        # depth space
        fig, ax = plt.subplots(1,3, figsize=(15,4))

        im = ax[0].contourf(X_layered_plotting, Y_layered_plotting,
                    cumsum_from_bottom(psi_bar.sum(dim='XC')), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[0].set_title('$\overline{\Psi}$')
        plt.colorbar(im, ax=ax[0])

        im2 = ax[1].contourf(X_layered_plotting, Y_layered_plotting,
                    cumsum_from_bottom(psi_zm.sum(dim='XC')), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[1].set_title('$\Psi_{zm}$')
        plt.colorbar(im2, ax=ax[1])

        im3 = ax[2].contourf(X_layered_plotting, Y_layered_plotting,
                    cumsum_from_bottom(psi_zp.sum(dim='XC')), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[2].set_title('$\Psi_{zp}$')
        plt.colorbar(im3, ax=ax[2])

        # blue rotates counter-clockwise
        fig.savefig(output_dir+'/psi_bar_decompostion_depth_space.png', dpi=200, bbox_inches='tight')

    # volume transport
    vh_prime = (layerwise_merid_vol_flux.sel(time=slice(time_range[0], time_range[1])) - psi)

    # Temperature
    # Need to use the linear transformation from xgcm for temperature, since it is not an
    # extensive property. Using xhistogram results in some weird values.
    # linear interpolation from xgcm
    layerwise_temperature = grid.transform(grid.interp(ds_state['THETA'] - Tref, 'Y', boundary='extend'),
                                              'Z',
                                              sigma2_layer_midpoints,
                                              method='linear',
                                              target_data=sigma2,
                                           mask_edges=False)

    Tbar = ds_state['THETA'].mean(dim='time').load()

    # can get this either by averaging T in z space and converting, or
    # by converting to layers and then averaging. Averaging in layers
    # means that isopycnal heave doesn't smear the temperature signal.
    layerwise_Tbar = layerwise_temperature.mean(dim='time')

                        # grid.transform(grid.interp(Tbar - Tref, 'Y', boundary='extend'),
                        #                       'Z',
                        #                       sigma2_layer_midpoints,
                        #                       method='linear',
                        #                       target_data=sigma2_bar,
                        #                    mask_edges=False).compute()


    if plotting:
        layerwise_Tbar.where(hbar>0.01).mean(dim='XC').plot(y='sigma2', yincrease=False)
        plt.savefig(output_dir+'/layerwise_Tbar_sigma_space.png', dpi=200, bbox_inches='tight')

        plt.pcolormesh(X_layered_plotting, Y_layered_plotting,
                    layerwise_Tbar.where(hbar>0.01).mean(dim='XC'),
                    cmap=cmocean.cm.thermal, vmin=-2, vmax=15)
        plt.colorbar()
        plt.savefig(output_dir+'/layerwise_Tbar_depth_space.png', dpi=200, bbox_inches='tight')

    # heat transport
    vhc_reconstructed = layerwise_merid_vol_flux*layerwise_temperature
    vhc_reconstructed_bar = vhc_reconstructed.mean(dim='time').compute()

    Tprime = layerwise_temperature - layerwise_Tbar

    # eddy diffusive transport
    vh_prime_Tprime_bar = (vh_prime*Tprime).mean(dim='time').compute()

    # Advective heat transport
    psi_Tbar = psi*layerwise_Tbar

    # is made up of Eulerian-mean and eddy

    # Eulerian-mean overturning heat transport
    psibar_Tbar = psi_bar*layerwise_Tbar
        # is made up of
        # zonal-mean
    psizm_Tbar = psi_zm*layerwise_Tbar
        # standing meander
    psizp_Tbar = psi_zp*layerwise_Tbar

    # eddy overturning heat ransport
    psistar_Tbar = psi_star*layerwise_Tbar


    # using model output heat advection to test
    # (only works if the reference temperature is set to 0°C)
    layerwise_heat_advection = histogram(sigma2,
                          bins=[sigma2_layer_bounds],
                          dim = ['Z'],
                          weights=ds_heat['ADVy_TH']).rename({'sigma2_bin':'sigma2'})


    vhc_bar = layerwise_heat_advection.sel(time=slice(time_range[0],
                                                  time_range[1])).mean(dim='time').compute()

    # save fields to NetCDF files
    layerwise_merid_vol_flux.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_merid_vol_flux.nc'),
                encoding={'layerwise_merid_vol_flux': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    psi.to_netcdf(os.path.join(model_dir, output_dir, 'psi.nc'),
                encoding={'psi': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    psi_bar.to_netcdf(os.path.join(model_dir, output_dir, 'psi_bar.nc'),
                encoding={'psi_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    hbar.to_netcdf(os.path.join(model_dir, output_dir, 'hbar.nc'),
                encoding={'hbar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    psi_zm.to_netcdf(os.path.join(model_dir, output_dir, 'psi_zm.nc'),
                encoding={'psi_zm': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    layerwise_temperature.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_temperature.nc'),
                encoding={'layerwise_temperature': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    Tbar.to_netcdf(os.path.join(model_dir, output_dir, 'Tbar.nc'),
                encoding={'Tbar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    layerwise_Tbar.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_Tbar.nc'),
                encoding={'layerwise_Tbar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    vhc_reconstructed_bar.to_netcdf(os.path.join(model_dir, output_dir, 'vhc_reconstructed_bar.nc'),
                encoding={'vhc_reconstructed_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    vh_prime_Tprime_bar.to_netcdf(os.path.join(model_dir, output_dir, 'vh_prime_Tprime_bar.nc'),
                encoding={'vh_prime_Tprime_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    # from model output heat advection - Use to check decomposition
    layerwise_heat_advection.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_heat_advection.nc'),
                encoding={'layerwise_heat_advection': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    vhc_bar.to_netcdf(os.path.join(model_dir, output_dir, 'vhc_bar.nc'),
                encoding={'vhc_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    return


if __name__ == '__main__':

    # Start a dask cluster with multiple cores
    climtas.nci.GadiClient()

    # This script calculates binned quantities for a single year.
        #### get year argument that was passed to python script ####

    # import sys

    # year = str(sys.argv[1])

    bin_fields(model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                output_dir='sigma_space_output/',
                Tref=-2)

