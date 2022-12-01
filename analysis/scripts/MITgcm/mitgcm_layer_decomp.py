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
import matplotlib.pyplot as plt

import climtas.nci

# Ignore warnings
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

def cumsum_from_bottom (transposrts, dim='sigma'):
    cumsum= (transposrts.cumsum(dim)-transposrts.sum(dim))
    return cumsum

def load_layer_bounds(sigma, sigma_bar, output_dir):
    """
    Find the layer bounds. Either load or calculate them.
    """
    # Always use control simulation to calculate/load layer bounds
    model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/'

    try:
        sigma_layer_bounds = np.loadtxt(os.path.join(model_dir, output_dir, 'sigma_layer_bounds.txt'))
    except FileNotFoundError:
        sigma_layer_bounds = calc_layer_bounds(sigma, sigma_bar, model_dir=model_dir,
                                output_dir=output_dir)
    return sigma_layer_bounds

def calc_layer_bounds(sigma, sigma_bar, delta_h=200,
                    model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                    output_dir='sigma_space_output/'):
    """
    Calculate layer bounds.
    """
    # Array spanning depth range of the model - currently hard coded.
    z = np.arange(-5, -3995, -delta_h)

    sigma_layer_bounds = sigma_bar.sel(XC=0.4e6, method='nearest').sel(YC=3e6, method='nearest').interp(Z=z).values
    # Capture the lower densities
    n_layers_upper = 11
    sigma_layer_bounds_upper = np.linspace(sigma.min().values, sigma_layer_bounds[0], n_layers_upper, endpoint=False)
    # Capture the larger densities
    n_layers_lower = 21
    sigma_layer_bounds_lower = np.linspace(sigma_layer_bounds[-1] + (sigma_layer_bounds[-1] - sigma_layer_bounds[-2]),
                                     sigma.max().values, n_layers_lower)

    sigma_layer_bounds = np.concatenate((sigma_layer_bounds_upper, sigma_layer_bounds, sigma_layer_bounds_lower))

    np.savetxt(os.path.join(model_dir, output_dir, 'sigma_layer_bounds.txt'),
                sigma_layer_bounds, delimiter=',')

    return sigma_layer_bounds


###########################
# Main function starts here
###########################

def bin_fields(model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                output_dir='sigma_space_output/',
                sigma_name='sigma1',
                Tref=0,
                plotting=False):
    """
    Convert z-coordinate fields to isopycnal coordinate.
    """

    output_dir = os.path.join(output_dir, sigma_name)


    # make sure the output_dir exists
    os.makedirs(os.path.join(model_dir, output_dir), exist_ok=True)

    # load desired iterations from iters.txt file
    iters = np.fromfile(os.path.join(model_dir, 'iters.txt'), dtype=int, sep=',')
    iters = iters.tolist()
    # to use only the last 10 years
    iters = iters[-720:]

    print('Loading model data')

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
    # uPs to convert it to preformed salinity.
    # gsw.gibbs.constants.uPS = 1.0047154285714286
    # And then just use model variables as inputs to
    # the gsw calculations.

    # calc potential density
    # Should possibly be sigma1, since the dynamics of interest happen at about that depth.

    print('Calculating density')

    if sigma_name == 'sigma0':
        sigma = gsw.density.sigma0(ds_state['SALT'].where(mask_TP)*1.0047154285714286,
                            ds_state['THETA'].where(mask_TP))
        sigma_bar = sigma.mean(dim='time').compute()
        sigma_layer_bounds = load_layer_bounds(sigma, sigma_bar, output_dir)

    elif sigma_name == 'sigma1':
        sigma = gsw.density.sigma1(ds_state['SALT'].where(mask_TP)*1.0047154285714286,
                            ds_state['THETA'].where(mask_TP))
        sigma_bar = sigma.mean(dim='time').compute()
        sigma_layer_bounds = load_layer_bounds(sigma, sigma_bar, output_dir)

    elif sigma_name == 'sigma2':
        sigma = gsw.density.sigma2(ds_state['SALT'].where(mask_TP)*1.0047154285714286,
                            ds_state['THETA'].where(mask_TP))
        sigma_bar = sigma.mean(dim='time').compute()
        sigma_layer_bounds = load_layer_bounds(sigma, sigma_bar, output_dir)

        # # original layer bounds
        # # sigma layer bounds
        # sigma_layer_bounds = np.append(np.linspace(31.5,36.8, 53, endpoint=False), np.linspace(36.8, 36.95,3, endpoint=False))
        # sigma_layer_bounds = np.append(sigma_layer_bounds, np.linspace(36.95, 37.1, 6, endpoint=False))
        # sigma_layer_bounds = np.append(sigma_layer_bounds, np.linspace(37.1, 38, 10))

    elif sigma_name == 'sigma3':
        sigma = gsw.density.sigma3(ds_state['SALT'].where(mask_TP)*1.0047154285714286,
                            ds_state['THETA'].where(mask_TP))
        sigma_bar = sigma.mean(dim='time').compute()
        sigma_layer_bounds = load_layer_bounds(sigma, sigma_bar, output_dir)

    elif sigma_name == 'sigma4':
        sigma = gsw.density.sigma4(ds_state['SALT'].where(mask_TP)*1.0047154285714286,
                            ds_state['THETA'].where(mask_TP))
        sigma_bar = sigma.mean(dim='time').compute()
        sigma_layer_bounds = load_layer_bounds(sigma, sigma_bar, output_dir)

    else:
        raise ValueError('sigma option not set correctly')
        return

    sigma.name = 'sigma'


    #move to required location for interpolation
    sigma_yp1 = grid.interp(sigma, 'Y', boundary='extend')
    sigma_xp1 = grid.interp(sigma, 'X', boundary='extend')
    # sigma_yp1_zp1 = grid.interp(sigma_yp1, 'Z', boundary='extend')#, to='outer')

    sigma_yp1_bar = sigma_yp1.mean(dim='time').compute()
    sigma_xp1_bar = sigma_xp1.mean(dim='time').compute()
    # sigma_yp1_zp1_bar = grid.interp(sigma_yp1_bar, 'Z', boundary='extend', to='outer')


    # Code for calculating layer bounds
    # BUT - want to use the same layer bounds for control and perturbation runs,
    # so need to be a bit smarter than just recalculating it.


    sigma_layer_midpoints = (sigma_layer_bounds[1:] + sigma_layer_bounds[:-1])/2


    # define time mean vvel at constant depth
    vbar = ds_state['VVEL'].sel(time=slice(time_range[0], time_range[1])).mean(dim=['time']).compute()

    meridional_volume_flux = ds_state['VVEL']*ds_state['drF']*ds_state['dxG']*ds_state['hFacS']
    zonal_volume_flux = ds_state['UVEL']*ds_state['drF']*ds_state['dyG']*ds_state['hFacW']

    # heat_content = grid.interp(ds_state['THETA']*ds_state['drF']*ds_state['hFacC'], 'Y', boundary='extend')
    # vertical_coordinate = xr.ones_like(ds_state['THETA'])*ds_state['drF']

    print('Calculating layerwise volume flux')
    layerwise_merid_vol_flux = histogram(sigma_yp1,
                              bins=[sigma_layer_bounds],
                              dim = ['Z'],
                              weights=meridional_volume_flux).rename(
                                    {'sigma_bin':'sigma'}).rename(
                                    'layerwise_merid_vol_flux')

    layerwise_merid_vol_flux.load()

    print('Saving layerwise_merid_vol_flux to NetCDF file')
    layerwise_merid_vol_flux.to_netcdf(os.path.join(model_dir, output_dir,
                                                    'layerwise_merid_vol_flux.nc'),
                encoding={'layerwise_merid_vol_flux': {'shuffle': True,
                                                        'zlib': True,
                                                        'complevel': 5}})

    # zonal volume fluxes
    layerwise_zonal_vol_flux = histogram(sigma_xp1,
                              bins=[sigma_layer_bounds],
                              dim = ['Z'],
                              weights=zonal_volume_flux).rename(
                                    {'sigma_bin':'sigma'}).rename(
                                    'layerwise_zonal_vol_flux')

    layerwise_zonal_vol_flux.load()

    print('Saving layerwise_zonal_vol_flux to NetCDF file')
    layerwise_zonal_vol_flux.to_netcdf(os.path.join(model_dir, output_dir,
                                                    'layerwise_zonal_vol_flux.nc'),
                encoding={'layerwise_zonal_vol_flux': {'shuffle': True,
                                                        'zlib': True,
                                                        'complevel': 5}})


    psi = layerwise_merid_vol_flux.mean(dim='time').compute().rename('psi')

    # This is the traditional way of computing the Eulerian-mean overturning
    # Can also be done by computing the layerwise velocity and then averaging.
    psi_bar = histogram(sigma_yp1_bar,
                          bins=[sigma_layer_bounds],
                          dim = ['Z'],
                          weights=vbar*ds_state['drF']*ds_state['dxG']*
                                    ds_state['hFacS']).rename({'sigma_bin':'sigma'}).rename('psi_bar')

    psi_bar.load()

    # Eddy overturning
    psi_star = (psi - psi_bar).rename('psi_star')

    print('Saving psi, psi_bar and psi_star  fields to NetCDF files')
    # save fields to NetCDF files
    psi.to_netcdf(os.path.join(model_dir, output_dir, 'psi.nc'),
                encoding={'psi': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    psi_bar.to_netcdf(os.path.join(model_dir, output_dir, 'psi_bar.nc'),
                encoding={'psi_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    psi_star.to_netcdf(os.path.join(model_dir, output_dir, 'psi_star.nc'),
                encoding={'psi_star': {'shuffle': True, 'zlib': True, 'complevel': 5}})


    # Eulerian-mean layer thickness.
    hbar = histogram(sigma_bar,
                          bins=[sigma_layer_bounds],
                          dim = ['Z'],
                          weights=xr.ones_like(sigma_bar)*ds_state['drF']).rename({'sigma_bin':'sigma'}).rename('hbar')

    hbar.load()
    print('Saving hbar to NetCDF file')
    # save fields to NetCDF files
    hbar.to_netcdf(os.path.join(model_dir, output_dir, 'hbar.nc'),
                encoding={'hbar': {'shuffle': True, 'zlib': True, 'complevel': 5}})


    X_layered_plotting = np.tile(ds_state['YG'], (len(sigma_layer_bounds)-1,1)).T
    Y_layered_plotting = -grid.interp(hbar.mean(dim=['XC']).cumsum(dim='sigma').compute(), 'Y', boundary='extend')

    if plotting is True:
        # isopycnal space
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        levels=np.linspace(-3,3,30,)*1e6

        cumsum_from_bottom(psi.sum(dim='XC')).plot.contourf(ax=ax[0], y='sigma', yincrease=False, robust=True, levels=levels)
        ax[0].set_title('$\Psi$')

        cumsum_from_bottom(psi_bar.sum(dim='XC')).plot.contourf(ax=ax[1], y='sigma', yincrease=False, robust=True, levels=levels)
        ax[1].set_title('$\overline{\Psi}$')

        cumsum_from_bottom(psi_star.sum(dim='XC')).plot.contourf(ax=ax[2], y='sigma', yincrease=False, robust=True, levels=levels)
        ax[2].set_title('$\Psi^{*}$')

        fig.savefig(os.path.join(model_dir, output_dir, 'psi_eddy_mean_sigma_space.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

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
        fig.savefig(os.path.join(model_dir, output_dir, 'psi_eddy_mean_depth_space.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

    # Zonal-mean and zonal perturbation streamfunctions
    psi_zm = (xr.ones_like(psi_bar)*histogram(sigma_yp1_bar.mean(dim='XC'),
                                          bins=[sigma_layer_bounds],
                                          dim = ['Z'],
                                          weights=(vbar*ds_state['drF']*ds_state['dxG']
                                                   *ds_state['hFacS']).mean(dim='XC')).rename(
                                                {'sigma_bin':'sigma'})).rename('psi_zm')

    psi_zm.load()

    psi_zp = (psi_bar - psi_zm).rename('psi_zp')

    print('Saving psi_zm and psi_zp fields to NetCDF files')
    # save fields to NetCDF files
    psi_zm.to_netcdf(os.path.join(model_dir, output_dir, 'psi_zm.nc'),
                encoding={'psi_zm': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    psi_zp.to_netcdf(os.path.join(model_dir, output_dir, 'psi_zp.nc'),
                encoding={'psi_zp': {'shuffle': True, 'zlib': True, 'complevel': 5}})


    if plotting is True:
        # isopycnal space
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        cumsum_from_bottom(psi_bar.sum(dim='XC')).plot.contourf(ax=ax[0], y='sigma', yincrease=False, robust=True, levels=levels)
        ax[0].set_title('$\overline{\Psi}$')

        cumsum_from_bottom(psi_zm.sum(dim='XC')).plot.contourf(ax=ax[1], y='sigma', yincrease=False, robust=True, levels=levels)
        ax[1].set_title('$\Psi_{zm}$')


        cumsum_from_bottom(psi_zp.sum(dim='XC')).plot.contourf(ax=ax[2], y='sigma', yincrease=False, robust=True, levels=levels)
        ax[2].set_title('$\Psi_{zp}$ - zonal perturbation')

        fig.savefig(os.path.join(model_dir, output_dir, 'psi_bar_decompostion_sigma_space.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

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
        fig.savefig(os.path.join(model_dir, output_dir, 'psi_bar_decompostion_depth_space.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

    # volume transport
    vh_prime = (layerwise_merid_vol_flux - psi)

    print('Calculating layerwise temperature')
    # Temperature
    # Need to use the linear transformation from xgcm for temperature, since it is not an
    # extensive property. Using xhistogram results in some weird values.
    # linear interpolation from xgcm
    layerwise_temperature = grid.transform(ds_state['THETA'] - Tref,
                                              'Z',
                                              sigma_layer_midpoints,
                                              method='linear',
                                              target_data=sigma,
                                           mask_edges=False).rename('layerwise_temperature')
    layerwise_temperature.load()

    # can get this either by averaging T in z space and converting, or
    # by converting to layers and then averaging. Averaging in layers
    # means that isopycnal heave doesn't smear the temperature signal.
    layerwise_Tbar = layerwise_temperature.mean(dim='time').rename('layerwise_Tbar')

                        # grid.transform(grid.interp(Tbar - Tref, 'Y', boundary='extend'),
                        #                       'Z',
                        #                       sigma_layer_midpoints,
                        #                       method='linear',
                        #                       target_data=sigma_bar,
                        #                    mask_edges=False).compute()

    Tbar = ds_state['THETA'].mean(dim='time').load().rename('Tbar')

    print('Saving temperature fields to NetCDF files')
    # save fields to NetCDF files
    layerwise_temperature.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_temperature.nc'),
                encoding={'layerwise_temperature': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    layerwise_Tbar.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_Tbar.nc'),
                encoding={'layerwise_Tbar': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    Tbar.to_netcdf(os.path.join(model_dir, output_dir, 'Tbar.nc'),
                encoding={'Tbar': {'shuffle': True, 'zlib': True, 'complevel': 5}})



    if plotting is True:
        layerwise_Tbar.where(hbar>0.01).mean(dim='XC').plot(y='sigma', yincrease=False)
        plt.savefig(os.path.join(model_dir, output_dir, 'layerwise_Tbar_sigma_space.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

        plt.pcolormesh(X_layered_plotting, Y_layered_plotting,
                    grid.interp(layerwise_Tbar.where(hbar>0.01).mean(dim='XC'), 'Y', boundary='extend'),
                    cmap=cmocean.cm.thermal, vmin=-2, vmax=15)
        plt.colorbar()
        plt.savefig(os.path.join(model_dir, output_dir, 'layerwise_Tbar_depth_space.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

    print('Calculating layerwise heat transport')
    # heat transport
    vhc_reconstructed = layerwise_merid_vol_flux*grid.interp(layerwise_temperature, 'Y', boundary='extend')
    vhc_reconstructed.load()

    vhc_reconstructed_bar = vhc_reconstructed.mean(dim='time').compute().rename('vhc_reconstructed_bar')

    print('Saving vhc_reconstructed_bar to NetCDF file')
    # save fields to NetCDF files
    vhc_reconstructed_bar.to_netcdf(os.path.join(model_dir, output_dir, 'vhc_reconstructed_bar.nc'),
                encoding={'vhc_reconstructed_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    Tprime = layerwise_temperature - layerwise_Tbar

    # eddy diffusive transport
    vh_prime_Tprime_bar = (vh_prime*grid.interp(Tprime, 'Y', boundary='extend')).mean(dim='time').compute().rename('vh_prime_Tprime_bar')

    print('Saving eddy diffusion term to NetCDF file')
    # save fields to NetCDF files
    vh_prime_Tprime_bar.to_netcdf(os.path.join(model_dir, output_dir, 'vh_prime_Tprime_bar.nc'),
                encoding={'vh_prime_Tprime_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Advective heat transport
    psi_Tbar = psi*grid.interp(layerwise_Tbar, 'Y', boundary='extend')

    # is made up of Eulerian-mean and eddy

    # Eulerian-mean overturning heat transport
    psibar_Tbar = psi_bar*grid.interp(layerwise_Tbar, 'Y', boundary='extend')
        # is made up of
        # zonal-mean
    psizm_Tbar = psi_zm*grid.interp(layerwise_Tbar, 'Y', boundary='extend')
        # standing meander
    psizp_Tbar = psi_zp*grid.interp(layerwise_Tbar, 'Y', boundary='extend')

    # eddy overturning heat ransport
    psistar_Tbar = psi_star*grid.interp(layerwise_Tbar, 'Y', boundary='extend')

    if plotting is True:
        # plot advective diffusive breakdown
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        im = ax[0].pcolormesh(X_layered_plotting, Y_layered_plotting, vhc_reconstructed_bar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[0].set_title('Total heat transport')
        plt.colorbar(im, ax=ax[0])

        im2 = ax[1].pcolormesh(X_layered_plotting, Y_layered_plotting, psi_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[1].set_title('Advective heat transport')
        plt.colorbar(im2, ax=ax[1])

        (-hbar.mean(dim='XC').cumsum(dim='sigma').sel(sigma=36.6, method='nearest')).plot(ax=ax[2], color='k')
        im3 = ax[2].pcolormesh(X_layered_plotting, Y_layered_plotting, vh_prime_Tprime_bar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[2].set_title('Diffusive heat transport')
        plt.colorbar(im3, ax=ax[2])

        fig.savefig(os.path.join(model_dir, output_dir, 'zonal_mean_heat_flux_depth_space.png'),
            dpi=200, bbox_inches='tight')
        plt.close('all')

        # plot advection decomposition
        fig, ax = plt.subplots(1,3, figsize=(15,4))

        im = ax[0].pcolormesh(X_layered_plotting, Y_layered_plotting, psi_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[0].set_title('Advective heat trans.')
        plt.colorbar(im2, ax=ax[0])

        im = ax[1].pcolormesh(X_layered_plotting, Y_layered_plotting, psibar_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[1].set_title('Eulerian-mean adv. heat trans.')
        plt.colorbar(im, ax=ax[1])

        (-hbar.mean(dim='XC').cumsum(dim='sigma').sel(sigma=36.6, method='nearest')).plot(ax=ax[2], color='k')
        im3 = ax[2].pcolormesh(X_layered_plotting, Y_layered_plotting, psistar_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[2].set_title('Eddy-induced adv. heat trans.')
        plt.colorbar(im3, ax=ax[2])

        fig.savefig(os.path.join(model_dir, output_dir,
                        'zonal_mean_heat_flux_advection_decomposition_depth_space.png'),
                    dpi=200, bbox_inches='tight')
        plt.close('all')

        # plot Eulerian-mean, zonal-mean, zonal-perturbation decompostion
        fig, ax = plt.subplots(1,3, figsize=(15,4))

        im = ax[0].pcolormesh(X_layered_plotting, Y_layered_plotting, psibar_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[0].set_title('Eulerian-mean adv. heat trans.')
        plt.colorbar(im2, ax=ax[0])

        im = ax[1].pcolormesh(X_layered_plotting, Y_layered_plotting, psizm_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[1].set_title('Zonal-mean adv. heat transport')
        plt.colorbar(im, ax=ax[1])


        (-hbar.mean(dim='XC').cumsum(dim='sigma').sel(sigma=36.6, method='nearest')).plot(ax=ax[2], color='k')
        im3 = ax[2].pcolormesh(X_layered_plotting, Y_layered_plotting, psizp_Tbar.sum(dim='XC'), cmap='RdBu_r', vmin=-3e6, vmax=3e6)
        ax[2].set_title('Zonal Perturbation adv. heat trans.')
        plt.colorbar(im3, ax=ax[2])

        fig.savefig(os.path.join(model_dir, output_dir,
                        'zonal_mean_and_perturbation_heat_flux_advection_decomposition_depth_space.png'),
                    dpi=200, bbox_inches='tight')
        plt.close('all')

    # using model output heat advection to test
    # (only works exactly if the reference temperature is set to 0°C)
    layerwise_heat_advection = histogram(sigma_yp1,
                          bins=[sigma_layer_bounds],
                          dim = ['Z'],
                          weights=ds_heat['ADVy_TH']).rename(
                                {'sigma_bin':'sigma'}).rename(
                                'layerwise_heat_advection')


    vhc_bar = layerwise_heat_advection.sel(time=slice(time_range[0],
                                            time_range[1])).mean(
                                            dim='time').compute().rename(
                                            'vhc_bar')

    print('Saving model heat advection fields to NetCDF files')
    # save fields to NetCDF files
    # from model output heat advection - Use to check decomposition
    layerwise_heat_advection.to_netcdf(os.path.join(model_dir, output_dir, 'layerwise_heat_advection.nc'),
                encoding={'layerwise_heat_advection': {'shuffle': True, 'zlib': True, 'complevel': 5}})
    vhc_bar.to_netcdf(os.path.join(model_dir, output_dir, 'vhc_bar.nc'),
                encoding={'vhc_bar': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    return


if __name__ == '__main__':

    # Start a dask cluster with multiple cores
    client = climtas.nci.GadiClient()
    print(client)

    # This script calculates binned quantities for a single year.
        #### get year argument that was passed to python script ####

    import sys

    sigma_name = str(sys.argv[1])

    bin_fields(model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                output_dir='sigma_space_output/',
                Tref=-2,
                sigma_name=sigma_name,
                plotting=True)

