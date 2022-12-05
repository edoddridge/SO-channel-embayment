"""
Cross-contour volume and heat transport calculations.
"""


import xarray as xr
import xmitgcm
import xgcm
import numpy as np
import ipywidgets
import dask
import cmocean
import pandas as pd
import os
import gsw
import matplotlib.pyplot as plt

import climtas.nci

# Ignore warnings
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)


def calc_masks(ssh_mean, ssh_contour_level):
    """
    Calculate masks for zonal and meridional transport.
    """
    contour_mask = (0*ssh_mean + 1).where(ssh_mean>ssh_contour_level, 0)

    u_transport_mask = xr.zeros_like(ds_2d['oceTAUX'].isel(time=0)) + (contour_mask - contour_mask.roll(XC=1)).values
    u_transport_mask = u_transport_mask.rename('u_transport_mask').drop_vars(['time', 'iter'])

    v_transport_mask = xr.zeros_like(ds_2d['oceTAUY'].isel(time=0)) + (contour_mask - contour_mask.roll(YC=1)).values
    v_transport_mask = v_transport_mask.rename('v_transport_mask').drop_vars(['time', 'iter'])

    masks = xr.Dataset({'contour_mask':contour_mask,
                        'u_transport_mask':u_transport_mask,
                        'v_transport_mask':v_transport_mask},
        attrs={'contour_level':ssh_contour_level})

    return masks



def cross_contour_transport(ssh_contour_levels, model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                output_dir='sigma_space_output/',
                sigma_name='sigma1',
                plotting=False):
    """
    Calculate cross-contour transport. Currently uses time-mean ssh contours.
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

    ds_state = xmitgcm.open_mdsdataset(data_dir=model_dir+'/Diags', grid_dir=model_dir,
                                   prefix=['state'], delta_t=500, calendar='360_day',
                                   ref_date='2000-1-1 0:0:0', geometry='Cartesian',
                               iters=iters)

    uh = xr.open_dataarray(os.path.join(model_dir, output_dir, 'layerwise_zonal_vol_flux.nc'), chunks='auto')
    vh = xr.open_dataarray(os.path.join(model_dir, output_dir, 'layerwise_merid_vol_flux.nc'), chunks='auto')

    h = xr.open_dataarray(os.path.join(model_dir, output_dir, 'h.nc'), chunks='auto')
    hbar = xr.open_dataarray(os.path.join(model_dir, output_dir, 'hbar.nc'), chunks='auto')

    temperature = xr.open_dataarray(os.path.join(model_dir, output_dir, 'layerwise_temperature.nc'), chunks='auto')
    Tbar = xr.open_dataarray(os.path.join(model_dir, output_dir, 'layerwise_Tbar.nc'), chunks='auto')

    ds_state['h'] = h
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

    # Calculate volume transports
    # divide transport by thickness to get grid-width weighted velocities
    # mask out regions with very thin layers
    u = (uh/grid.interp(h.where(h>0.01), 'X', boundary='extend')).fillna(0)#.where(grid.interp(h, 'X', boundary='extend')>0.01)
    v = (vh/grid.interp(h.where(h>0.01), 'Y', boundary='extend')).fillna(0)#.where(grid.interp(h, 'Y', boundary='extend')>0.01)

    ubar = u.mean(dim='time').load()
    vbar = v.mean(dim='time').load()

    # Total transport (residual transport in TEM lingo)
    uh_bar = uh.mean(dim='time')
    vh_bar = vh.mean(dim='time')

    # Eulerian-mean transport
    ubar_hbar = ubar*grid.interp(hbar, 'X', boundary='extend')
    vbar_hbar = vbar*grid.interp(hbar, 'Y', boundary='extend')

    # Eddy transport
    up_hp_bar = uh_bar - ubar_hbar
    vp_hp_bar = vh_bar - vbar_hbar


    # Calculate heat transport
    # Total heat transport
    uhT_bar = (u*grid.interp(h*temperature, 'X', boundary='extend')).mean(dim='time').rename('uhT_bar').load()
    vhT_bar = (v*grid.interp(h*temperature, 'Y', boundary='extend')).mean(dim='time').rename('vhT_bar').load()



    # Advective heat transport
    uh_bar_Tbar = uh_bar*grid.interp(Tbar, 'X', boundary='extend').rename('uh_bar_Tbar')
    vh_bar_Tbar = vh_bar*grid.interp(Tbar, 'Y', boundary='extend').rename('vh_bar_Tbar')

    # is made up of Eulerian-mean and eddy

    # Eulerian-mean overturning heat transport
    ubar_hbar_Tbar = ubar*grid.interp(hbar*Tbar, 'X',
        boundary='extend').rename('ubar_hbar_Tbar')
    vbar_hbar_Tbar = vbar*grid.interp(hbar*Tbar, 'Y',
        boundary='extend').rename('vbar_hbar_Tbar')
    #     # is made up of
    #     # zonal-mean
    # psizm_Tbar = psi_zm*layerwise_Tbar
    #     # standing meander
    # psizp_Tbar = psi_zp*layerwise_Tbar


    # eddy overturning heat transport
    up_hp_bar_Tbar = up_hp_bar*grid.interp(Tbar, 'X',
        boundary='extend').rename('up_hp_bar_Tbar')
    vp_hp_bar_Tbar = vp_hp_bar*grid.interp(Tbar, 'Y',
        boundary='extend').rename('up_hp_bar_Tbar')


    # Diffusive heat flux
    uhp_Tp_bar = ((uh_bar - uh)*grid.interp(temperature - Tbar, 'X',
        boundary='extend')).mean(dim='time').compute().rename('uhp_Tp_bar')
    vhp_Tp_bar = ((vh_bar - vh)*grid.interp(temperature - Tbar, 'Y',
        boundary='extend')).mean(dim='time').compute().rename('vhp_Tp_bar')


    # calculate masks
    ssh_mean = ds_2d['ETAN'].mean(dim='time').load()

    mask_list = []

    for contour_level in ssh_contour_levels:
        mask_list.append(calc_masks(ssh_mean, contour_level))

    masks = xr.concat(mask_list, pd.Index(ssh_contour_levels, name='ssh_contour'))

    # Cross Contour Transports

    # Total volume
    total_cross_cont_vol = (grid.interp((uh_bar*masks['u_transport_mask']), 'X') +
                        grid.interp((vh_bar*masks['v_transport_mask']), 'Y')).rename(
                        'total_cross_cont_vol').assign_coords(
                        ssh_contour=("ssh_contour", ssh_contour_levels))

    total_cross_cont_vol.load()
    total_cross_cont_vol.to_netcdf(os.path.join(model_dir, output_dir, 'total_cross_cont_vol.nc'),
                encoding={'total_cross_cont_vol': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Eulerian-mean volume
    EM_cross_cont_vol = (grid.interp((ubar_hbar*masks['u_transport_mask']), 'X') +
                     grid.interp((vbar_hbar*masks['v_transport_mask']), 'Y')).rename(
                     'EM_cross_cont_vol').assign_coords(
                     ssh_contour=("ssh_contour", ssh_contour_levels))

    EM_cross_cont_vol.load()
    EM_cross_cont_vol.to_netcdf(os.path.join(model_dir, output_dir, 'EM_cross_cont_vol.nc'),
                    encoding={'EM_cross_cont_vol': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Eddy volume
    eddy_cross_cont_vol = (grid.interp((up_hp_bar*masks['u_transport_mask']), 'X') +
                     grid.interp((vp_hp_bar*masks['v_transport_mask']), 'Y')).rename(
                     'eddy_cross_cont_vol').assign_coords(
                     ssh_contour=("ssh_contour", ssh_contour_levels))

    eddy_cross_cont_vol.load()
    eddy_cross_cont_vol.to_netcdf(os.path.join(model_dir, output_dir, 'eddy_cross_cont_vol.nc'),
                    encoding={'eddy_cross_cont_vol': {'shuffle': True, 'zlib': True, 'complevel': 5}})


    # Heat transports

    # total heat
    total_cross_cont_heat = (grid.interp((uhT_bar*masks['u_transport_mask']), 'X') +
                     grid.interp((vhT_bar*masks['v_transport_mask']), 'Y')).rename(
                     'total_cross_cont_heat').assign_coords(
                     ssh_contour=("ssh_contour", ssh_contour_levels))

    total_cross_cont_heat.load()
    total_cross_cont_heat.to_netcdf(os.path.join(model_dir, output_dir, 'total_cross_cont_heat.nc'),
                    encoding={'total_cross_cont_heat': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Advective heat
    advec_cross_cont_heat = (grid.interp((uh_bar_Tbar*masks['u_transport_mask']), 'X') +
                     grid.interp((vh_bar_Tbar*masks['v_transport_mask']), 'Y')).rename(
                     'advec_cross_cont_heat').assign_coords(
                     ssh_contour=("ssh_contour", ssh_contour_levels))

    advec_cross_cont_heat.load()
    advec_cross_cont_heat.to_netcdf(os.path.join(model_dir, output_dir, 'advec_cross_cont_heat.nc'),
                    encoding={'advec_cross_cont_heat': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Advective mean heat
    advec_mean_cross_cont_heat = (grid.interp((ubar_hbar_Tbar*masks['u_transport_mask']), 'X') +
                         grid.interp((vbar_hbar_Tbar*masks['v_transport_mask']), 'Y')).rename(
                         'advec_mean_cross_cont_heat').assign_coords(
                         ssh_contour=("ssh_contour", ssh_contour_levels))

    advec_mean_cross_cont_heat.load()
    advec_mean_cross_cont_heat.to_netcdf(os.path.join(model_dir, output_dir, 'advec_mean_cross_cont_heat.nc'),
                    encoding={'advec_mean_cross_cont_heat': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Advective eddy heat
    advec_eddy_cross_cont_heat = (grid.interp((up_hp_bar_Tbar*masks['u_transport_mask']), 'X') +
                     grid.interp((vp_hp_bar_Tbar*masks['v_transport_mask']), 'Y')).rename(
                     'advec_eddy_cross_cont_heat').assign_coords(
                     ssh_contour=("ssh_contour", ssh_contour_levels))

    advec_eddy_cross_cont_heat.load()
    advec_eddy_cross_cont_heat.to_netcdf(os.path.join(model_dir, output_dir, 'advec_eddy_cross_cont_heat.nc'),
                    encoding={'advec_eddy_cross_cont_heat': {'shuffle': True, 'zlib': True, 'complevel': 5}})

    # Diffusive heat
    diff_cross_cont_heat = (grid.interp((uhp_Tp_bar*masks['u_transport_mask']), 'X') +
                     grid.interp((vhp_Tp_bar*masks['v_transport_mask']), 'Y')).rename(
                     'diff_cross_cont_heat').assign_coords(
                     ssh_contour=("ssh_contour", ssh_contour_levels))


    diff_cross_cont_heat.load()
    diff_cross_cont_heat.to_netcdf(os.path.join(model_dir, output_dir, 'diff_cross_cont_heat.nc'),
                    encoding={'diff_cross_cont_heat': {'shuffle': True, 'zlib': True, 'complevel': 5}})




     return


if __name__ == '__main__':

    # Start a dask cluster with multiple cores
    client = climtas.nci.GadiClient()
    print(client)

    import sys

    sigma_name = str(sys.argv[1])

    ssh_contour_levels = np.linspace(-0.6, 0.6, 13)

    cross_contour_transport(ssh_contour_levels,
                model_dir='/g/data/jk72/ed7737/SO-channel_embayment/simulations/run/',
                output_dir='sigma_space_output/',
                sigma_name=sigma_name,
                plotting=True)
