import pandas as pd

from .fleets import *
from .read_models import read_models
from .apply_models import apply_models

def _e_per_type(explicitbin_models, fleet_comp, volume): 
    
    """Applies each RouteE model coefficient to the network dataframe.
    
    Note:
        Private function.
    
    Args:
        explicitbin_models (list): 
            List of column names containing the RouteE
            model coefficients.
        
        fleet_comp (dict): 
            Dictionary describing the proportion each
            model represents in the overall fleet.
                            
        volume (float): 
            Estimated segment-level volume
        
    Returns:
        (Series): pandas Series object for the total energy
                    of each vehicle type
    
    """
    
    vol_bins = [fleet_comp[k]*volume for k,v in fleet_comp.items()]
    e_bins = [x*y for x,y in zip(explicitbin_models, vol_bins)]

    return pd.Series({
                'cv_g_tot':e_bins[0],
                'cv_d_tot':e_bins[1],
                'phev_cd_tot':e_bins[2],
                'phev_cs_tot':e_bins[3],
                'hev_tot':e_bins[4],
                'bev_tot':e_bins[5],
                })

def network_prediction(links, fleet_comp=None, year='2017', **models): 
    """ Main wrapper function to append RouteE estimates to a network. 
    This converts the input road network and several parameters for
    the submodules to estimate the road network energy consumption.
    
    Args:
        links (pandas.DataFrame): 
            Input link-level road network DataFrame.

        
        fleet_comp (str or dict, optional): 
            Parameter describing the area of interest to use
            for the input fleet composition. Accepts 'columbus'
            or 'dc' currently. Default is nationally.
            Alternatively, accepts a dictionary which explicitly
            indicates the proportion of the fleet that is represented
            by the 6 RouteE model types. Example:
                fleet_comp = {'cv_gas': .9551,
                              'cv_diesel': .0319,
                              'phev_cd': .0003,
                              'phev_cs': .0003,
                              'hev': .0118,
                              'bev': .0005}
        
        year (str):
            Year used to estimate fleet composition based on values 
            derived from the 2014-2017 IHS vehicle registration data.
        
                                    
        **models: 
            Dictionary with of model names and path to the .pkl files of
            these objects. Optional, default models used if not provided.
                                 
    """
    
    if int(year) < 2014:
        year = '2014'
    elif int(year) > 2017:
        year = '2017'
    
    if fleet_comp is None:
        fleet_dict = NATIONAL_FLEET[year]
        
    elif fleet_comp == 'columbus':
        fleet_dict = COLUMBUS_FLEET[year]
        
    elif fleet_comp == 'dc':
        fleet_dict = DC_FLEET[year]

    else:
        fleet_dict = fleet_comp
        
        keys = {'cv_gas','cv_diesel',
                'phev_cd','phev_cs',
                'hev','bev'}

        error_msg = '''
        Incorrect input fleet composition. This 
        parameter requires a dictionary with the 
        following format and order: 
        {
        'cv_gas': float,
        'cv_diesel': float,
        'phev_cd': float,
        'phev_cs': float,
        'hev': float,
        'bev': float,
        }
        '''
        assert(set(fleet_dict.keys()) == keys), error_msg
        
        
        
    model_dict = read_models(**models)
    links = apply_models(links, model_dict)
    
    output_cols = ['cv_g_tot', 'cv_d_tot',
                   'phev_cd_tot', 'phev_cs_tot',
                   'hev_tot', 'bev_tot']
    
    links[output_cols] = links.apply(lambda row: _e_per_type(
            [row.cv_g_e,
             row.cv_d_e,
             row.phev_cd_e,
             row.phev_cs_e,
             row.hev_e,
             row.bev_e],
            fleet_dict, row.volume), axis=1)

    links['total_e'] = links.apply(lambda row: sum([
            row.cv_g_tot, row.cv_d_tot, 
            row.phev_cd_tot, row.phev_cs_tot,
            row.hev_tot, row.bev_tot
            ]), axis=1)
    
    links['avg_e_per_veh'] = links.apply(lambda row:
                                         row.total_e/row.volume,
                                         axis=1)

    return links