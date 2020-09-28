import os
import powertrain

def read_models(**kwargs):
    """Reads in the necessary RouteE model objects.
    
    Note:
        Loads default models packaged with RouteE 
        if no kwargs provided.
    Args:    
        Kwargs: pairs of RouteE model types and paths
        to the corresponding .pickle file.
    
    """
    
    #Temporary placeholder for default models
    cv_d_model = os.path.abspath('..//trained_models//2016_AUDI_A3_4cyl_2WD.pickle')
    cv_g_model = os.path.abspath('..//trained_models//2016_BMW_328d_4cyl_2WD.pickle')
    phev_cd_model = os.path.abspath('..//trained_models//2017_Prius_Prime_Charge_Depleting.pickle')
    phev_cs_model = os.path.abspath('..//trained_models//2016_BMW_i3_REx_PHEV_Charge_Sustaining.pickle')
    hev_model = os.path.abspath('..//trained_models//2015_Honda_Accord_HEV.pickle')
    bev_model = os.path.abspath('..//trained_models//2016_CHEVROLET_Spark_EV.pickle')
    
    model_dict = {}
    
    if 'cv_gas' in kwargs:
        model_dict['cv_gas']=routee.read_model(kwargs['cv_gas'])
    else:
        model_dict['cv_gas']=routee.read_model(cv_g_model)
        
    if 'cv_diesel' in kwargs:
        model_dict['cv_diesel']=routee.read_model(kwargs['cv_diesel'])
    else:
        model_dict['cv_diesel'] = routee.read_model(cv_d_model)
    
    if 'phev_cd' in kwargs:
        model_dict['phev_cd']=routee.read_model(kwargs['phev_cd'])                                          
    else:
        model_dict['phev_cd']=routee.read_model(phev_cd_model)
    
    if 'phev_cs' in kwargs:
        model_dict['phev_cs']=routee.read_model(kwargs['phev_cs'])                                          
    else:
        model_dict['phev_cs']=routee.read_model(phev_cs_model)
        
    if 'hev' in kwargs:
        model_dict['hev']=routee.read_model(kwargs['hev'])
    else:
        model_dict['hev']=routee.read_model(hev_model)
    
    if 'bev' in kwargs:
        model_dict['bev']=routee.read_model(kwargs['bev'])
    else:
        model_dict['bev']=routee.read_model(bev_model)

    return model_dict