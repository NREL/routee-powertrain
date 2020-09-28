def apply_models(links, model_dict):
    
    """Applies the six core RouteE models to the input network.
    
    Args:
        links (pandas.DataFrame): 
            Input road network with necessary RouteE input columns
        
        model_dict (dict): 
            Dictionary of the name and model object. 
            Generated in read_models.py
    
    Returns:
        links (pandas.DataFrame): 
            road network with appended RouteE model coefficients.
    
    """
    
    gge_conversion = 33.4 #convert kWh to gge
    
    base_model_dict = {
        'cv_g_e': ('cv_gas', 'gallons'),
        'cv_d_e': ('cv_diesel', 'gallons'),
        'phev_cd_e': ('phev_cd', 'kWh'),
        'phev_cs_e': ('phev_cs', 'gallons'),
        'hev_e': ('hev', 'gallons'),
        'bev_e': ('bev', 'kWh')
    }
    
    for model, attributes in base_model_dict.items():
        
        if attributes[1] == 'gallons':
            links[model] = model_dict[attributes[0]].predict(links)

        else:
            links[model] = model_dict[attributes[0]].predict(links)
            links[model] = links[model] / gge_conversion

    return links