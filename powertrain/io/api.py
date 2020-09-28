'''Basic interfact function for reading RouteE models.

This is the base API for interacting with the RouteE package.

Example:
    import routee as rte
    
    model_path = 'path\to\model.pickle'
    model = rte.read_model(model_path)

'''

import pickle

from powertrain.core.model import Model

def read_model(infile):
    """Function to read model from file.

    Args:
        infile (str):
            Path and filename for saved file to read. 
            
    """
    
    in_dict = pickle.load(open(infile, 'rb'))
	
    model = Model(
                    in_dict['metadata']['veh_desc'],
		    in_dict['option'],
                    estimator = in_dict['estimator']
                    )
    model.metadata = in_dict['metadata']
    model.errors = in_dict['errors']
    model.option = in_dict['option']

    return model
