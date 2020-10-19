from pathlib import Path

from powertrain.core.model import Model


def read_model(infile: str):
    """Function to read model from file.

    Args:
        infile (str):
            Path and filename for saved file to read. 
            
    """
    return Model.from_json(Path(infile))
