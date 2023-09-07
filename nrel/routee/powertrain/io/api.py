from pathlib import Path
from typing import Union

from nrel.routee.powertrain.core.model import Model


def read_model(infile: Union[str, Path]) -> Model:
    """Function to read model from file.

    Args:
        infile (str):
            Path and filename for saved file to read.

    """
    path = Path(infile)

    if path.suffix != ".onnx":
        raise ValueError("File must be an .onnx file.")

    return Model.from_file(path)
