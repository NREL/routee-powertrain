from pathlib import Path

from powertrain.core.model import VehicleModel


def read_model(infile: str) -> VehicleModel:
    """Function to read model from file.

    Args:
        infile (str):
            Path and filename for saved file to read.

    """
    path = Path(infile)

    if path.suffix != ".onnx":
        raise ValueError("File must be an .onnx file.")

    return VehicleModel.from_file(path)
