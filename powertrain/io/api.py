from pathlib import Path

from powertrain.core.model import Model


def read_model(infile: str):
    """Function to read model from file.

    Args:
        infile (str):
            Path and filename for saved file to read.

    """
    path = Path(infile)

    if path.suffix == ".json":
        return Model.from_json(Path(path))
    elif path.suffix == ".pickle":
        return Model.from_pickle(path)
    else:
        raise ImportError(
            f"file type of {path.suffix} not supported by routee-powertrain"
        )
