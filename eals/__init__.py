import importlib.metadata

from .eals import ElementwiseAlternatingLeastSquares, load_model

__version__ = importlib.metadata.version("eals")
__all__ = ["ElementwiseAlternatingLeastSquares", "load_model", "__version__"]
