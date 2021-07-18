from pathlib import Path
from typing import Union

import joblib

import eals


def serialize_eals_joblib(
    file: Union[Path, str],
    model: "eals.eals.ElementwiseAlternatingLeastSquares",
    compress: Union[bool, int] = True,
):
    joblib.dump(model, Path(file), compress=compress)


def deserialize_eals_joblib(
    file: Union[Path, str]
) -> "eals.eals.ElementwiseAlternatingLeastSquares":
    model: eals.eals.ElementwiseAlternatingLeastSquares = joblib.load(Path(file))
    return model
