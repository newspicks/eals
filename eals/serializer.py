import gzip
from pathlib import Path
from typing import Union

import numpy as np
import scipy.sparse as sps
import orjson

import eals


def serialize_eals_json(file: Union[Path, str], model: "eals.eals.ElementwiseAlternatingLeastSquares", compress: bool = True):
    filepath = Path(file)
    model_dict = _serialize_eals_json_lil(model)
    if compress:
        if not str(filepath).endswith(".gz"):
            filepath = Path(str(filepath) + ".gz")
        f = gzip.open(filepath, "wb")
    else:
        f = open(filepath, "wb")
    f.write(orjson.dumps(model_dict, option=orjson.OPT_SERIALIZE_NUMPY))


def deserialize_eals_json(file: Union[Path, str]) -> "eals.eals.ElementwiseAlternatingLeastSquares":
    filepath = Path(file)
    with open(filepath, "rb") as testfile:
        is_gzip = str(filepath).endswith(".gz") and testfile.read(2) == b"\x1f\x8b"  # gzip magic number
    if is_gzip:
        f = gzip.open(filepath, "rb")
    else:
        f = open(filepath, "rb")
    model_dict = orjson.loads(f.read())
    model = _deserialize_eals_json_lil(model_dict)
    return model


def _serialize_eals_json_lil(model: "eals.eals.ElementwiseAlternatingLeastSquares") -> dict:
    model_dict = dict()
    # model initializer arguments
    model_dict["factors"] = model.factors
    model_dict["w0"] = model.w0
    model_dict["alpha"] = model.alpha
    model_dict["reg"] = model.reg
    model_dict["init_mean"] = model.init_mean
    model_dict["init_stdev"] = model.init_stdev
    # model_dict["dtype"] = model.dtype
    model_dict["max_iter"] = model.max_iter
    model_dict["max_iter_online"] = model.max_iter_online
    model_dict["random_state"] = model.random_state
    model_dict["show_loss"] = model.show_loss
    # model parameters
    model_dict["U"] = model.U
    model_dict["V"] = model.V
    # data
    # TODO: coo形式を経由せずにlil形式のままserialize/deserialize
    user_items = model.user_items_lil.tocoo()
    model_dict["user_items"] = {"data": [], "row": [], "col": []}
    model_dict["user_items"]["data"] = user_items.data
    model_dict["user_items"]["row"] = user_items.row
    model_dict["user_items"]["col"] = user_items.col
    model_dict["user_items"]["shape"] = user_items.shape
    # TODO: WとWiの扱いをどうするか
    #   W: 現状では1固定なのでまあ無視でOK
    #   Wi: deserialize時にinit_data()で計算されるが、モデル保存時の値とは一致しない可能性あり（init_data()とupdate_model()が整合的でないので）
    return model_dict


def _deserialize_eals_json_lil(model_dict: dict) -> "eals.eals.ElementwiseAlternatingLeastSquares":
    # model initializer arguments
    factors = model_dict.get("factors") or 64
    w0 = model_dict.get("w0") or 10
    alpha = model_dict.get("alpha") or 0.75
    reg = model_dict.get("reg") or 0.01
    init_mean = model_dict.get("init_mean") or 0
    init_stdev = model_dict.get("init_stdev") or 0.01
    # dtype = (model_dict.get("dtype") or np.float32,)
    max_iter = model_dict.get("max_iter") or 500
    max_iter_online = model_dict.get("max_iter_online") or 1
    random_state = model_dict.get("random_state") or None
    show_loss = model_dict.get("show_loss") or False
    # model parameters
    U = np.array(model_dict.get("U")) if model_dict.get("U") else None
    V = np.array(model_dict.get("V")) if model_dict.get("V") else None
    # data
    # TODO: csr経由せずにlil形式でserialize/deserialize
    user_items_dict = model_dict.get("user_items")
    if user_items_dict:
        user_items_data = user_items_dict["data"]
        user_items_rows = user_items_dict["row"]
        user_items_cols = user_items_dict["col"]
        user_items_shape = user_items_dict["shape"]
        user_items = sps.csr_matrix(
            (user_items_data, (user_items_rows, user_items_cols)), shape=user_items_shape
        )
    else:
        user_items = sps.csr_matrix(([], ([], [])), shape=(0, 0))
    # create a model object
    model = eals.eals.ElementwiseAlternatingLeastSquares(
        factors,
        w0,
        alpha,
        reg,
        init_mean,
        init_stdev,
        # TODO: Enable serialization of dtype
        np.float32,
        max_iter,
        max_iter_online,
        random_state,
        show_loss,
    )
    model.init_data(user_items, U, V)
    # conversion to lil_matrix
    # TODO: private method使いたくない
    # TODO: init_data()でcsr形式で読み込んでからlilに変換するのは非効率
    model._convert_data_for_online_training()
    # TODO: WとWiの扱いをどうするか
    return model
