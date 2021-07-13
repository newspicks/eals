import gzip
import json

import numpy as np
import scipy.sparse as sps

from .eals import ElementwiseAlternatingLeastSquares


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def serialize_json(file: str, model: ElementwiseAlternatingLeastSquares, compress: bool = True):
    model_dict = _serialize_json_lil(model)
    if compress:
        with gzip.open(file, "wb") as g:
            g.write(json.dumps(model_dict, cls=NumpyArrayEncoder).encode("utf-8"))
    else:
        with open(file, "w") as f:
            json.dump(model_dict, f, cls=NumpyArrayEncoder)


def deserialize_json(file: str) -> ElementwiseAlternatingLeastSquares:
    is_gzip = file.endswith(".gz")  # TODO: more robust checking
    if is_gzip:
        with gzip.open(file, "rb") as g:
            model_dict = json.loads(g.read())
    else:
        with open(file, "r") as f:
            model_dict = json.load(f)
    model = _deserialize_json_lil(model_dict)
    return model


def _serialize_json_lil(model: ElementwiseAlternatingLeastSquares) -> dict:
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
    model_dict["U"] = model.U.tolist()
    model_dict["V"] = model.V.tolist()
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


def _deserialize_json_lil(model_dict: dict) -> ElementwiseAlternatingLeastSquares:
    # model initializer arguments
    factors = (model_dict.get("factors") or 64)
    w0 = (model_dict.get("w0") or 10)
    alpha = (model_dict.get("alpha") or 0.75)
    reg = (model_dict.get("reg") or 0.01)
    init_mean = (model_dict.get("init_mean") or 0)
    init_stdev = (model_dict.get("init_stdev") or 0.01)
    # dtype = (model_dict.get("dtype") or np.float32,)
    max_iter = (model_dict.get("max_iter") or 500)
    max_iter_online = (model_dict.get("max_iter_online") or 1)
    random_state = (model_dict.get("random_state") or None)
    show_loss = (model_dict.get("show_loss") or False)
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
        # TODO: 行列のshapeも保存しておくべき？
        user_items = sps.csr_matrix(
            (user_items_data, (user_items_rows, user_items_cols)),
            shape=user_items_shape
        )
    else:
        user_items = sps.csr_matrix(([], ([], [])), shape=(0, 0))
    # create a model object
    model = ElementwiseAlternatingLeastSquares(
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
