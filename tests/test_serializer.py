from typing import Union
import pytest
import numpy as np
import scipy.sparse as sps

from eals.eals import ElementwiseAlternatingLeastSquares
from eals.serializer import serialize_json, deserialize_json


def test_serialize_and_deserialize(tmp_path):
    # 3 users x 2 items
    user_items = sps.csr_matrix([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    model = ElementwiseAlternatingLeastSquares(
        factors=64,  # dimension of latent vectors
        w0=10,  # overall weight of missing data
        alpha=0.75,  # control parameter for significance level of popular items
        reg=0.01,  # regularization parameter lambda
        init_mean=0,  # mean of initial lantent vectors
        init_stdev=0.01,  # stdev of initial lantent vectors
        dtype=np.float32,
        max_iter=1,
        max_iter_online=1,
        random_state=None,
        show_loss=False,
    )
    model.fit(user_items)

    # test with compressiion
    file_gzip = (tmp_path / "model.json.gz").as_posix()
    serialize_json(file_gzip, model, compress=True)
    model_actual = deserialize_json(file_gzip)
    assert model.factors == model_actual.factors
    assert model.w0 == model_actual.w0
    assert model.alpha == model_actual.alpha
    assert model.reg == model_actual.reg
    assert model.init_mean == model_actual.init_mean
    assert model.init_stdev == model_actual.init_stdev
    assert model.dtype == model_actual.dtype
    assert model.max_iter == model_actual.max_iter
    assert model.max_iter_online == model_actual.max_iter_online
    assert model.random_state == model_actual.random_state
    assert model.show_loss == model_actual.show_loss
    assert np.allclose(model.U, model_actual.U)
    assert np.allclose(model.V, model_actual.V)
    assert (model.user_items_lil.data == model_actual.user_items_lil.data).all()
    assert (model.user_items_lil.rows == model_actual.user_items_lil.rows).all()

    # test without compressiion
    file_json = (tmp_path / "model.json").as_posix()
    serialize_json(file_json, model, compress=False)
    model_actual = deserialize_json(file_json)
    assert model.factors == model_actual.factors
    assert model.w0 == model_actual.w0
    assert model.alpha == model_actual.alpha
    assert model.reg == model_actual.reg
    assert model.init_mean == model_actual.init_mean
    assert model.init_stdev == model_actual.init_stdev
    assert model.dtype == model_actual.dtype
    assert model.max_iter == model_actual.max_iter
    assert model.max_iter_online == model_actual.max_iter_online
    assert model.random_state == model_actual.random_state
    assert model.show_loss == model_actual.show_loss
    assert np.allclose(model.U, model_actual.U)
    assert np.allclose(model.V, model_actual.V)
    assert (model.user_items_lil.data == model_actual.user_items_lil.data).all()
    assert (model.user_items_lil.rows == model_actual.user_items_lil.rows).all()
