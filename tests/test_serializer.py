import numpy as np
import scipy.sparse as sps

from eals.eals import ElementwiseAlternatingLeastSquares
from eals.serializer import deserialize_eals_joblib, serialize_eals_joblib, serialize_eals_json, deserialize_eals_json


def assert_model_equality(model1, model2):
    assert model1.factors == model2.factors
    assert model1.w0 == model2.w0
    assert model1.alpha == model2.alpha
    assert model1.reg == model2.reg
    assert model1.init_mean == model2.init_mean
    assert model1.init_stdev == model2.init_stdev
    assert model1.dtype == model2.dtype
    assert model1.max_iter == model2.max_iter
    assert model1.max_iter_online == model2.max_iter_online
    assert model1.random_state == model2.random_state
    assert model1.show_loss == model2.show_loss
    assert np.allclose(model1.U, model2.U)
    assert np.allclose(model1.V, model2.V)
    assert (model1.user_items_lil.data == model2.user_items_lil.data).all()
    assert (model1.user_items_lil.rows == model2.user_items_lil.rows).all()


def test_serialize_and_deserialize(tmp_path):
    # setup: 3 users x 2 items
    user_items = sps.csr_matrix([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    model = ElementwiseAlternatingLeastSquares(
        factors=64,
        w0=10,
        alpha=0.75,
        reg=0.01,
        init_mean=0,
        init_stdev=0.01,
        dtype=np.float32,
        max_iter=1,
        max_iter_online=1,
        random_state=None,
        show_loss=False,
    )
    model.fit(user_items)
    
    # test .json
    file_json = (tmp_path / "model.json").as_posix()
    serialize_eals_json(file_json, model, compress=False)
    model_actual = deserialize_eals_json(file_json)
    assert_model_equality(model, model_actual)

    # test .json.gz
    file_gzip = (tmp_path / "model.json.gz").as_posix()
    serialize_eals_json(file_gzip, model, compress=True)
    model_actual = deserialize_eals_json(file_gzip)
    assert_model_equality(model, model_actual)

    # test .joblib without compression
    file_joblib = (tmp_path / "model0.joblib").as_posix()
    serialize_eals_joblib(file_joblib, model, compress=0)
    model_actual = deserialize_eals_joblib(file_joblib)
    assert_model_equality(model, model_actual)

    # test .joblib with compression
    file_joblib = (tmp_path / "model9.joblib").as_posix()
    serialize_eals_joblib(file_joblib, model, compress=9)
    model_actual = deserialize_eals_joblib(file_joblib)
    assert_model_equality(model, model_actual)
