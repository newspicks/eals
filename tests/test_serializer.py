import filecmp

import numpy as np
import scipy.sparse as sps

from eals.eals import ElementwiseAlternatingLeastSquares
from eals.serializer import deserialize_eals_joblib, serialize_eals_joblib


def assert_model_equality(model1, model2):
    assert model1.factors == model2.factors
    assert model1.w0 == model2.w0
    assert model1.alpha == model2.alpha
    assert model1.regularization == model2.regularization
    assert model1.init_mean == model2.init_mean
    assert model1.init_stdev == model2.init_stdev
    assert model1.num_iter == model2.num_iter
    assert model1.num_iter_online == model2.num_iter_online
    assert model1.random_state == model2.random_state
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
        regularization=0.01,
        init_mean=0,
        init_stdev=0.01,
        num_iter=1,
        num_iter_online=1,
        random_state=None,
    )
    model.fit(user_items)

    # Test .joblib without compression.
    # Also check that compress=0 and compress=False give the same result.
    file_joblib = (tmp_path / "model_compress-0.joblib").as_posix()
    serialize_eals_joblib(file_joblib, model, compress=0)
    model_actual = deserialize_eals_joblib(file_joblib)
    assert_model_equality(model, model_actual)

    file_joblib = (tmp_path / "model_compress-false.joblib").as_posix()
    serialize_eals_joblib(file_joblib, model, compress=False)
    model_actual = deserialize_eals_joblib(file_joblib)
    assert_model_equality(model, model_actual)

    assert filecmp.cmp(tmp_path / "model_compress-0.joblib", tmp_path / "model_compress-false.joblib", shallow=False)

    # Test .joblib with compression.
    # compress=3 and compress=True give the same result.
    file_joblib = (tmp_path / "model_compress-3.joblib").as_posix()
    serialize_eals_joblib(file_joblib, model, compress=3)
    model_actual = deserialize_eals_joblib(file_joblib)
    assert_model_equality(model, model_actual)

    file_joblib = (tmp_path / "model_compress-true.joblib").as_posix()
    serialize_eals_joblib(file_joblib, model, compress=True)
    model_actual = deserialize_eals_joblib(file_joblib)
    assert_model_equality(model, model_actual)

    assert filecmp.cmp(tmp_path / "model_compress-3.joblib", tmp_path / "model_compress-true.joblib", shallow=False)
