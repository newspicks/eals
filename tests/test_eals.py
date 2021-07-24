from unittest import mock

import numpy as np
import scipy.sparse as sps

from eals.eals import ElementwiseAlternatingLeastSquares, load_model


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
    assert model1.show_loss == model2.show_loss
    assert np.allclose(model1.U, model2.U)
    assert np.allclose(model1.V, model2.V)
    assert (model1.user_items_lil.data == model2.user_items_lil.data).all()
    assert (model1.user_items_lil.rows == model2.user_items_lil.rows).all()


def test_init_data():
    # Test initialization for some instance variables
    user_items = sps.csc_matrix([[0, 1], [1, 0]])
    alpha = 0.5
    w0 = 10
    model = ElementwiseAlternatingLeastSquares(alpha=alpha, w0=w0)
    model.init_data(user_items)
    assert np.allclose(model.Wi, [w0 / 2, w0 / 2])
    assert np.allclose(model.W.toarray(), [[0, 1], [1, 0]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_update_user(mock_init_V, mock_init_U):
    # Test for a trivial case:
    # - 1x1 rating matrix and 1x1 latent vectors with all initial values being 1
    # - This implies U[0,0] = 1 / (1 + regularization) after the 1st update
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    regularization = 0.01
    model = ElementwiseAlternatingLeastSquares(regularization=regularization, factors=U0.shape[1])
    model.init_data(user_items)
    old_user_vec = model.update_user(0)
    assert np.allclose(old_user_vec, [[1.0]])
    assert np.allclose(model.U, [[1 / (1 + regularization)]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_update_item(mock_init_V, mock_init_U):
    # Test for a trivial case:
    # - 1x1 rating matrix and 1x1 latent vectors with all initial values being 1
    # - This implies V[0,0] = 1 / (1 + regularization) after the 1st update
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    regularization = 0.02
    model = ElementwiseAlternatingLeastSquares(regularization=regularization, factors=U0.shape[1])
    model.init_data(user_items)
    old_item_vec = model.update_item(0)
    assert np.allclose(old_item_vec, [[1.0]])
    assert np.allclose(model.V, [[1 / (1 + regularization)]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
def test_update_SU_with_factor1d(mock_init_U):
    user_items = sps.csc_matrix([[1.0]])
    U0 = np.array([[3.0]])
    mock_init_U.return_value = U0
    model = ElementwiseAlternatingLeastSquares(factors=U0.shape[1])
    model.init_data(user_items)  # SU = 3*3 = 9
    model.update_SU(u=0, old_user_vec=np.array([[2.0]]))
    # SU = 9 - 2*2 + 3*3 = 14
    assert np.allclose(model.SU, [[14.0]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
def test_update_SU_with_factor2d(mock_init_U):
    user_items = sps.csc_matrix([[1.0]])
    U0 = np.array([[1.0, 2.0]])
    mock_init_U.return_value = U0
    model = ElementwiseAlternatingLeastSquares(factors=U0.shape[1])
    model.init_data(user_items)  # SU = [[1],[2]] @ [[1,2]] = [[1,2],[2,4]]
    model.update_SU(u=0, old_user_vec=np.array([[3.0, 4.0]]))
    # SU = [[1,2],[2,4]] - [[3],[4]] @ [[3,4]] + [[1],[2]] @ [[1,2]] = [[-7, -8], [-8, -8]]
    assert np.allclose(model.SU, [[-7.0, -8.0], [-8.0, -8.0]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_update_SV_with_factor1d(mock_init_V):
    user_items = sps.csc_matrix([[1.0]])
    w0 = 5
    alpha = 1  # user_items, w0, and alpha give model.Wi = 5
    V0 = np.array([[3.0]])
    mock_init_V.return_value = V0
    model = ElementwiseAlternatingLeastSquares(w0=w0, alpha=alpha, factors=V0.shape[1])
    model.init_data(user_items)  # SV = 3*3 * 5 = 45
    model.update_SV(i=0, old_item_vec=np.array([[2.0]]))
    # SV = 45 - (2*2 - 3*3) * 5 = 70
    assert np.allclose(model.SV, [[70.0]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_update_SV_with_factor2d(mock_init_V):
    user_items = sps.csc_matrix([[1.0]])
    w0 = 5
    alpha = 1  # user_items, w0, and alpha give model.Wi = 5
    V0 = np.array([[3.0, 4.0]])
    mock_init_V.return_value = V0
    model = ElementwiseAlternatingLeastSquares(w0=w0, alpha=alpha, factors=V0.shape[1])
    model.init_data(user_items)  # SV = [[3],[4]] @ [[3,4]] * 5 = [[45,60],[60,80]]
    model.update_SV(i=0, old_item_vec=np.array([[2.0, 3.0]]))
    # SV = [[45,60],[60,80]] - ([[2],[3]] @ [[2,3]] - [[3],[4]] @ [[3,4]]) * 5 = [[70,90],[90,115]]
    assert np.allclose(model.SV, [[70, 90], [90, 115]])


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_update_user_and_SU_all(mock_init_V, mock_init_U):
    # Almost the same test as test_update_user()
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    regularization = 0.01
    model = ElementwiseAlternatingLeastSquares(regularization=regularization, factors=U0.shape[1])
    model.init_data(user_items)
    model.update_user_and_SU_all()
    assert np.allclose(model.U, [[1 / (1 + regularization)]])
    assert np.allclose(model.SU, model.U.T @ model.U)


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_update_item_and_SV_all(mock_init_V, mock_init_U):
    # Almost the same test as test_update_item()
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    regularization = 0.02
    model = ElementwiseAlternatingLeastSquares(regularization=regularization, factors=U0.shape[1])
    model.init_data(user_items)
    model.update_item_and_SV_all()
    assert np.allclose(model.V, [[1 / (1 + regularization)]])
    assert np.allclose(model.SV, (model.V.T * model.Wi) @ model.V)


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_fit_no_iteration(mock_init_V, mock_init_U):
    user_items = sps.csc_matrix([[1.0, 0.0], [1.0, 1.0]])
    U0 = V0 = np.array([[0.5, 0.1, 0.2], [0.7, 0.8, 0.9]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    # Nothing happens if num_iter=0
    model = ElementwiseAlternatingLeastSquares(num_iter=0, factors=U0.shape[1])
    model.fit(user_items)
    assert np.allclose(model.U, U0)
    assert np.allclose(model.V, V0)


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_fit_one_iteration(mock_init_V, mock_init_U):
    user_items = sps.csc_matrix([[1.0, 0.0], [1.0, 1.0]])
    U0 = V0 = np.array([[0.5, 0.1, 0.2], [0.7, 0.8, 0.9]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    # (fit with num_iter=1) == init_data + update_user_and_SU_all + update_item_and_SV_all
    model_actual = ElementwiseAlternatingLeastSquares(num_iter=1, factors=U0.shape[1])
    model_actual.fit(user_items)
    model_expected = ElementwiseAlternatingLeastSquares(factors=U0.shape[1])
    model_expected.init_data(user_items)
    model_expected.update_user_and_SU_all()
    model_expected.update_item_and_SV_all()
    assert np.allclose(model_actual.U, model_expected.U)
    assert np.allclose(model_actual.V, model_expected.V)


def test_update_model():
    # TODO: Implement it.
    pass


def test_update_model_for_existing_user_and_item():
    user_items = sps.csc_matrix([[1, 0, 0, 2], [1, 1, 0, 0], [0, 0, 1, 2]])
    model = ElementwiseAlternatingLeastSquares(num_iter=1)
    model.fit(user_items)
    model.update_model(2, 3)
    assert model.user_factors.shape[0] == 3
    assert model.item_factors.shape[0] == 4


def test_update_model_for_new_user():
    user_items = sps.csc_matrix([[1, 0, 0, 2], [1, 1, 0, 0], [0, 0, 1, 2]])
    model = ElementwiseAlternatingLeastSquares(num_iter=1)
    model.fit(user_items)
    model.update_model(3, 3)
    assert model.user_factors.shape[0] == 103
    assert model.item_factors.shape[0] == 4


def test_update_model_for_new_item():
    user_items = sps.csc_matrix([[1, 0, 0, 2], [1, 1, 0, 0], [0, 0, 1, 2]])
    model = ElementwiseAlternatingLeastSquares(num_iter=1)
    model.fit(user_items)
    model.update_model(2, 4)
    assert model.user_factors.shape[0] == 3
    assert model.item_factors.shape[0] == 104


def test_update_model_for_new_user_and_item():
    user_items = sps.csc_matrix([[1, 0, 0, 2], [1, 1, 0, 0], [0, 0, 1, 2]])
    model = ElementwiseAlternatingLeastSquares(num_iter=1)
    model.fit(user_items)
    model.update_model(3, 4)
    assert model.user_factors.shape[0] == 103
    assert model.item_factors.shape[0] == 104


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_calc_loss_csr(mock_init_V, mock_init_U):
    # 2 users, 1 item
    user_items = sps.csc_matrix([[1.0], [0.0]])
    U0 = np.array([[0.9], [0.5]])
    V0 = np.array([[1.0]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    regularization = 1
    w0 = 1
    alpha = 1
    Wi = 1
    model = ElementwiseAlternatingLeastSquares(regularization=regularization, w0=w0, alpha=alpha, factors=U0.shape[1])
    model.init_data(user_items)

    l_regularization = regularization * ((U0 ** 2).sum() + (V0 ** 2).sum())  # regularization term
    l_user0 = (user_items[0, 0] - U0[0] @ V0[0]) ** 2  # usual loss term
    l_user1 = Wi * (U0[1] @ V0[0]) ** 2  # missing data term
    loss_expected = l_regularization + l_user0 + l_user1
    assert np.allclose(model.calc_loss(), loss_expected)


@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_U")
@mock.patch.object(ElementwiseAlternatingLeastSquares, "_init_V")
def test_calc_loss_lil(mock_init_V, mock_init_U):
    # 2 users, 1 item
    user_items = sps.csc_matrix([[1.0], [0.0]])
    U0 = np.array([[0.9], [0.5]])
    V0 = np.array([[1.0]])
    mock_init_U.return_value = U0
    mock_init_V.return_value = V0
    regularization = 1
    w0 = 1
    alpha = 1
    Wi = 1
    model = ElementwiseAlternatingLeastSquares(regularization=regularization, w0=w0, alpha=alpha, factors=U0.shape[1])
    model.init_data(user_items)
    model._convert_data_for_online_training()

    l_regularization = regularization * ((U0 ** 2).sum() + (V0 ** 2).sum())  # regularization term
    l_user0 = (user_items[0, 0] - U0[0] @ V0[0]) ** 2  # usual loss term
    l_user1 = Wi * (U0[1] @ V0[0]) ** 2  # missing data term
    loss_expected = l_regularization + l_user0 + l_user1
    assert np.allclose(model.calc_loss(), loss_expected)


def test_save_and_load_model(tmp_path):
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
        show_loss=False,
    )
    model.fit(user_items)
    # test .joblib without compression
    file_joblib = (tmp_path / "model_nocompress.joblib").as_posix()
    model.save(file_joblib, compress=False)
    model_actual = load_model(file_joblib)
    assert_model_equality(model, model_actual)

    # test .joblib with compression
    file_joblib = (tmp_path / "model_compress.joblib").as_posix()
    model.save(file_joblib, compress=True)
    model_actual = load_model(file_joblib)
    assert_model_equality(model, model_actual)
