import numpy as np
import scipy.sparse as sps

from eals.eals import ElementwiseAlternatingLeastSquares


def test_init_data():
    # 初期化ロジックがややこしめのインスタンス変数のみテストする
    user_items = sps.csc_matrix([[0, 1], [1, 0]])
    alpha = 0.5
    w0 = 10
    model = ElementwiseAlternatingLeastSquares(alpha=alpha, w0=w0)
    model.init_data(user_items)
    assert np.allclose(model.Wi, [w0 / 2, w0 / 2])
    assert np.allclose(model.W.toarray(), [[0, 1], [1, 0]])


def test_update_user():
    # Test for a trivial case:
    # - 1x1 rating matrix and 1x1 latent vectors with all initial values being 1
    # - This implies U[0,0] = 1 / (1 + reg) after the 1st update
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    reg = 0.01
    model = ElementwiseAlternatingLeastSquares(reg=reg, factors=U0.shape[1])
    model.init_data(user_items, U0, V0)
    old_user_vec = model.update_user(0)
    assert np.allclose(old_user_vec, [[1.0]])
    assert np.allclose(model.U, [[1 / (1 + reg)]])


def test_update_item():
    # Test for a trivial case:
    # - 1x1 rating matrix and 1x1 latent vectors with all initial values being 1
    # - This implies V[0,0] = 1 / (1 + reg) after the 1st update
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    reg = 0.02
    model = ElementwiseAlternatingLeastSquares(reg=reg, factors=U0.shape[1])
    model.init_data(user_items, U0, V0)
    old_item_vec = model.update_item(0)
    assert np.allclose(old_item_vec, [[1.0]])
    assert np.allclose(model.V, [[1 / (1 + reg)]])


def test_update_SU_with_factor1d():
    user_items = sps.csc_matrix([[1.0]])
    U0 = np.array([[3.0]])
    model = ElementwiseAlternatingLeastSquares(factors=U0.shape[1])
    model.init_data(user_items, U0)  # SU = 3*3 = 9
    model.update_SU(u=0, old_user_vec=np.array([[2.0]]))
    # SU = 9 - 2*2 + 3*3 = 14
    assert np.allclose(model.SU, [[14.0]])


def test_update_SU_with_factor2d():
    user_items = sps.csc_matrix([[1.0]])
    U0 = np.array([[1.0, 2.0]])
    model = ElementwiseAlternatingLeastSquares(factors=U0.shape[1])
    model.init_data(user_items, U0)  # SU = [[1],[2]] @ [[1,2]] = [[1,2],[2,4]]
    model.update_SU(u=0, old_user_vec=np.array([[3.0, 4.0]]))
    # SU = [[1,2],[2,4]] - [[3],[4]] @ [[3,4]] + [[1],[2]] @ [[1,2]] = [[-7, -8], [-8, -8]]
    assert np.allclose(model.SU, [[-7.0, -8.0], [-8.0, -8.0]])


def test_update_SV_with_factor1d():
    user_items = sps.csc_matrix([[1.0]])
    w0 = 5
    alpha = 1  # user_items, w0, and alpha give model.Wi = 5
    V0 = np.array([[3.0]])
    model = ElementwiseAlternatingLeastSquares(w0=w0, alpha=alpha, factors=V0.shape[1])
    model.init_data(user_items, V0=V0)  # SV = 3*3 * 5 = 45
    model.update_SV(i=0, old_item_vec=np.array([[2.0]]))
    # SV = 45 - (2*2 - 3*3) * 5 = 70
    assert np.allclose(model.SV, [[70.0]])


def test_update_SV_with_factor2d():
    user_items = sps.csc_matrix([[1.0]])
    w0 = 5
    alpha = 1  # user_items, w0, and alpha give model.Wi = 5
    V0 = np.array([[3.0, 4.0]])
    model = ElementwiseAlternatingLeastSquares(w0=w0, alpha=alpha, factors=V0.shape[1])
    model.init_data(user_items, V0=V0)  # SV = [[3],[4]] @ [[3,4]] * 5 = [[45,60],[60,80]]
    model.update_SV(i=0, old_item_vec=np.array([[2.0, 3.0]]))
    # SV = [[45,60],[60,80]] - ([[2],[3]] @ [[2,3]] - [[3],[4]] @ [[3,4]]) * 5 = [[70,90],[90,115]]
    assert np.allclose(model.SV, [[70, 90], [90, 115]])


def test_update_user_and_SU_all():
    # Almost the same test as test_update_user()
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    reg = 0.01
    model = ElementwiseAlternatingLeastSquares(reg=reg, factors=U0.shape[1])
    model.init_data(user_items, U0, V0)
    model.update_user_and_SU_all()
    assert np.allclose(model.U, [[1 / (1 + reg)]])
    assert np.allclose(model.SU, model.U.T @ model.U)


def test_update_item_and_SV_all():
    # Almost the same test as test_update_item()
    user_items = sps.csc_matrix([[1.0]])
    U0 = V0 = np.array([[1.0]])
    reg = 0.02
    model = ElementwiseAlternatingLeastSquares(reg=reg, factors=U0.shape[1])
    model.init_data(user_items, U0, V0)
    model.update_item_and_SV_all()
    assert np.allclose(model.V, [[1 / (1 + reg)]])
    assert np.allclose(model.SV, (model.V.T * model.Wi) @ model.V)


def test_fit_no_iteration():
    user_items = sps.csc_matrix([[1.0, 0.0], [1.0, 1.0]])
    U0 = V0 = np.array([[0.5, 0.1, 0.2], [0.7, 0.8, 0.9]])
    # Nothing happens if max_iter=0
    model = ElementwiseAlternatingLeastSquares(max_iter=0, factors=U0.shape[1])
    model.fit(user_items, U0, V0)
    assert np.allclose(model.U, U0)
    assert np.allclose(model.V, V0)


def test_fit_one_iteration():
    user_items = sps.csc_matrix([[1.0, 0.0], [1.0, 1.0]])
    U0 = V0 = np.array([[0.5, 0.1, 0.2], [0.7, 0.8, 0.9]])
    # (fit with max_iter=1) == init_data + update_user_and_SU_all + update_item_and_SV_all
    model_actual = ElementwiseAlternatingLeastSquares(max_iter=1, factors=U0.shape[1])
    model_actual.fit(user_items, U0, V0)
    model_expected = ElementwiseAlternatingLeastSquares(factors=U0.shape[1])
    model_expected.init_data(user_items, U0, V0)
    model_expected.update_user_and_SU_all()
    model_expected.update_item_and_SV_all()
    assert np.allclose(model_actual.U, model_expected.U)
    assert np.allclose(model_actual.V, model_expected.V)


def test_update_model():
    # TODO: Implement it.
    pass


def test_calc_loss_csr():
    # 2 users, 1 item
    user_items = sps.csc_matrix([[1.0], [0.0]])
    U0 = np.array([[0.9], [0.5]])
    V0 = np.array([[1.0]])
    reg = 1
    w0 = 1
    alpha = 1
    Wi = 1
    model = ElementwiseAlternatingLeastSquares(reg=reg, w0=w0, alpha=alpha, factors=U0.shape[1])
    model.init_data(user_items, U0, V0)

    l_reg = reg * ((U0 ** 2).sum() + (V0 ** 2).sum())  # regularization term
    l_user0 = (user_items[0, 0] - U0[0] @ V0[0]) ** 2  # usual loss term
    l_user1 = Wi * (U0[1] @ V0[0]) ** 2  # missing data term
    loss_expected = l_reg + l_user0 + l_user1
    assert np.allclose(model.calc_loss(), loss_expected)


def test_calc_loss_lil():
    # 2 users, 1 item
    user_items = sps.csc_matrix([[1.0], [0.0]])
    U0 = np.array([[0.9], [0.5]])
    V0 = np.array([[1.0]])
    reg = 1
    w0 = 1
    alpha = 1
    Wi = 1
    model = ElementwiseAlternatingLeastSquares(reg=reg, w0=w0, alpha=alpha, factors=U0.shape[1])
    model.init_data(user_items, U0, V0)
    model._convert_data_for_online_training()

    l_reg = reg * ((U0 ** 2).sum() + (V0 ** 2).sum())  # regularization term
    l_user0 = (user_items[0, 0] - U0[0] @ V0[0]) ** 2  # usual loss term
    l_user1 = Wi * (U0[1] @ V0[0]) ** 2  # missing data term
    loss_expected = l_reg + l_user0 + l_user1
    assert np.allclose(model.calc_loss(), loss_expected)
