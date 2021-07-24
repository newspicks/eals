import os
from distutils.util import strtobool
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.sparse as sps

_USE_NUMBA = bool(strtobool(os.environ.get("USE_NUMBA", "True")))
_USE_NUMBA_PARALLEL = bool(strtobool(os.environ.get("USE_NUMBA_PARALLEL", "True")))

if _USE_NUMBA:
    from numba import njit, prange
else:
    prange = range

    def njit(*args, **kwargs):
        def nojit(f):
            return f

        return nojit


from .serializer import deserialize_eals_joblib, serialize_eals_joblib
from .util import Timer


class ElementwiseAlternatingLeastSquares:
    """Element-wise Alternating Least Squares (eALS)

    Parameters
    ----------
    factors: int
        Dimension of latent vectors
    w0: float
        Overall weight of missing data
    alpha: float
        Control parameter for significance level of popular items
    regularization: float
        Regularization parameter lambda
    init_mean: float
        Mean of initial latent vectors
    init_stdev: float
        Standard deviation of initial latent vectors
    num_iter: int
        The number of iterations for batch training
    num_iter_online: int
        The number of iterations for online training
    random_state: int
        Numpy random seed

    Attributes
    ----------
    user_factors: numpy.ndarray
        Latent vectors for users
    item_factors: numpy.ndarray
        Latent vectors for items

    Notes
    ----------
    Original eALS paper and Java inplementation
    - https://arxiv.org/abs/1708.05024
    - https://github.com/hexiangnan/sigir16-eals
    """

    def __init__(
        self,
        factors: int = 64,
        w0: float = 10,
        alpha: float = 0.75,
        regularization: float = 0.01,
        init_mean: float = 0,
        init_stdev: float = 0.01,
        num_iter: int = 500,
        num_iter_online: int = 1,
        random_state: Optional[int] = None,
    ):
        self.factors = factors
        self.w0 = w0
        self.alpha = alpha
        self.regularization = regularization
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.num_iter = num_iter
        self.num_iter_online = num_iter_online
        self.random_state = random_state

        self._training_mode = "batch"  # "batch" (use csr/csc matrix) or "online" (use lil matrix)

    def fit(self, user_items: sps.spmatrix, show_loss: bool = False):
        """Fit the model to the given rating data in batch mode

        Parameters
        ----------
        user_items: scipy.sparse.spmatrix
            Rating matrix for user-item pairs
        show_loss: bool
            Whether to compute and print the loss after each iteration
        """
        self.init_data(user_items)

        timer = Timer()
        for iter in range(self.num_iter):
            self.update_user_and_SU_all()
            if show_loss:
                self.print_loss(iter, "update_user", timer.elapsed())
            self.update_item_and_SV_all()
            if show_loss:
                self.print_loss(iter, "update_item", timer.elapsed())

        self._convert_data_for_online_training()

    def init_data(self, user_items: sps.spmatrix):
        """Initialize parameters and hyperparameters before batch training
        """
        # coerce user_items to csr matrix with float32 type
        if not isinstance(user_items, sps.csr_matrix):
            print("converting user_items to CSR matrix")
            user_items = user_items.tocsr()
        if user_items.dtype != np.float32:
            print("converting type of user_items to np.float32")
            user_items = user_items.astype(np.float32)

        self.user_items = user_items
        self.user_items_csc = self.user_items.tocsc()
        self.user_count, self.item_count = self.user_items.shape
        p = self.user_items_csc.getnnz(axis=0)  # item frequencies
        p = (p / p.sum()) ** self.alpha  # item popularities
        self.Wi = (
            p / p.sum() * self.w0
        )  # confidence that item i missed by users is a true negative assessment

        # weights for squared errors of ratings
        # NOTE: Elements of W are fixed to be 1 as in the original implementation
        self.W = self.user_items.copy()
        self.W.data = np.ones(len(self.W.data)).astype(np.float32)
        self.W_csc = self.W.tocsc()

        # data and weights for online training
        self.user_items_lil = sps.lil_matrix((0, 0))
        self.user_items_lil_t = sps.lil_matrix((0, 0))
        self.W_lil = sps.lil_matrix((0, 0))
        self.W_lil_t = sps.lil_matrix((0, 0))

        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.U = self._init_U()
        self.V = self._init_V()
        self.SU = self.U.T @ self.U
        self.SV = (self.V.T * self.Wi) @ self.V

        self._training_mode = "batch"

    def _init_U(self):
        return np.random.normal(self.init_mean, self.init_stdev, (self.user_count, self.factors))

    def _init_V(self):
        return np.random.normal(self.init_mean, self.init_stdev, (self.item_count, self.factors))

    def _convert_data_for_online_training(self):
        # convert matrices to lil for online training
        if self._training_mode != "online":
            self.user_items_lil = self.user_items.tolil()
            self.user_items_lil_t = self.user_items_lil.T
            self.W_lil = self.W.tolil()
            self.W_lil_t = self.W_lil.T
            del self.user_items
            del self.user_items_csc
            del self.W
            del self.W_csc
            self._training_mode = "online"

    def _convert_data_for_batch_training(self):
        # convert matrices to csr for batch training
        if self._training_mode != "batch":
            self.user_items = self.user_items_lil.tocsr()
            self.user_items_csc = self.user_items.tocsc()
            self.W = self.W_lil.tocsr()
            self.W_csc = self.W.tocsc()
            del self.user_items_lil
            del self.user_items_lil_t
            del self.W_lil
            del self.W_lil_t
            self._training_mode = "batch"

    @property
    def user_factors(self):
        return self.U

    @property
    def item_factors(self):
        return self.V

    def update_user(self, u):
        """Update the user latent vector"""
        self._convert_data_for_online_training()
        old_user_vec = self.U[[u]]
        _update_user(
            u,
            np.array(self.user_items_lil.rows[u], dtype=np.int32),
            np.array(self.user_items_lil.data[u], dtype=np.float32),
            self.U,
            self.V,
            self.SV,
            np.array(self.W_lil.data[u], dtype=np.float32),
            self.Wi,
            self.factors,
            self.regularization,
        )
        return old_user_vec

    def update_SU(self, u, old_user_vec):
        _update_SU(self.SU, old_user_vec, self.U[[u]])

    def update_user_and_SU_all(self):
        self._convert_data_for_batch_training()
        _update_user_and_SU_all(
            self.user_items.indptr,
            self.user_items.indices,
            self.user_items.data,
            self.U,
            self.V,
            self.SU,
            self.SV,
            self.W.indptr,
            self.W.data,
            self.Wi,
            self.factors,
            self.regularization,
            self.user_count,
        )

    def update_item(self, i):
        """Update the item latent vector"""
        self._convert_data_for_online_training()
        old_item_vec = self.V[[i]]
        _update_item(
            i,
            np.array(self.user_items_lil_t.rows[i], dtype=np.int32),
            np.array(self.user_items_lil_t.data[i], dtype=np.float32),
            self.U,
            self.V,
            self.SU,
            np.array(self.W_lil_t.data[i], dtype=np.float32),
            self.Wi,
            self.factors,
            self.regularization,
        )
        return old_item_vec

    def update_SV(self, i, old_item_vec):
        _update_SV(self.SV, old_item_vec, self.V[[i]], self.Wi[i])

    def update_item_and_SV_all(self):
        self._convert_data_for_batch_training()
        _update_item_and_SV_all(
            self.user_items_csc.indptr,
            self.user_items_csc.indices,
            self.user_items_csc.data,
            self.U,
            self.V,
            self.SU,
            self.SV,
            self.W_csc.indptr,
            self.W_csc.data,
            self.Wi,
            self.factors,
            self.regularization,
            self.item_count,
        )

    def _expand_data(self, u, i):
        """Expand matrices for a new user-item pair if necessary"""
        extra_count = 100
        if u >= self.user_count:
            new_user_count = u + extra_count
        else:
            new_user_count = self.user_count
        if i >= self.item_count:
            new_item_count = i + extra_count
        else:
            new_item_count = self.item_count

        if new_user_count > self.user_count or new_item_count > self.item_count:
            self.user_items_lil.resize(new_user_count, new_item_count)
            self.user_items_lil_t.resize(new_item_count, new_user_count)
            self.W_lil.resize(new_user_count, new_item_count)
            self.W_lil_t.resize(new_item_count, new_user_count)
        if new_user_count > self.user_count:
            adding_user_count = new_user_count - self.user_count
            # user_count, factors
            self.U = np.vstack((self.U, np.zeros((adding_user_count, self.U.shape[1]))))
        if new_item_count > self.item_count:
            adding_item_count = new_item_count - self.item_count
            # item_count, factors
            self.V = np.vstack((self.V, np.zeros((adding_item_count, self.V.shape[1]))))
            self.Wi = np.append(self.Wi, np.zeros(adding_item_count))

        self.user_count = new_user_count
        self.item_count = new_item_count

    def update_model(self, u, i, show_loss: bool = False):
        """Update the model for single, possibly new user-item pair

        Parameters
        ----------
        u: int
            User index
        i: int
            Item index
        show_loss: bool
            Whether to compute and print the loss after each iteration
        """
        timer = Timer()
        self._convert_data_for_online_training()
        self._expand_data(u, i)
        self.user_items_lil[u, i] = 1
        self.user_items_lil_t[i, u] = 1
        self.W_lil[u, i] = 1  # w_new
        self.W_lil_t[i, u] = 1  # w_new
        # a new item
        if self.Wi[i] == 0:
            # NOTE: This update rule for Wi does not seem to be described in the paper.
            self.Wi[i] = self.w0 / self.item_count
            for f in range(self.factors):
                for k in range(f + 1):
                    val = self.SV[f, k] + self.V[i, f] * self.V[i, k] * self.Wi[i]
                    self.SV[f, k] = val
                    self.SV[k, f] = val

        for _ in range(self.num_iter_online):
            old_user_vec = self.update_user(u)
            self.update_SU(u, old_user_vec)
            old_item_vec = self.update_item(i)
            self.update_SV(i, old_item_vec)

        if show_loss:
            self.print_loss(1, "update_model", timer.elapsed())

    def print_loss(self, iter, message, elapsed):
        loss = self.calc_loss()
        print(f"iter={iter} {message} loss={loss:.4f} ({elapsed:.4f} sec)")

    def calc_loss(self):
        if self._training_mode == "batch":
            loss = _calc_loss_csr(
                self.user_items.indptr,
                self.user_items.indices,
                self.user_items.data,
                self.U,
                self.V,
                self.SV,
                self.W.indptr,
                self.W.data,
                self.Wi,
                self.user_count,
                self.regularization,
            )
        elif self._training_mode == "online":
            loss = _calc_loss_lil(
                self.user_items_lil_t.rows,
                self.user_items_lil_t.data,
                self.U,
                self.V,
                self.SV,
                self.W_lil_t.data,
                self.Wi,
                self.user_count,
                self.item_count,
                self.regularization,
            )
        else:
            raise NotImplementedError(
                f"calc_loss() for self._training_mode='{self._training_mode}' is not defined"
            )
        return loss

    def save(self, file: Union[Path, str], compress: Union[bool, int] = True):
        """Save the model in joblib format

        Parameters
        ----------
        file: Union[pathlib.Path, str]
            File to save the model
        compress: Union[bool, int]
            Joblib compression level (0-9).
            False or 0 disables compression.
            True (default) is equal to compression level 3.
        """
        serialize_eals_joblib(file, self, compress=compress)


def load_model(file: Union[Path, str]) -> ElementwiseAlternatingLeastSquares:
    """Load the model from a joblib file

    Parameters
    ----------
    file: Union[pathlib.Path, str]
        File to load the model from
    """
    return deserialize_eals_joblib(file)


# Actual implementation of eALS with numba jit

@njit(
    # TODO: Explicit type annotations slow down computation. Why?
    # "(i8,i4[:],f4[:],f8[:,:],f8[:,:],f8[:,:],f4[:],f8[:],i8,f8)"
)
def _update_user(u, item_inds, item_ratings, U, V, SV, w_items, Wi, factors, regularization):
    # Matrix U will be modified. Other arguments are read-only.
    if len(item_inds) == 0:
        return
    V_items = V[item_inds]
    pred_items = V_items @ U[u]

    w_diff = w_items - Wi[item_inds]
    w_item_ratings = w_items * item_ratings
    for f in range(factors):
        numer = 0
        for k in range(factors):
            if k != f:
                numer -= U[u, k] * SV[f, k]

        denom = SV[f, f] + regularization
        for i in range(len(item_inds)):
            pred_items[i] -= V_items[i, f] * U[u, f]
            numer += (w_item_ratings[i] - w_diff[i] * pred_items[i]) * V_items[i, f]
            denom += w_diff[i] * (V_items[i, f] ** 2)

        new_u = numer / denom
        U[u, f] = new_u
        for i in range(len(item_inds)):
            pred_items[i] += V_items[i, f] * new_u


@njit()
def _update_SU(SU, old_user_vec, new_user_vec):
    SU -= old_user_vec.T @ old_user_vec - new_user_vec.T @ new_user_vec


@njit(
    # "(i4[:],i4[:],f4[:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],i4[:],f4[:],f8[:],i8,f8,i8)",
    parallel=_USE_NUMBA_PARALLEL,
)
def _update_user_and_SU_all(
    indptr, indices, data, U, V, SU, SV, w_indptr, w_data, Wi, factors, regularization, user_count
):
    # U and SU will be modified. Other arguments are read-only.
    for u in prange(user_count):
        item_inds = indices[indptr[u] : indptr[u + 1]]
        item_ratings = data[indptr[u] : indptr[u + 1]]
        w_items = w_data[w_indptr[u] : w_indptr[u + 1]]
        _update_user(u, item_inds, item_ratings, U, V, SV, w_items, Wi, factors, regularization)
    # in-place assignment
    SU[:] = U.T @ U


@njit(
    # "(i8,i4[:],f4[:],f8[:,:],f8[:,:],f8[:,:],f4[:],f8[:],i8,f8)"
)
def _update_item(i, user_inds, user_ratings, U, V, SU, w_users, Wi, factors, regularization):
    # Matrix V will be modified. Other arguments are read-only.
    if len(user_inds) == 0:
        return
    U_users = U[user_inds]
    pred_users = U_users @ V[i]

    w_diff = w_users - Wi[i]
    w_users_rating = w_users * user_ratings
    for f in range(factors):
        numer = 0
        for k in range(factors):
            if k != f:
                numer -= V[i, k] * SU[f, k]
        numer *= Wi[i]

        denom = SU[f, f] * Wi[i] + regularization
        for u in range(len(user_inds)):
            pred_users[u] -= U_users[u, f] * V[i, f]
            numer += (w_users_rating[u] - w_diff[u] * pred_users[u]) * U_users[u, f]
            denom += w_diff[u] * (U_users[u, f] ** 2)

        new_i = numer / denom
        V[i, f] = new_i
        for u in range(len(user_inds)):
            pred_users[u] += U_users[u, f] * new_i


@njit()
def _update_SV(SV, old_item_vec, new_item_vec, Wii):
    SV -= (old_item_vec.T @ old_item_vec - new_item_vec.T @ new_item_vec) * Wii


@njit(
    # "(i4[:],i4[:],f4[:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],i4[:],f4[:],f8[:],i8,f8,i8)",
    parallel=_USE_NUMBA_PARALLEL,
)
def _update_item_and_SV_all(
    indptr, indices, data, U, V, SU, SV, w_indptr, w_data, Wi, factors, regularization, item_count
):
    # V and SV will be modified. Other arguments are read-only.
    for i in prange(item_count):
        user_inds = indices[indptr[i] : indptr[i + 1]]
        user_ratings = data[indptr[i] : indptr[i + 1]]
        w_users = w_data[w_indptr[i] : w_indptr[i + 1]]
        _update_item(i, user_inds, user_ratings, U, V, SU, w_users, Wi, factors, regularization)
    SV[:] = (V.T * Wi) @ V  # in-place assignment


@njit(
    # "(i4[:],i4[:],f4[:],f8[:,:],f8[:,:],f8[:,:],i4[:],f4[:],f8[:],i8,f8)"
)
def _calc_loss_csr(indptr, indices, data, U, V, SV, w_indptr, w_data, Wi, user_count, regularization):
    loss = ((U ** 2).sum() + (V ** 2).sum()) * regularization
    for u in range(user_count):
        item_indices = indices[indptr[u] : indptr[u + 1]]
        ratings = data[indptr[u] : indptr[u + 1]]
        weights = w_data[w_indptr[u] : w_indptr[u + 1]]
        for i, w, rating in zip(item_indices, weights, ratings):
            pred = U[u] @ V[i]
            loss += w * ((rating - pred) ** 2)
            loss -= Wi[i] * (pred ** 2)  # for non-missing items
        loss += (
            SV @ U[u] @ U[u]
        )  # sum of (Wi[i] * (pred ** 2)) for all (= missing + non-missing) items
    return loss


@njit(
    # "f8(f8[:,:],f8[:,:],f8[:,:],i8,f8)"
)
def _calc_loss_lil_init(U, V, SV, user_count, regularization):
    loss = ((U ** 2).sum() + (V ** 2).sum()) * regularization
    for u in range(user_count):
        loss += SV @ U[u] @ U[u]
    return loss


@njit(
    # "f8(i8,i4[:],f4[:],f4[:],f8[:,:],f8[:,:],f8[:])"
)
def _calc_loss_lil_inner_loop(i, indices, ratings, weights, U, V, Wi):
    l = 0
    for u, w, rating in zip(indices, weights, ratings):
        pred = U[u] @ V[i]
        l += w * ((rating - pred) ** 2)
        l -= Wi[i] * (pred ** 2)
    return l


def _calc_loss_lil(cols, data, U, V, SV, w_t_data, Wi, user_count, item_count, regularization):
    # TODO: @njit does not improve performance of this function. Better way to implement it?
    loss = _calc_loss_lil_init(U, V, SV, user_count, regularization)
    for i in range(item_count):
        if not cols[i]:
            continue
        user_indices = np.array(cols[i], dtype=np.int32)
        ratings = np.array(data[i], dtype=np.float32)
        weights = np.array(w_t_data[i], dtype=np.float32)
        loss += _calc_loss_lil_inner_loop(i, user_indices, ratings, weights, U, V, Wi)
    return loss
