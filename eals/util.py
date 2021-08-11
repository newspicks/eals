import datetime

import numpy as np
import scipy.sparse as sps


def create_user_items(
    user_count: int = 2000,
    item_count: int = 1000,
    data_count: int = 2000 * 20,
    # rating_fn must return a float array of shape (data_count,)
    rating_fn=lambda data_count: (np.random.rand(data_count) * 10 + 2).astype(np.float32),
    random_seed=None,
) -> sps.spmatrix:
    """Create random rating matrix

    Parameters
    ----------
    user_count: int
        The number of users
    item_count: int
        The number of items
    data_count: int
        The number of non-zero elements in the matrix
    rating_fn: Callable[[int], float]
        The function to generate the rating matrix
    random_seed: int
        The random seed
    """
    if random_seed:
        np.random.seed(random_seed)
    data = rating_fn(data_count)
    u = np.random.randint(0, user_count, size=data_count)
    i = np.random.randint(0, item_count, size=data_count)
    return sps.csr_matrix((data, (u, i)), shape=(user_count, item_count))


class Timer:
    """Measure elapsed time"""

    def __init__(self) -> None:
        self.start_time = datetime.datetime.now()

    def elapsed(self) -> float:
        """Returns the elapsed time since the last call"""
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        self.start_time = end_time
        return elapsed_time.total_seconds()
