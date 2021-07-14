import numpy as np
import scipy.sparse as sps


def create_user_items(
    user_count=2000,
    item_count=1000,
    data_count=2000 * 20,
    # rating_fn must return a float array of shape (data_count,)
    rating_fn=lambda data_count: (np.random.rand(data_count) * 10 + 2).astype(np.float32),
    random_seed=None,
):
    if random_seed:
        np.random.seed(random_seed)
    data = rating_fn(data_count)
    u = np.random.randint(0, user_count, size=data_count)
    i = np.random.randint(0, item_count, size=data_count)
    return sps.csr_matrix((data, (u, i)), shape=(user_count, item_count))
