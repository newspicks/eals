import numpy as np
import scipy.sparse as sps
import json
import datetime


def create_user_items(
    user_count=2000,
    item_count=1000,
    data_count=2000 * 20,
    new_user_count=200,
    new_item_count=100,
    # rating_fn must return a float array of shape (data_count,)
    rating_fn=lambda data_count: (np.random.rand(data_count) * 10 + 2).astype(np.float32),
    random_seed=None,
):
    if random_seed:
        np.random.seed(random_seed)
    data = rating_fn(data_count)
    u = np.random.randint(0, user_count, size=data_count)
    i = np.random.randint(0, item_count, size=data_count)
    # new_user_count, new_item_countの分だけ新規ユーザ、新規アイテム格納用の余白を持たせておく
    return sps.csr_matrix((data, (u, i)), shape=(user_count + new_user_count, item_count + new_item_count))


class Timer:
    def __init__(self):
        self.start_time = datetime.datetime.now()

    def elapsed(self):
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        self.start_time = end_time
        return elapsed_time.total_seconds()
