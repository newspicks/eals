import random

import click
import numpy as np
import json
import scipy.sparse as sp

from .eals import ElementwiseAlternatingLeastSquares
from .util import create_user_items


@click.command(help="run MF for all feedback data")
@click.option("--max-iter", type=int, default=500)
def main(max_iter):
    batch_user_items, online_user_items = create_user_items2()
    model = ElementwiseAlternatingLeastSquares(
        random_state=8, show_loss=True, max_iter=max_iter
    )
    model.fit(batch_user_items)
    for vec in model.user_factors()[:3]:
        print(f"user: {vec}")
    for vec in model.item_factors()[:3]:
        print(f"item: {vec}")
    for i in range(online_user_items.shape[0]):
        for j in online_user_items.indices[online_user_items.indptr[i] : online_user_items.indptr[i + 1]]:
            model.update_model(i, j)

def create_user_items1():
    batch_user_items = create_user_items(
        user_count=2000,
        item_count=1000,
        data_count=2000 * 20,
        new_user_count=0,
        new_item_count=0,
        rating_fn=lambda data_count: (np.random.rand(data_count) * 10 + 2).astype(np.float32),
        random_seed=8,
    )
    online_user_items = create_user_items(
        user_count=2000,
        item_count=1000,
        data_count=100,
        new_user_count=0,
        new_item_count=0,
        rating_fn=lambda data_count: (np.random.rand(data_count) * 10 + 2).astype(np.float32),
        random_seed=8,
    )
    return batch_user_items, online_user_items


def create_user_items2():
    filename = "../exp02/data/2020-08-24-view0100.json"

    print(f"loading {filename}")
    with open(filename) as f:
        data = json.load(f)
    train_data = data["train_data"]
    test_data = data["test_data"]
    train_max_user_ind = max([x[0] for x in train_data])
    train_max_item_ind = max([x[1] for x in train_data])
    test_max_user_ind = max([x[0] for x in test_data])
    test_max_item_ind = max([x[1] for x in test_data])
    print(f"train_max_user_ind={train_max_user_ind}, train_max_item_ind={train_max_item_ind}")
    print(f"test_max_user_ind={test_max_user_ind}, test_max_item_ind={test_max_item_ind}")
    max_user_ind = max([train_max_user_ind, test_max_user_ind])
    max_item_ind = max([train_max_item_ind, test_max_item_ind])
    batch_user_items = create_csr_matrix(train_data, user_count=max_user_ind + 1, item_count=max_item_ind + 1)
    online_user_items = create_csr_matrix(test_data)
    return batch_user_items, online_user_items

def create_csr_matrix(data, user_count=None, item_count=None):
    # 重みを変更
    weights = [1, 1, 1.5, 1.5]
    print("modifying weights")
    new_data = [(x[0], x[1], weights[0]) for x in data if x[2] == 0.01]
    new_data += [(x[0], x[1], weights[1]) for x in data if x[2] == 0.05]
    new_data += [(x[0], x[1], weights[2]) for x in data if x[2] == 1]
    new_data += [(x[0], x[1], weights[3]) for x in data if x[2] == 2]
    data = new_data

    print("creating CSR matrix")
    user_inds, item_inds, weights = zip(*data)
    if user_count and item_count:
        user_items = sp.csr_matrix((weights, (user_inds, item_inds)), shape=(user_count, item_count), dtype=np.float32)
    else:
        user_items = sp.csr_matrix((weights, (user_inds, item_inds)), dtype=np.float32)
    print(f"user_items: size={user_items.shape}, nnz={user_items.nnz}")
    return user_items

if __name__ == "__main__":
    main()
