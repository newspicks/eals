import json
import os
import time
import warnings

os.environ["USE_NUMBA"] = "1"
warnings.simplefilter("ignore")

import click
import numpy as np
import scipy.sparse as sps

from eals.eals import ElementwiseAlternatingLeastSquares
from eals.util import create_user_items


def load_data(file):
    # load cache if it exists
    file_npy = os.path.splitext(file)[0] + "-train_data.npy"
    if os.path.exists(file_npy):
        print(f"(use cache: {file_npy}) ", end="")
        with open(file_npy, "rb") as f_npy:
            data = np.load(f_npy)
    else:
        with open(file) as f:
            rawdata = json.load(f)["train_data"]
        data = np.array(rawdata)
        np.save(file_npy, data)
    u = data[:, 0].astype(np.int64)
    i = data[:, 1].astype(np.int64)
    d = data[:, 2].astype(np.float32)
    return u, i, d


def bench_real_data(file, mat_type):
    print(f"real data ({os.path.basename(file)}), {mat_type} matrix")

    print("  load data: ", end="")
    t = time.time()
    u, i, d = load_data(file)
    user_count = u.max()
    item_count = i.max()
    nnz = len(d)
    new_user_count = int(user_count / 100)
    new_item_count = int(item_count / 100)
    train_data = sps.csr_matrix(
        (d, (u, i)), shape=(user_count + new_user_count, item_count + new_item_count)
    )
    del u, i, d
    print(f"{time.time() - t} sec")
    print(f"    {user_count=}, {item_count=}, {nnz=}")

    print("  setup: ", end="")
    t0 = time.time()
    model = ElementwiseAlternatingLeastSquares()
    model._init_data(train_data)
    if mat_type == "lil":
        model._convert_data_for_online_training()
    t1 = time.time()
    print(f"{t1-t0} sec")

    print("  elapsed: ", end="")
    t2 = time.time()
    model.calc_loss()
    t3 = time.time()
    print(f"{t3-t2} sec")


def bench_random_data(mat_type):
    print(f"random data, {mat_type} matrix")
    for user_count, item_count in [
        (2000, 1000),
        (20000, 10000),
        (200000, 100000),
        (2000000, 1000000),
    ]:
        print("  create data: ", end="")
        t = time.time()
        data_count = user_count * 20
        train_data = create_user_items(
            user_count=user_count,
            item_count=item_count,
            data_count=data_count,
        )
        print(f"{time.time() - t} sec")
        print(f"    {user_count=}, {item_count=}")

        print("  setup: ", end="")
        t0 = time.time()
        model = ElementwiseAlternatingLeastSquares()
        model._init_data(train_data)
        if mat_type == "lil":
            model._convert_data_for_online_training()
        t1 = time.time()
        print(f"{t1-t0} sec")
        print("  elapsed: ", end="")
        t2 = time.time()
        model.calc_loss()
        t3 = time.time()
        print(f"{t3-t2} sec")


@click.command()
@click.option("-f", "--file", type=str, required=False)
@click.option("-m", "--mat-type", type=click.Choice(["csr", "lil"]), required=True, default="csr")
def main(file, mat_type):
    print("Benchmarking calc_loss()")
    if file:
        bench_real_data(file, mat_type)
    else:
        bench_random_data(mat_type)


if __name__ == "__main__":
    main()
