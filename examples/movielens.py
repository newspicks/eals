"""
usage: movielens.py [-h] {fit,update,recommend} ...

Example recommender based on the MovieLens 20M dataset

positional arguments:
  {fit,update,recommend}
    fit                 Fit the model
    update              Update the model when a new rating is added
    recommend           Recommend top k movies to the given user

optional arguments:
  -h, --help            show this help message and exit
"""
import csv
import os
from argparse import ArgumentParser
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import scipy.sparse as sps

from eals import ElementwiseAlternatingLeastSquares, load_model

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")


# Data utilities
def download_data():
    # Download movielens data if it doesn't exist
    zip_path = os.path.join(BASE_DIR, "ml-20m.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading movielens data from {DATA_URL} to {BASE_DIR}")
        data = urlopen(DATA_URL).read()
        with open(zip_path, mode="wb") as f:
            f.write(data)
        print("Extracting movielens data")
        with ZipFile(zip_path) as zf:
            zf.extractall(BASE_DIR)


def load_ratings():
    download_data()
    # Create the rating matrix
    # Keep only rows with rating > 3 for the implicit feedback setting
    print("Loading the training data")
    with open(os.path.join(BASE_DIR, "ml-20m", "ratings.csv"), newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        cols = []
        vals = []
        for line in reader:
            if float(line["rating"]) > 3:
                rows.append(int(line["userId"]))
                cols.append(int(line["movieId"]))
                vals.append(1.0)
    ratings = sps.csr_matrix(
        (vals, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1), dtype=np.float32
    )
    return ratings


def load_titles():
    download_data()
    # Create the movie title dictionary
    print("Loading the title dictionary")
    with open(os.path.join(BASE_DIR, "ml-20m", "movies.csv"), newline="") as f:
        reader = csv.DictReader(f)
        titles = {int(line["movieId"]): line["title"] for line in reader}
    return titles


# Commands
def parse_args():
    parser = ArgumentParser(description="Example recommender based on the MovieLens 20M dataset")
    subparsers = parser.add_subparsers(dest="subcommand")
    parser_fit = subparsers.add_parser("fit", help="Fit the model")
    parser_fit.add_argument(
        "--num_iter", type=int, default=500, help="Number of training iterations"
    )
    parser_update = subparsers.add_parser(
        "update", help="Update the model when a new rating is added"
    )
    parser_update.add_argument("--user_id", type=int, default=0)
    parser_update.add_argument("--movie_id", type=int, default=0)
    parser_recommend = subparsers.add_parser(
        "recommend", help="Recommend top k movies to the given user"
    )
    parser_recommend.add_argument("--user_id", type=int, default=0)
    parser_recommend.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def fit(args):
    ratings = load_ratings()
    print("Fitting the model")
    model = ElementwiseAlternatingLeastSquares(num_iter=args.num_iter)
    model.fit(ratings, show_loss=True)
    print(f"Saving the model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Done")


def update(args):
    print(f"Loading the model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Updating the model")
    model.update_model(args.user_id, args.movie_id)
    print(f"Saving the model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Done")


def recommend(args):
    print(f"Loading the model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    titles = load_titles()
    print(f"Searching Top {args.k} recommended movies for user_id={args.user_id}")
    user_vector = model.user_factors[args.user_id]
    pred_ratings = model.item_factors @ user_vector
    topk_movie_ids = reversed(np.argsort(pred_ratings)[-args.k :])
    print("Done\n")
    print("rank (score): title")
    for rank, id_ in enumerate(topk_movie_ids, start=1):
        print(f"{rank:4d} ( {pred_ratings[id_]:3.2f}): {titles[id_]}")


def main():
    args = parse_args()
    if args.subcommand == "fit":
        fit(args)
    if args.subcommand == "update":
        update(args)
    if args.subcommand == "recommend":
        recommend(args)


if __name__ == "__main__":
    main()
