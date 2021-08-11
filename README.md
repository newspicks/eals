# Element-wise Alternating Least Squares (eALS)

A Python implementation of the element-wise alternating least squares (eALS) for fast online matrix factorization proposed by [arXiv:1708.05024](https://arxiv.org/abs/1708.05024).

## Prerequisites

- Python >= 3.8

## Installation

```sh
pip install git+https://github.com/newspicks/implicit-eals.git
```

## Usage

```python
from eals import ElementwiseAlternatingLeastSquares, load_model

# Batch training
model = ElementwiseAlternatingLeastSquares()
model.fit(rating_data)

# Learned latent vectors
model.user_factors
model.item_factors

# Online training for new data
model.update_model(user_id, item_id)

# Save and load the model
model.save("model.joblib")
model = load_model("model.joblib")
```

See the [examples](examples/) directory for complete examples.

## Development

### Setup development environment

```sh
git clone https://github.com/newspicks/implicit-eals.git
cd implicit-eals
poetry run pip install -U pip
poetry install
```

### Tests

```sh
poetry run pytest
```

Set `USE_NUMBA=0` for faster testing without numba JIT overhead.

```sh
USE_NUMBA=0 poetry run pytest
```

To run tests against all supported Python versions, use [tox](https://tox.readthedocs.io/).

```sh
poetry run tox
```
