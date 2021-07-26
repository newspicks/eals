# Element-wise Alternating Least Squares (eALS)

A Python implementation of the element-wise alternating least squares (eALS) for fast online matrix factorization proposed by [arXiv:1708.05024](https://arxiv.org/abs/1708.05024).

## Prerequisites

- Python >= 3.9

## Installation

```sh
pip install git+https://github.com/newspicks/implicit-eals.git
```

## Usage

```python
from eals import ElementWiseAlternatingLeastSquares, load_model

# Batch training
model = ElementWiseAlternatingLeastSquares()
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
poetry install
```

### Tests

Set `USE_NUMBA=0` for faster testing without numba JIT overhead.

```sh
USE_NUMBA=0 pytest
```
