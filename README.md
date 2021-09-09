# eALS - Element-wise Alternating Least Squares

A Python implementation of the element-wise alternating least squares (eALS) for fast online matrix factorization proposed by [arXiv:1708.05024](https://arxiv.org/abs/1708.05024).

## Prerequisites

- Python >= 3.8

## Installation

```sh
pip install eals
```

## Usage

```python
import numpy as np
import scipy.sparse as sps
from eals import ElementwiseAlternatingLeastSquares, load_model

# batch training
user_items = sps.csr_matrix([[1, 2, 0, 0], [0, 3, 1, 0], [0, 4, 0, 4]], dtype=np.float32)
model = ElementwiseAlternatingLeastSquares(factors=2)
model.fit(user_items)

# learned latent vectors
model.user_factors
model.item_factors

# online training for new data (user_id, item_id)
model.update_model(1, 0)

# rating matrix and latent vectors will be expanded for a new user or item
model.update_model(0, 5)

# current rating matrix
model.user_items

# save and load the model
model.save("model.joblib")
model = load_model("model.joblib")
```

See the [examples](examples/) directory for complete examples.

## Development

### Setup development environment

```sh
git clone https://github.com/newspicks/eals.git
cd eals
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
You may need to put the Python version numbers in the `.python-version` file.

```sh
poetry run tox
```
