## Tests

```sh
USE_NUMBA=0 pytest --disable-pytest-warnings
```

`USE_NUMBA=0`はテスト高速化のため（numbaのコンパイル時間をなくす）。numbaありでテストしたければ`USE_NUMBA=1`にする。
