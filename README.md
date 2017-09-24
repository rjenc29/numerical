# numerical
A repository of numerical routines and research ideas implemented in Python, using [numba](http://numba.pydata.org/) with CPU and GPU targets, which may offer substantial speed improvements over numpy / pandas equivalents.

Example:

```bash
data = np.random.rand(1000000)
series = pd.Series(data)
window = 200
%time pandas_output = series.rolling(window=window).sum()
Wall time: 50.9 ms
%time fast_output = rolling_sum(data, window)
Wall time: 5.89 ms
np.allclose(pandas_output, fast_output, equal_nan=True)
Out[23]: True

```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
