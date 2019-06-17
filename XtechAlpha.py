import numpy as np
import pandas as pd
from scipy.stats import rankdata


def ts_max(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).max()


def ts_argmax(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).apply(np.argmax)


def ts_argmin(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).apply(np.argmin)


def ts_rank(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).apply(lambda x: d + 1 - rankdata(x)[-1])


def sum(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).sum()


def product(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).apply(np.prod)


def stddev(x: pd.core.frame.DataFrame, d: int) -> pd.core.frame.DataFrame:

    return x.rolling(d).std()
