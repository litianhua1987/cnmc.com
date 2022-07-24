import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def get_targets(y, tau):
    """
    (y[t+1]-y[t])/y[t+1] approximates to: np.log(y[t+1]) - np.log(y[t]) = np.log(y[t+1]/y[t])
    https://stats.stackexchange.com/questions/244199/why-is-it-that-natural-log-changes-are-percentage-changes-what-is-about-logs-th
    :param y:
    :param tau:
    :return:
    """
    agg = y.rolling(tau).median()
    return np.log(agg.shift(-tau).div(agg)).iloc[tau:-tau]

def run_adf_test(df: pd.DataFrame, *args, **kwargs):

    select_cols = ['adf', 'pvalue']
    res = df.apply(lambda col: adfuller(col), *args, **kwargs, axis=0)
    res.index = ['adf', 'pvalue', 'usedlag', 'nobs', 'crit_values', 'icbest']

    return res.T[select_cols]

def get_wave_features(X, periods=[12], n_harmonics=1):
    X_wave = pd.DataFrame(dtype=float)
    for tau in periods:
        for i in range(1, n_harmonics+1):
            X_wave[f"comp_{tau}_sin{i}"] = np.sin((X - 1) * (2 * np.pi / tau)) ** i
            X_wave[f"comp_{tau}_cos{i}"] = np.cos((X - 1) * (2 * np.pi / tau)) ** i

    return X_wave

def compute_macd(x: pd.DataFrame, short_tau: int, long_tau: int):
    short = x.rolling(short_tau, min_periods=1).mean()
    long = x.rolling(long_tau, min_periods=1).mean()
    return short.subtract(long)




class MACD(BaseEstimator, TransformerMixin):
    """
    Computes MACD (Moving Average Convergence Divergence) Indicator
    Parameters
    ----------
    short_tau: short moving average period
    long_tau: long moving average period
    zscore_tau:
    flg_zcore:
    See Also
    --------
    https://www.investopedia.com/terms/m/macd.asp
    """

    def __init__(self, short_tau=5, long_tau=20, zscore_tau=60, flg_zcore=True):
        self.short_tau = short_tau
        self.long_tau = long_tau
        self.zscore_tau = zscore_tau
        self.flg_zcore = flg_zcore

    def fit(self, X=None, y=None, **fit_params):
        """
        Fit each model on its correspondent set of features-targets. Retrieves column metadata in xcols
        :param X: array-like, full set of features
        :param y: array-like, full set of targets
        :return: returns an instance of self.
        """

        return self

    def transform(self, X, **transform_params):
        """
        Get predictions for each model based on its correspondent set of features. Retrieves column metadata in xcols
        Predictions are horizontally stacked.
        :param X: array-like, Full set of features
        :return: array-like, 2d array of predictions
        """
        X = check_array(X)

        short = pd.DataFrame(X).rolling(self.short_tau, min_periods=1).mean()
        long = pd.DataFrame(X).rolling(self.long_tau, min_periods=1).mean()
        macd = (short-long)

        if self.flg_zcore:
            macd_zscored = z_score(x=macd, win_size=self.zscore_tau, min_periods=1, fillna=True)

            return macd_zscored
        else:
            return macd


class BBands(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------

    Attributes
    ----------

    See Also
    --------

    """

    def __init__(self, win_size: int = 60, threshold: float = 2):
        self.threshold = threshold
        self.win_size = win_size

    def fit(self, X=None, y=None, **fit_params):
        """
        Fit each model on its correspondent set of features-targets. Retrieves column metadata in xcols
        :param X: array-like, full set of features
        :param y: array-like, full set of targets
        :return: returns an instance of self.
        """

        return self

    def transform(self, X, **transform_params):
        """
        Get predictions for each model based on its correspondent set of features. Retrieves column metadata in xcols
        Predictions are horizontally stacked.
        :param X: array-like, Full set of features
        :return: array-like, 2d array of predictions
        """
        X = check_array(X)

        bbands = lambda x: [np.sign(ex) if abs(ex) >= 2 else 0 for ex in x]

        roll = pd.DataFrame(X).rolling(self.win_size, min_periods=1)
        mu = roll.mean()
        std = roll.std()
        zscore = (X - mu)/std
        zscore.fillna(method='bfill', inplace=True)

        return zscore.apply(bbands, axis=0).astype(int)


class Momentum(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------

    Attributes
    ----------

    See Also
    --------

    """

    def __init__(self, tau: int = 60, zscore_tau: int = 120, fit_intercept: bool = True, degree: int = 2, flg_zcore: bool = True):
        self.tau = tau
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.flg_zcore = flg_zcore
        self.zscore_tau = zscore_tau

    def fit(self, X=None, y=None, **fit_params):
        """
        Fit each model on its correspondent set of features-targets. Retrieves column metadata in xcols
        :param X: array-like, full set of features
        :param y: array-like, full set of targets
        :return: returns an instance of self.
        """

        return self

    def transform(self, X, **transform_params):
        """
        Get predictions for each model based on its correspondent set of features. Retrieves column metadata in xcols
        Predictions are horizontally stacked.
        :param X: array-like, Full set of features
        :return: array-like, 2d array of predictions
        """
        X = check_array(X)

        momentum = pd.DataFrame(X).rolling(self.tau, min_periods=1).\
            apply(deter_trend_beta,  args=(self.degree,))

        if self.flg_zcore:
            zscored = z_score(x=momentum, win_size=self.zscore_tau, min_periods=1, fillna=True)
            return zscored
        else:
            return momentum

class Diff(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------

    Attributes
    ----------

    See Also
    --------

    """

    def __init__(self, tau: int = 60, zscore_tau=60, flg_zcore: bool = True):
        self.tau = tau
        self.flg_zcore = flg_zcore
        self.zscore_tau = zscore_tau

    def fit(self, X=None, y=None, **fit_params):
        """
        Fit each model on its correspondent set of features-targets. Retrieves column metadata in xcols
        :param X: array-like, full set of features
        :param y: array-like, full set of targets
        :return: returns an instance of self.
        """

        return self

    def transform(self, X, **transform_params):
        """
        Get predictions for each model based on its correspondent set of features. Retrieves column metadata in xcols
        Predictions are horizontally stacked.
        :param X: array-like, Full set of features
        :return: array-like, 2d array of predictions
        """
        X = check_array(X)

        raw = pd.DataFrame(X).pct_change(self.tau).fillna(0.)

        if self.flg_zcore:
            zscored = z_score(x=raw, win_size=self.zscore_tau, min_periods=1, fillna=True)
            return zscored
        else:
            return raw

class Volatility(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------

    Attributes
    ----------

    See Also
    --------

    """

    def __init__(self, tau: int = 60, zscore_tau=60, flg_zcore: bool = True):
        self.tau = tau
        self.flg_zcore = flg_zcore
        self.zscore_tau = zscore_tau

    def fit(self, X=None, y=None, **fit_params):
        """
        Fit each model on its correspondent set of features-targets. Retrieves column metadata in xcols
        :param X: array-like, full set of features
        :param y: array-like, full set of targets
        :return: returns an instance of self.
        """

        return self

    def transform(self, X, **transform_params):
        """
        Get predictions for each model based on its correspondent set of features. Retrieves column metadata in xcols
        Predictions are horizontally stacked.
        :param X: array-like, Full set of features
        :return: array-like, 2d array of predictions
        """
        X = check_array(X)

        raw = pd.DataFrame(X).rolling(self.tau, min_periods=1).std()

        if self.flg_zcore:
            zscored = z_score(x=raw, win_size=self.zscore_tau, min_periods=1, fillna=True)
            return zscored
        else:
            return raw


def z_score(x, win_size, min_periods=1, fillna=True):
    """
    Regress a vector y against its index, starting at 0
    :param y: vector to regress on
    :return: regression coefficient as float
    """
    roll = x.rolling(win_size, min_periods)
    mu = roll.mean()
    std = roll.std()
    zscored = (x - mu) / std
    if fillna:
        zscored.fillna(method='bfill', inplace=True)

    return zscored

def deter_trend_beta(y: np.array, degree=2):
    """
    Regress a vector y against its index, starting at 0
    :param y: vector to regress on
    :return: regression coefficient as float
    """
    nrows = len(y)

    x = np.zeros((nrows, degree))
    for idx_col in range(degree):
        deg = idx_col+1
        x[:, idx_col] = np.arange(0, nrows) ** deg

    lr = LinearRegression()
    lr.fit(x, y)

    return lr.coef_.sum()