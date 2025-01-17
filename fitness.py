# encoding:utf-8
"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers
import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata
import pandas as pd

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better,stock_is = None):
        self.function = function
        self.stock_is = stock_is
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args, **kargs):
        return self.function(*args, **kargs)


def make_fitness(function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, ww, min_size=20, decay=False):
    """Calculate the weighted Pearson correlation coefficient."""
    corrs = []
    avail_dates = y.columns.intersection(y_pred.columns)
    for dt in avail_dates:
        y_pred_sub = y_pred[dt].dropna()
        y_sub = y[dt].dropna()
        mask = y_pred_sub.index.intersection(y_sub.index)
        if len(y_pred_sub.loc[mask].unique()) > min_size:
            if decay:
                decay_f = 0.5 ** (1 / y_sub.shape[0] / decay)
                ww_sub = list(decay_f ** (y_pred_sub.loc[mask].rank(ascending=False) - 1))
            else:
                ww_sub = list(ww[mask])        
            y_pred_sub = y_pred_sub.loc[mask].values
            y_sub = y_sub.loc[mask].values
            with np.errstate(divide='ignore', invalid='ignore'):
                y_pred_demean = y_pred_sub - np.average(y_pred_sub, weights=ww_sub)
                y_demean = y_sub - np.average(y_sub, weights=ww_sub)
                corr = ((np.sum(ww_sub * y_pred_demean * y_demean) / np.sum(ww_sub)) / np.sqrt((np.sum(ww_sub * y_pred_demean ** 2) * np.sum(ww_sub * y_demean ** 2)) / (np.sum(ww_sub) ** 2)))
                corrs.append(corr)
    return pd.Series(corrs)
    # if np.isfinite(corr):
    #     return np.abs(corr)
    # return 0.


def _weighted_spearman(yy, y_pred, w, min_size=20, decay=False):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = y_pred.rank()
    yy = yy.copy()
    y_ranked = yy.rank()
    return _weighted_pearson(y_ranked, y_pred_ranked, w, min_size, decay)

def _weighted_spearman_icir(y, y_pred, w, min_size=(10, 20), max_res=1000, err_res=-10, decay=False):
    """Calculate the weighted Spearman correlation coefficient icir."""
    min_size1, min_size2 = min_size
    corrs = _weighted_spearman(y, y_pred, w, min_size2, decay)
    res1 = corrs.mean() / corrs.std()

    if len(corrs) <= min_size1:
        return err_res

    if np.isnan(res1) or res1 > max_res:
        return err_res

    return res1

def _weighted_spearman_icir_mix(y, y_pred, w, min_size=(10, 20), max_res=1000, err_res=-10, decay=False, shift_num=10, mix_wt=1):
    """Calculate the weighted Spearman correlation coefficient icir. Calculated for both y_pred and y_pred shifted and combine"""
    res1 = _weighted_spearman_icir(y, y_pred, w, min_size, max_res, err_res, decay)
    y_pred2 = y_pred.T.sort_index().shift(shift_num).T
    res2 = _weighted_spearman_icir(y, y_pred2, w, min_size, max_res, err_res, decay)
    wt1 = 1/(1+mix_wt)
    wt2 = mix_wt/(1+mix_wt)

    return res1 * wt1 + res2 * wt2

def _defrequency_ret_data(ret_data, holding):
    ret_data = ret_data.sort_index()
    if holding > 1:
        ret_data['idx'] = np.arange(ret_data.shape[0])
        test_ret = ((ret_data + 1).groupby(ret_data.idx // holding)).prod().drop('idx', axis=1)
        test_ret.index = ret_data.index[0::holding]
        test_ret.index.name = 'TradingDay'
        test_ret -= 1
        return test_ret.replace(0,np.nan)
    return ret_data

def _weighted_spearman_icir_defreq(yy, y_pred, w, min_size=(10, 20), max_res=1000, err_res=-10, decay=False, step_size=5):
    """Calculate the weighted Spearman correlation coefficient icir. Use subsampling y_pred, and rolling sum y."""
    yy2 = _defrequency_ret_data(yy.T.sort_index(), step_size)
    y_pred2 = y_pred.T.loc[yy2.index.intersection(y_pred.T.index)].T
    yy2 = yy2.T
    return _weighted_spearman_icir(yy2, y_pred2, w, min_size, max_res, err_res, decay)

def _long_only_performance(y, y_pred, w, top_ratio=0.2):
    rets = []
    avail_dates = y.columns.intersection(y_pred.columns)
    for dt in avail_dates:
        y_pred_sub = y_pred[dt].dropna()
        y_sub = y[dt].dropna()
        mask = y_pred_sub.index.intersection(y_sub.index)
        if len(mask):
            y_pred_sub = y_pred_sub.loc[mask]
            y_sub = y_sub.loc[mask]
            ret = y_sub.loc[y_pred_sub[y_pred_sub.rank(pct=True, ascending=False) <= top_ratio].index].mean()
            rets.append(ret)
    return pd.Series(rets)

def _long_only_sharpe(y, y_pred, w, min_size=10, max_res=1000, err_res=-10, top_ratio=0.2):
    rets = _long_only_performance(y, y_pred, w, top_ratio)
    sharpe = rets.mean() / rets.std()

    if len(rets) <= min_size:
        return err_res

    if np.isnan(sharpe) or sharpe > max_res:
        return err_res

    return sharpe

def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)

#为了不破坏整体的结构，以原结构进行比较
def _stock_dedicated(y,y_pred,w):
    return np.average(y_pred,weights = w)

def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)


weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True,
                            stock_is=True)
weighted_spearman_icir = _Fitness(function=_weighted_spearman_icir,
                             greater_is_better=True,
                             stock_is=True)
weighted_spearman_icir_defreq = _Fitness(function=_weighted_spearman_icir_defreq,
                             greater_is_better=True,
                             stock_is=True)
weighted_spearman_icir_mix = _Fitness(function=_weighted_spearman_icir_mix,
                             greater_is_better=True,
                             stock_is=True)
long_only_sharpe = _Fitness(function=_long_only_sharpe,
                             greater_is_better=True,
                             stock_is=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)
stock_dedicated = _Fitness(function=_stock_dedicated,
                    greater_is_better=True,
                    stock_is = True)
                    
                    

_fitness_map = {'pearson': weighted_pearson,
                'spearman_icir': weighted_spearman_icir,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss,
                'stock_dedicated':stock_dedicated,
                'long_only_sharpe': long_only_sharpe,
                'spearman_icir_mix': weighted_spearman_icir_mix,
                'spearman_icir_defreq': weighted_spearman_icir_defreq}
