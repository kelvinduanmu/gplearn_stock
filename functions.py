"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, ts=False, assist=False):
        self.function = function
        self.name = name
        self.arity = arity
        self.ts = ts
        self.assist = assist

    def __call__(self, *args):
        args2 = []
        for ele in args:
            if ele.index.nlevels != 2:
                print("Input should have 2 levels of index")
                raise
            if ele.shape[0] == 0:
                return ele
            args2.append(ele.sort_index())
        if self.ts:
            args2.append(self.d1)
        return self.function(*args2)

    def set_d1(self, d1):
        self.d1 = d1

    def set_d1_list(self, d1_list):
        self.d1_list = d1_list

    def set_assist_col(self, idx):
        """
        idx: index of risk_cols
        """
        self.assist_col = idx

def make_function(function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    #print (arity,'niaho')
    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(x1, x2).fillna(1)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.abs(x1)).fillna(0)

def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return (1. / x1).fillna(0)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def _rank_cross(x1):
    return x1.groupby(level=1).rank(pct=True)

def _rank(x1, d1):
    d1 = max(3, d1)
    y1 = x1.groupby(level=0).rolling(d1).apply(lambda x: (x<x.iloc[-1]).sum()/x.shape[0])
    return y1.droplevel(0)

def _delay(x1, d1):
    y1 = x1.groupby(level=0).shift(d1)
    return y1

def _ts_corr(x1, y1, d1):
    d1 = max(3, d1)
    comb_data = pd.concat([x1, y1], axis=1).unstack()
    result = pd.Series(index=x1.index).unstack(level=0)
    for code in comb_data.index:
        res = comb_data.loc[code].unstack(level=0).sort_index().rolling(d1).corr().drop(comb_data.columns.levels[0][0], level=1).droplevel(1)[comb_data.columns.levels[0][0]]
        if res.dropna().shape[0]:
            result[code] = res
    return result.stack().swaplevel(0,1)

def _ts_cov(x1, y1, d1):
    d1 = max(3, d1)
    comb_data = pd.concat([x1, y1], axis=1).unstack()
    result = pd.Series(index=x1.index).unstack(level=0)
    for code in comb_data.index:
        res = comb_data.loc[code].unstack(level=0).sort_index().rolling(d1).cov().drop(comb_data.columns.levels[0][0], level=1).droplevel(1)[comb_data.columns.levels[0][0]]
        if res.dropna().shape[0]:
            result[code] = res
    return result.stack().swaplevel(0,1)

def _scale(x1):
    return x1 / x1.abs().groupby(level=1).sum()

def _delta(x1, d1):
    y1 = x1.groupby(level=0).diff(d1)
    return y1

# def _decay_linear(x1, d1):
#     d1 = max(3, d1)
#     y1 = x1.unstack(level=0).rolling(d1).apply(lambda x: (x*np.arange(1,d1+1)).sum()/np.arange(1, d1+1).sum(), raw=True)
#     return y1.stack().swaplevel(0,1)

# def _decay_exp(x1, d1):
#     d1 = max(3, d1)
#     y1 = x1.unstack(level=0).rolling(d1).apply(lambda x: (x*np.exp(-np.arange(d1-1,-1,-1)/1/4)).sum()/np.exp(-np.arange(d1-1,-1,-1)/1/4).sum(), raw=True)
#     return y1.stack().swaplevel(0,1)

def _decay_linear(x1, d1):
    d1 = max(3, d1)
    y1 = x1.unstack(level=0).sort_index().rolling(d1).apply(lambda x: (x*np.arange(1,d1+1)).sum(), raw=True)
    return y1.stack().swaplevel(0,1)

def _decay_exp(x1, d1):
    d1 = max(3, d1)
    y1 = x1.unstack(level=0).sort_index().rolling(d1).apply(lambda x: (x*np.exp(-np.arange(d1-1,-1,-1)/1/4)).sum(), raw=True)
    return y1.stack().swaplevel(0,1)

# 返回值为向量，过去d天的最小值————时序函数
def _ts_min(X1,d1):
    d1=max(3,d1)
    y1=X1.groupby(level=0).rolling(d1,min_periods=1).min()
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

# 返回值为向量，过去d天的最小值————时序函数
def _ts_max(X1,d1):
    d1=max(3,d1)
    y1=X1.groupby(level=0).rolling(d1,min_periods=1).max()
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

# 注：数据量大的时候，速度很慢
# 返回值为向量，过去d天的最小值的位置————时序函数
# 使用了pandas自带的argmin和argmax
def _ts_argmin(X1,d1):
    d1=max(3,d1)
    y1=X1.groupby(level=0).rolling(d1,min_periods=1).agg(lambda x:x.argmin())
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

def _ts_argmax(X1,d1):
    d1=max(3,d1)
    y1=X1.groupby(level=0).rolling(d1,min_periods=1).agg(lambda x:x.argmax())
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

def _ts_sum(X1,d1):
    d1=max(3,d1)
    y1= X1.groupby(level=0).rolling(d1,min_periods=1).sum()
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

def _ts_product(X1,d1):
    d1=max(3,d1)
    y1=X1.groupby(level=0).rolling(d1,min_periods=1).apply(lambda x:x.product())
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

def _ts_stddev(X1,d1):
    d1=max(3,d1)
    y1=X1.groupby(level=0).rolling(d1,min_periods=1).apply(lambda x:x.std())
    y1=y1.fillna(0)
    y1=y1.droplevel(0)
    # y1.sort_index(level=1,inplace=True)
    return y1

def _norm(x1):
    return (x1 - x1.groupby(level=1).mean()) / x1.groupby(level=1).std()

def _fill(x1):
    return x1.fillna(0)

def _neu_cross(x1, y1):
    factor_df = x1.unstack(level=0).sort_index()
    zscores_f = pd.DataFrame(columns=factor_df.columns, index=factor_df.index, dtype=float)
    for idx in factor_df.index:
        response = factor_df.loc[idx]
        train = y1.unstack(level=0).loc[idx].unstack(level=0)

        if response.notnull().any() and train.notnull().any().any():
            response = response[response.notnull()]
            train = train[train.notnull().all(axis=1)]
            mask = response.index.intersection(train.index)
            if len(mask) > 10:
                train = train.loc[mask]
                train = train.loc[:, ~(train == 0).all()].values
                response = response[mask].values
                resid = _neu(train, response)
                zscores_f.loc[idx] = pd.Series(resid, index=mask)
    return zscores_f.stack().swaplevel(0,1)

def _neu(train, response):
    if train.shape[1] == 1:
        a_c = np.ones((response.shape[0], 1))
        train = np.hstack((a_c, train))
    train_trans = np.transpose(train)
    m = np.linalg.inv(np.dot(train_trans, train))
    param = np.dot(np.dot(m, train_trans), response)
    y_pred = np.dot(train, param)
    return response - y_pred

# fundamental functions
add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)
fillna0 = _Function(function=_fill, name='fill0', arity=1)

# cross-sectional functions
rank_cross = _Function(function=_rank_cross, name='rank', arity=1)
scale = _Function(function=_scale, name='scale', arity=1)
normalize = _Function(function=_norm, name='norm', arity=1)

# special functions: neutralize
neutralize = _Function(function=_neu_cross, name='neu', arity=1, assist=True)

# time-series functions
rank = _Function(function=_rank, name='rank_ts', arity=1, ts=True)
delay = _Function(function=_delay, name='delay', arity=1, ts=True)
ts_corr = _Function(function=_ts_corr, name='cor', arity=2, ts=True)
ts_cov = _Function(function=_ts_cov, name='cov', arity=2, ts=True)
delta = _Function(function=_delta, name='delta', arity=1, ts=True)
decay_linear = _Function(function=_decay_linear, name='decay_linear', arity=1, ts=True)
decay_exp = _Function(function=_decay_exp, name='decay_exp', arity=1, ts=True)
ts_min = _Function(function=_ts_min, name='ts_min', arity=1, ts=True)
ts_max = _Function(function=_ts_max, name='ts_max', arity=1, ts=True)
ts_argmin = _Function(function=_ts_argmin, name='ts_argmin', arity=1, ts=True)
ts_argmax = _Function(function=_ts_argmax, name='ts_argmax', arity=1, ts=True)
ts_sum = _Function(function=_ts_sum, name='ts_sum', arity=1, ts=True)
ts_product = _Function(function=_ts_product, name='ts_product', arity=1, ts=True)
ts_stddev = _Function(function=_ts_stddev, name='ts_stddev', arity=1, ts=True)









_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'sig': sig1,
                 'rkc': rank_cross,
                 'rnk': rank,
                 'sft': delay,
                 'cor': ts_corr,
                 'cov': ts_cov,
                 'scl': scale,
                 'dif': delta,
                 'dcl': decay_linear,
                 'dce': decay_exp,
                 'ts_min': ts_min,
                 'ts_max': ts_max,
                 'ts_argmin': ts_argmin,
                 'ts_argmax': ts_argmax,
                 'sum': ts_sum,
                 'product': ts_product,
                 'std': ts_stddev,
                 'norm': normalize,
                 'fill0': fillna0,
                 'neu': neutralize}
