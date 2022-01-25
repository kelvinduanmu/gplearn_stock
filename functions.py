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

    def __init__(self, function, name, arity, ts=False):
        self.function = function
        self.name = name
        self.arity = arity
        self.ts = ts

    def __call__(self, *args):
        args2 = []
        for ele in args:
            if ele.index.nlevels != 2:
                print("Input should have 2 levels of index")
                raise
            args2.append(ele.sort_index())
        if self.ts:
            args2.append(self.d1)
        return self.function(*args2)

    def set_d1(self, d1):
        self.d1 = d1


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
        result[code] = comb_data.loc[code].unstack(level=0).rolling(d1).corr().drop(comb_data.columns.levels[0][0], level=1).droplevel(1)[comb_data.columns.levels[0][0]]
    return result.stack().swaplevel(0,1)

def _ts_cov(x1, y1, d1):
    d1 = max(3, d1)
    comb_data = pd.concat([x1, y1], axis=1).unstack()
    result = pd.Series(index=x1.index).unstack(level=0)
    for code in comb_data.index:
        result[code] = comb_data.loc[code].unstack(level=0).rolling(d1).cov().drop(comb_data.columns.levels[0][0], level=1).droplevel(1)[comb_data.columns.levels[0][0]]
    return result.stack().swaplevel(0,1)

def _scale(x1):
    return x1 / x1.abs().groupby(level=1).sum()

def _delta(x1, d1):
    y1 = x1.groupby(level=0).diff(d1)
    return y1

def _decay_linear(x1, d1):
    d1 = max(3, d1)
    y1 = x1.unstack(level=0).rolling(d1).apply(lambda x: (x*np.arange(1,d1+1)).sum()/np.arange(1, d1+1).sum(), raw=True)
    return result.stack().swaplevel(0,1)

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

# cross-sectional functions
rank_cross = _Function(function=_rank_cross, name='rank', arity=1)
scale = _Function(function=_scale, name='scale', arity=1)

# time-series functions
rank = _Function(function=_rank, name='rank_ts', arity=1, ts=True)
delay = _Function(function=_delay, name='delay', arity=1, ts=True)
ts_corr = _Function(function=_ts_corr, name='cor', arity=2, ts=True)
ts_cov = _Function(function=_ts_cov, name='cov', arity=2, ts=True)
delta = _Function(function=_delta, name='delta', arity=1, ts=True)
decay_linear = _Function(function=_decay_linear, name='decay_linear', arity=1, ts=True)

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
                 'dcl': decay_linear}
