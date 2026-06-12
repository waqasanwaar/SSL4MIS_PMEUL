# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 15:24:28 2026

@author: 
"""
from typing import List, cast
from torch import Tensor, einsum
import torch
import numpy as np
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
