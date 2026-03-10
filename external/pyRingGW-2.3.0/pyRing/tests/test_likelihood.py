# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University

"""Test suite for `pyRing.likelihood`
"""

import numpy
from numpy.testing import assert_array_equal

from .. import likelihood


def test_inner_product():
    a = numpy.asarray((0.5, 0.5, 0.))
    b = numpy.asarray([  # identity
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ], dtype=float)
    assert likelihood.residuals_inner_product_direct_inversion(a, b) == .5
