# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University

"""Test suite for `pyRing.utils`
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from .. import utils as pyring_utils

#
#@mock.patch.dict("os.environ")
#def test_set_prefix(tmp_path):
#    # make sure PYRING_PREFIX isn't set
#    os.environ.pop("PYRING_PREFIX", None)
#    os.environ["HOME"] = str(tmp_path)
#    # check that we get a warning
#    with pytest.warns(UserWarning):
#        assert pyring_utils.set_prefix() == str(tmp_path / "src" / "pyring")
#
#
#@mock.patch.dict("os.environ")
#def test_set_prefix_environ():
#    os.environ["PYRING_PREFIX"] = "test"
#    assert pyring_utils.set_prefix() == "test"
