# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University

"""Test suite for `pyRing.waveform`
"""

import pytest

from .. import waveform


class TestTEOBPM:
    WAVEFORM = waveform.TEOBPM

    @classmethod
    @pytest.fixture
    def smwaveform(cls):
        return cls.WAVEFORM(
            0.,             # t0
            10.,            # m1
            10.,            # m2
            0.,             # chi1
            0.,             # chi2
            {},             # phases
            1000,           # distance
            0.,             # iota
            0.,             # phi
            [(2,2), (3,3)], # modes
            {},             # TGR params
        )

    @pytest.mark.parametrize(("l", "m", "a1"), (
        (2, 2, pytest.approx(1.7666767219721071)),
    ))
    def test_EOBPM_SetupFitCoefficients(self, smwaveform, l, m, a1):
        coeffs = smwaveform.EOBPM_SetupFitCoefficients(l, m)
        assert coeffs["a1"] == a1
