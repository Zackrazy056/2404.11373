import importlib

import numpy as np
import pytest

from rd_sbi.detector.patterns import (
    antenna_pattern,
    gmst_from_gps,
    h1_geometry,
    l1_geometry,
    time_delay_from_geocenter_s,
)


def _reference_backend():
    bilby_spec = importlib.util.find_spec("bilby")
    if bilby_spec is None:
        pytest.skip("No external detector-response reference backend is installed")

    from bilby.gw.detector import InterferometerList

    ifos = {ifo.name: ifo for ifo in InterferometerList(["H1", "L1"])}

    def evaluate(detector_name: str, gps: float, ra: float, dec: float, psi: float) -> tuple[float, float, float]:
        ifo = ifos[detector_name]
        f_plus = float(ifo.antenna_response(ra, dec, gps, psi, mode="plus"))
        f_cross = float(ifo.antenna_response(ra, dec, gps, psi, mode="cross"))
        delay = float(ifo.time_delay_from_geocenter(ra, dec, gps))
        return f_plus, f_cross, delay

    return "bilby", evaluate


def test_detector_response_matches_external_reference() -> None:
    backend_name, reference = _reference_backend()

    gps = 1126259462.42323
    ra = 1.95
    dec = -1.27
    psi = 0.82
    gmst = gmst_from_gps(gps)

    for geometry in (h1_geometry(), l1_geometry()):
        ref_f_plus, ref_f_cross, ref_delay = reference(geometry.name, gps, ra, dec, psi)
        local_f_plus, local_f_cross = antenna_pattern(geometry, ra, dec, psi, gmst)
        local_delay = time_delay_from_geocenter_s(geometry, ra, dec, gmst)

        assert local_f_plus == pytest.approx(ref_f_plus, abs=5e-4), backend_name
        assert local_f_cross == pytest.approx(ref_f_cross, abs=5e-4), backend_name
        assert local_delay == pytest.approx(ref_delay, abs=5e-6), backend_name

        assert np.isfinite([local_f_plus, local_f_cross, local_delay]).all()
