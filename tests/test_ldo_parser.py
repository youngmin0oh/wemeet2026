"""The LDO result parser must be safe to run for several corners at once.

NgSpiceWrapper.run() parses all corners in parallel threads on one CircuitClass
object. The parser used to keep intermediate values on the instance, so one
corner could report another corner's PSRR or phase margin.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from eval_engines.ngspice import CircuitClass as circuit_module
from eval_engines.ngspice.CircuitClass import CircuitClass

FREQ = np.logspace(0, 9, 91)          # 10 points per decade, like the testbench


def write_corner(folder, k):
    """Fake wrdata files whose values depend on the corner number k."""
    folder.mkdir()
    vin = np.arange(1.0, 3.0, 0.01)
    vout = np.minimum(vin - 0.1 - 0.01 * k, 1.8)
    np.savetxt(folder / "ldo_tb_dc.txt", np.c_[vin, vout])
    for name, level in (("psrr_maxload", 0.1), ("psrr_minload", 0.05)):
        mag = np.full_like(FREQ, level * (k + 1))
        np.savetxt(folder / f"ldo_tb_{name}.txt", np.c_[FREQ, mag, FREQ, np.zeros_like(FREQ)])
    for name in ("loop_gain_maxload", "loop_gain_minload"):
        mag = 1e3 / (1 + FREQ / 1e3)                        # crosses 0 dB near 1 MHz
        phase = np.full_like(FREQ, -120.0 + k)              # phase margin = 60 + k degrees
        np.savetxt(folder / f"ldo_tb_{name}.txt", np.c_[FREQ, mag, FREQ, phase])


def test_parallel_parsing_keeps_each_corner_separate(tmp_path, monkeypatch):
    corners = 9
    folders = []
    for k in range(corners):
        write_corner(tmp_path / f"corner{k}", k)
        folders.append(tmp_path / f"corner{k}")

    parser = CircuitClass.__new__(CircuitClass)
    parser.comparator, parser.ldo = False, True

    # Widen the race window: give up the GIL inside every file read.
    real_ac = circuit_module.ldo_ac

    def slow_ac(output_path, file_name):
        import time
        time.sleep(0.001)
        return real_ac(output_path, file_name)

    monkeypatch.setattr(circuit_module, "ldo_ac", slow_ac)
    expected = [parser.translate_result(str(f)) for f in folders]
    for repeat in range(5):
        with ThreadPoolExecutor(corners) as pool:
            results = list(pool.map(lambda f: parser.translate_result(str(f)), folders))
        assert results == expected
    for k, result in enumerate(expected):
        assert result["phm_maxload"] == 60 + k
        assert np.isclose(result["mPSRR_maxload_worst_10kHz"], -20 * np.log10(0.1 * (k + 1)))
