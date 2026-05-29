"""Tests for the eval_network.py verdict helpers."""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from eval_network import compute_vs_native_verdict  # noqa: E402


def test_v2_beats_native_on_both_metrics():
    """When v2 throughput > native and v2 wait < native on all seeds,
    the verdict should report +pct on throughput and +pct (lower) on wait
    with full seed wins."""
    v2_runs = [
        {"arrived": 2200, "wait_per_vehicle": 5000.0},
        {"arrived": 2300, "wait_per_vehicle": 4800.0},
        {"arrived": 2150, "wait_per_vehicle": 5100.0},
    ]
    nat_runs = [
        {"arrived": 2046, "wait_per_vehicle": 5938.0},
        {"arrived": 2076, "wait_per_vehicle": 5503.0},
        {"arrived": 1966, "wait_per_vehicle": 7113.0},
    ]
    v = compute_vs_native_verdict(v2_runs, nat_runs)
    assert v["throughput_pct"] > 0
    assert v["wait_pct"] > 0  # positive = lower wait
    assert v["throughput_wins"] == 3
    assert v["wait_wins"] == 3
    assert v["n_seeds"] == 3


def test_v2_loses_to_native():
    """V2 numbers from the last run (3-seed eval). Confirms the verdict
    correctly reports negative percentages and 0 wins."""
    v2_runs = [
        {"arrived": 1632, "wait_per_vehicle": 7534.59},
        {"arrived": 1558, "wait_per_vehicle": 10446.99},
        {"arrived": 1726, "wait_per_vehicle": 5089.16},
    ]
    nat_runs = [
        {"arrived": 2095, "wait_per_vehicle": 5196.23},
        {"arrived": 2076, "wait_per_vehicle": 5503.70},
        {"arrived": 1966, "wait_per_vehicle": 7113.79},
    ]
    v = compute_vs_native_verdict(v2_runs, nat_runs)
    assert v["throughput_pct"] < 0          # v2 throughput < native
    assert v["wait_pct"] < 0                # v2 wait > native (so pct lower < 0)
    assert v["throughput_wins"] == 0
    # one seed (seed 3) has v2 wait < native wait
    assert v["wait_wins"] == 1


if __name__ == "__main__":
    test_v2_beats_native_on_both_metrics()
    test_v2_loses_to_native()
    print("eval_network tests OK")
