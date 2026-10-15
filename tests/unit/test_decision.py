"""The pre-registered fit-variant decision rule, on toy raw tables with a known answer."""

import numpy as np
import pandas as pd

from bpk.rnd.decision import SPECS, boot_mean_p90, decide, final_default


def _raw(l1: dict[str, float], fail: dict[str, float], n: int = 60, n_mat: int = 5) -> pd.DataFrame:
    """Toy raw table: per variant a constant-ish L1, and a share of dates with no horizon."""
    rng = np.random.default_rng(0)
    rows = []
    for spec in SPECS:
        for v in ("legacy", "robust", "robust_monotone"):
            for i in range(n):
                d = f"d{i}"
                failed = i < int(fail[v] * n)
                base = {"spec": spec, "variant": v, "date": d}
                rows.append({**base, "n_slices": 0 if failed else n_mat, "horizon_ok": not failed})
                if failed:
                    continue
                rows.append(
                    {
                        **base,
                        "interpolated": True,
                        "dte": 30,
                        "l1_total": l1[v] * (1 + 0.1 * rng.standard_normal()),
                    }
                )
                for dte in (21, 45, 80, 120, 180):
                    rows.append({**base, "interpolated": False, "dte": dte, "l1_total": 0.01})
    return pd.DataFrame(rows)


def test_lowest_objective_wins_when_coverage_ok() -> None:
    raw = _raw(
        {"legacy": 0.10, "robust": 0.05, "robust_monotone": 0.03},
        {"legacy": 0.0, "robust": 0.0, "robust_monotone": 0.0},
    )
    assert decide(raw, 1).winner == "robust_monotone"


def test_coverage_drop_over_two_points_makes_variant_ineligible() -> None:
    # robust_monotone has the best L1 on fitted dates but loses 5 percent of dates (counted at 2.0)
    raw = _raw(
        {"legacy": 0.10, "robust": 0.05, "robust_monotone": 0.03},
        {"legacy": 0.0, "robust": 0.0, "robust_monotone": 0.05},
    )
    d = decide(raw, 1)
    assert not bool(d.table.set_index("variant").loc["robust_monotone", "eligible"])
    assert d.winner == "robust"


def test_failures_counted_at_worst_value_can_flip_the_ranking() -> None:
    # A 12 percent failure share pushes the 90th percentile to the worst value 2.0.
    raw = _raw(
        {"legacy": 0.10, "robust": 0.05, "robust_monotone": 0.01},
        {"legacy": 0.12, "robust": 0.0, "robust_monotone": 0.0},
    )
    d = decide(raw, 1)
    t = d.table.set_index("variant")
    assert float(t["objective"].to_numpy()[0]) > 1.0  # failures dominate
    assert d.winner == "robust_monotone"


def test_exact_tie_goes_to_simpler_variant() -> None:
    raw = _raw(
        {"legacy": 0.05, "robust": 0.05, "robust_monotone": 0.05},
        {"legacy": 0.0, "robust": 0.0, "robust_monotone": 0.0},
    )
    raw["l1_total"] = raw["l1_total"].where(raw["interpolated"] != True, 0.05)  # noqa: E712
    assert decide(raw, 1).winner == "legacy"


def test_confirmation_logic() -> None:
    raw1 = _raw(
        {"legacy": 0.10, "robust": 0.05, "robust_monotone": 0.03},
        {"legacy": 0.0, "robust": 0.0, "robust_monotone": 0.0},
    )
    raw2 = _raw(
        {"legacy": 0.10, "robust": 0.02, "robust_monotone": 0.03},
        {"legacy": 0.0, "robust": 0.0, "robust_monotone": 0.0},
    )
    a, b = decide(raw1, 1), decide(raw1, 2)
    assert final_default(a, None)[0] == "robust_monotone"
    assert final_default(a, b) == ("robust_monotone", "confirmed")
    default, status = final_default(a, decide(raw2, 1))
    assert default == "legacy" and "not confirmed" in status


def test_bootstrap_is_deterministic() -> None:
    x = np.random.default_rng(3).random(50)
    r1 = boot_mean_p90(x, np.random.default_rng(1))
    r2 = boot_mean_p90(x, np.random.default_rng(1))
    assert r1 == r2 and 0 < r1 <= 1


def test_fallback_diagnostic_separates_convex_bracket_dates() -> None:
    from bpk.rnd.decision import fallback_diagnostic

    rows = []
    for i, conv in enumerate([False, False, True]):
        for dte in (21, 45):
            rows.append(
                {
                    "spec": "s",
                    "variant": "legacy",
                    "date": f"d{i}",
                    "interpolated": False,
                    "dte": dte,
                    "method": "convex" if (conv and dte == 21) else "svi",
                    "l1_total": 1.5 if (conv and dte == 21) else 0.01,
                }
            )
        rows.append(
            {
                "spec": "s",
                "variant": "legacy",
                "date": f"d{i}",
                "interpolated": True,
                "dte": 30,
                "method": "interp",
                "l1_total": 1.2 if conv else 0.02,
            }
        )
    t = fallback_diagnostic(pd.DataFrame(rows))
    nat = t[t["block"] == "native_by_method"].set_index("group")
    assert nat.loc["convex", "n"] == 1 and nat.loc["convex", "l1_median"] == 1.5
    h = t[t["block"] == "30d_by_bracket_convex"].set_index("group")
    assert h.loc["True", "l1_median"] == 1.2 and h.loc["False", "l1_median"] == 0.02
