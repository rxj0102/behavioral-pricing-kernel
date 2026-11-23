"""Post hoc robustness of the surviving finding to the physical density.

Not pre-registered. docs/limitations.md names the physical density as the joint
hypothesis behind every pricing kernel result; this asks whether the one finding that
survived Amendment 3 holds when that density changes. The finding: probability
weighting fits the gap between the published risk-neutral points and the physical
density better out of sample than CRRA, and the fitted curvature overweights tails
(alpha below 1) on most dates.

Variants (bpk.study.run.PHYSICAL_VARIANTS): the pre-specified GJR-GARCH filtered
historical simulation, the same with a zero or doubled equity premium, the same with
Gaussian shocks, and unfiltered historical simulation from the last five years. Same
dates, same published points, same walk-forward and Diebold-Mariano settings as the
main study, 6 month tenor only. Every row is reported and logged as a specification.

    python -m bpk.study.physical_robustness

writes results/real/physical_robustness.json and results/real/physical_robustness.md.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from bpk.data import real
from bpk.estimation import sdf
from bpk.study.run import (
    LOG,
    OUT,
    PHYSICAL_VARIANTS,
    build,
    diebold_mariano,
    fit_table,
    rounded,
)
from bpk.utils.results_log import log_spec

MODELS = ("none", "crra", "habit", "weight")
PHASE = "real_physical_robustness"


def one(data: real.RealData, variant: str, say: Any) -> dict[str, Any]:
    b = build(data, "sp6m", 5000, 1, physical=variant)
    fit, wf = fit_table(b, MODELS, say)
    alpha = sdf.date_curvature(b.panel, wf.theta["weight"][:, 0])
    return {
        "variant": variant,
        "description": PHYSICAL_VARIANTS[variant],
        "rmse_oos": {m: fit["models"][m]["rmse_oos"] for m in MODELS},
        "weight_vs_crra": diebold_mariano(wf.loss["crra"], wf.loss["weight"]),
        "weight_vs_habit": diebold_mariano(wf.loss["habit"], wf.loss["weight"]),
        "alpha_median": float(np.median(alpha)),
        "share_alpha_below_one": float((alpha < 1.0).mean()),
        "n_oos": fit["models"]["weight"]["n_oos"],
    }


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Robustness of the surviving finding to the physical density (post hoc)",
        "",
        "Not pre-registered. Written by `python -m bpk.study.physical_robustness`. Same dates, "
        "published points, walk-forward and Diebold-Mariano settings as the main study, 6 month "
        "tenor. RMSE is the out-of-sample error in percentage points of probability. The t "
        "statistics are Diebold-Mariano, positive when probability weighting has the smaller loss.",
        "",
        "| Physical density | No adjustment | CRRA | Habit | Weighting | Weighting vs CRRA t | "
        "Weighting vs habit t | Median alpha | Dates with alpha below 1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["rmse_oos"]
        lines.append(
            f"| {r['description']} | {100 * e['none']:.1f} | {100 * e['crra']:.1f} | "
            f"{100 * e['habit']:.1f} | {100 * e['weight']:.1f} | {r['weight_vs_crra']['t']:.1f} | "
            f"{r['weight_vs_habit']['t']:.1f} | {r['alpha_median']:.2f} | "
            f"{100 * r['share_alpha_below_one']:.0f}% |"
        )
    beats = [r for r in rows if r["weight_vs_crra"]["t"] > 1.96]
    below = [r for r in rows if r["share_alpha_below_one"] > 0.5]
    habit = [r for r in rows if r["weight_vs_habit"]["t"] > 1.96]
    lines += [
        "",
        f"Probability weighting beats CRRA at the 5% level under {len(beats)} of {len(rows)} "
        f"physical densities and overweights tails on most dates under {len(below)} of "
        f"{len(rows)}. It beats the habit-style benchmark at that level under {len(habit)} of "
        f"{len(rows)}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = real.load()

    def say(msg: str) -> None:
        print(msg, flush=True)

    rows = []
    for v in PHYSICAL_VARIANTS:
        say(f"== {v}")
        rows.append(rounded(one(data, v, say)))
    (OUT / "physical_robustness.json").write_text(json.dumps(rows, indent=1) + "\n")
    (OUT / "physical_robustness.md").write_text(render(rows))
    for r in rows:
        if r["variant"] == "fhs":
            continue  # the pre-specified density is already logged by bpk.study.run
        log_spec(
            LOG,
            phase=PHASE,
            spec_id=f"physical_{r['variant']}",
            status="ok",
            params={"physical": r["variant"], "tenor": "6m"},
            metrics={
                "rmse_oos_weight": r["rmse_oos"]["weight"],
                "rmse_oos_crra": r["rmse_oos"]["crra"],
            },
            note="post hoc robustness of the physical density, not pre-registered",
        )
    print((OUT / "physical_robustness.md").read_text())


if __name__ == "__main__":
    main()
