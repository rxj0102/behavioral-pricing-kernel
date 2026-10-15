"""Build the cleaned dataset and the cleaning statistics table.

Usage: python -m bpk.data.build [--config configs/fast.yaml]
Writes cleaned quotes (parquet, git-ignored) and cleaning_stats.csv
(committed) under the config paths. Synthetic runs are labeled in the file
name so they cannot be mistaken for empirical data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bpk.config import Config, load_config, result_hash
from bpk.data.base import DataAdapter, DataUnavailableError, SeriesSpec
from bpk.data.cache import CachedSource
from bpk.data.clean import clean_panel
from bpk.data.manifest import Manifest
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.data.wrds_adapter import WrdsAdapter
from bpk.utils.stats_io import write_stats

SPEC = SeriesSpec("spx_options", "panel")


def make_adapter(cfg: Config) -> DataAdapter:
    """Select the adapter named by data.source."""
    if cfg.data.source == "synthetic":
        return SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    if cfg.data.source == "wrds":
        return WrdsAdapter()
    raise DataUnavailableError(
        "cboe is a snapshot-only source; use the collector, not the batch build"
    )


def build(cfg: Config) -> tuple[Path, Path]:
    """Fetch (cached), clean, write outputs. Returns (quotes path, stats path).

    Every stats file carries the config hash and the requested and observed date range.
    """
    adapter = make_adapter(cfg)
    src = CachedSource(adapter, cfg.data.cache_dir, Manifest(cfg.data.manifest_path))
    panel = src.get(SPEC, cfg.data.start, cfg.data.end)
    result = clean_panel(panel, cfg.cleaning)
    tag = adapter.source_name
    cfg.data.cache_dir.mkdir(parents=True, exist_ok=True)
    quotes_path = cfg.data.cache_dir / f"clean_quotes__{tag}.parquet"
    result.quotes.to_parquet(quotes_path)
    meta = {
        "source": tag,
        "config_hash": result_hash(cfg),
        "requested_start": cfg.data.start.isoformat(),
        "requested_end": cfg.data.end.isoformat(),
        "observed_start": pd.Timestamp(panel["date"].min()).date().isoformat(),
        "observed_end": pd.Timestamp(panel["date"].max()).date().isoformat(),
        "raw_rows": len(panel),
        "clean_rows": len(result.quotes),
    }
    stats_path = cfg.tables_dir / f"cleaning_stats__{tag}.csv"
    write_stats(result.stats, stats_path, meta)
    write_stats(
        result.stats_by_moneyness,
        cfg.tables_dir / f"cleaning_stats_by_moneyness__{tag}.csv",
        {**meta, "bucket_variable": "ln(strike / spot)"},
    )
    if isinstance(adapter, SyntheticHestonAdapter):
        params = {**adapter.truth_params(), "config_hash": meta["config_hash"]}
        (cfg.tables_dir / f"synthetic_truth_params__{tag}.json").write_text(
            json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"source={tag} raw_rows={len(panel)} clean_rows={len(result.quotes)}")
    print(result.stats.to_string(index=False))
    return quotes_path, stats_path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
