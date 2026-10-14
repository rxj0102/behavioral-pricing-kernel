"""Build the cleaned dataset and the cleaning statistics table.

Usage: python -m bpk.data.build [--config configs/fast.yaml]
Writes cleaned quotes (parquet, git-ignored) and cleaning_stats.csv
(committed) under the config paths. Synthetic runs are labeled in the file
name so they cannot be mistaken for empirical data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bpk.config import Config, load_config
from bpk.data.base import DataAdapter, DataUnavailableError, SeriesSpec
from bpk.data.cache import CachedSource
from bpk.data.clean import clean_panel
from bpk.data.manifest import Manifest
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.data.wrds_adapter import WrdsAdapter

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
    """Fetch (cached), clean, write outputs. Returns (quotes path, stats path)."""
    adapter = make_adapter(cfg)
    src = CachedSource(adapter, cfg.data.cache_dir, Manifest(cfg.data.manifest_path))
    panel = src.get(SPEC, cfg.data.start, cfg.data.end)
    result = clean_panel(panel, cfg.cleaning)
    tag = adapter.source_name
    cfg.data.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    quotes_path = cfg.data.cache_dir / f"clean_quotes__{tag}.parquet"
    stats_path = cfg.tables_dir / f"cleaning_stats__{tag}.csv"
    result.quotes.to_parquet(quotes_path)
    stats = result.stats.copy()
    stats["share_of_raw"] = stats["share_of_raw"].round(6)
    stats.to_csv(stats_path, index=False)
    raw = len(panel)
    print(f"source={tag} raw_rows={raw} clean_rows={len(result.quotes)}")
    print(pd.DataFrame(result.stats).to_string(index=False))
    return quotes_path, stats_path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
