"""Free real-data sources for the study (Amendment 3) and their frozen snapshot.

Sources, none of which needs a key or a login:
  Minneapolis Fed market-based probability statistics (S&P 500, 6 and 12 month)
  Ken French Data Library daily factors (market excess return, bill rate)
  Baker-Wurgler investor sentiment (Jeffrey Wurgler's page)
  CBOE VIX from FRED (VIXCLS)

refresh() downloads everything and rewrites data/real. load() reads the snapshot only, so
results do not change when a source revises its history.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SNAPSHOT = Path(__file__).resolve().parents[3] / "data" / "real"
URLS = {
    "mpd": "https://www.minneapolisfed.org/-/media/files/banking/mpd/mpd_stats.csv",
    "french": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip",
    "sentiment": "https://pages.stern.nyu.edu/~jwurgler/data/SENTIMENT.xlsx",
    "vix": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
}
HEADERS = {"User-Agent": "curl/8.5.0", "Accept": "*/*"}
MPD_COLUMNS = ["mu", "sd", "skew", "kurt", "p10", "p50", "p90", "prDec", "prInc"]
MPD_MARKETS = {"sp6m": 6, "sp12m": 12}


@dataclass(frozen=True)
class RealData:
    """mpd: one row per (market, date). market: daily mkt_rf and rf. sentiment: monthly."""

    mpd: pd.DataFrame
    market: pd.DataFrame
    sentiment: pd.Series
    vix: pd.Series


def _get(url: str, tries: int = 5) -> bytes:
    req = Request(url, headers=HEADERS)
    for k in range(tries):
        try:
            with urlopen(req, timeout=120) as resp:
                return bytes(resp.read())
        except OSError:  # transient resets are common on these hosts
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise RuntimeError("unreachable")


def parse_mpd(raw: bytes) -> pd.DataFrame:
    """Keep the S&P 500 rows whose maturity target matches the series name."""
    df = pd.read_csv(io.BytesIO(raw), skiprows=3)
    df = df[df["market"].isin(list(MPD_MARKETS))].copy()
    df = df[df["maturity_target"] == df["market"].map(MPD_MARKETS)]
    df["date"] = pd.to_datetime(df["idt"], format="%m/%d/%Y")
    out = df[["market", "date", *MPD_COLUMNS]].sort_values(["market", "date"])
    return out.reset_index(drop=True)


def parse_french(raw: bytes) -> pd.DataFrame:
    """Daily market excess return and bill rate as decimals."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append((parts[0], float(parts[1]) / 100.0, float(parts[4]) / 100.0))
    out = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    return out


def parse_sentiment(raw: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(raw), sheet_name="DATA")
    out = df[["yearmo", "SENT", "SENT_ORTH"]].dropna()
    return out.astype({"yearmo": int}).reset_index(drop=True)


def parse_vix(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), na_values=".")
    df.columns = ["date", "vix"]
    return df.dropna().reset_index(drop=True)


def refresh(snapshot: Path = SNAPSHOT) -> dict[str, dict[str, str | int]]:
    """Download every source, write the snapshot CSVs and a manifest with their hashes."""
    snapshot.mkdir(parents=True, exist_ok=True)
    frames = {
        "mpd": parse_mpd(_get(URLS["mpd"])),
        "french": parse_french(_get(URLS["french"])),
        "sentiment": parse_sentiment(_get(URLS["sentiment"])),
        "vix": parse_vix(_get(URLS["vix"])),
    }
    manifest: dict[str, dict[str, str | int]] = {}
    for name, df in frames.items():
        path = snapshot / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.10g", date_format="%Y-%m-%d")
        first = df.iloc[0, 1 if name == "mpd" else 0]
        last = df.iloc[-1, 1 if name == "mpd" else 0]
        manifest[name] = {
            "url": URLS[name],
            "rows": len(df),
            "first": str(first)[:10],
            "last": str(last)[:10],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(snapshot: Path = SNAPSHOT) -> RealData:
    mpd = pd.read_csv(snapshot / "mpd.csv", parse_dates=["date"])
    market = pd.read_csv(snapshot / "french.csv", parse_dates=["date"]).set_index("date")
    sent = pd.read_csv(snapshot / "sentiment.csv")
    idx = pd.PeriodIndex(sent["yearmo"].astype(str), freq="M")
    sentiment = pd.Series(sent["SENT_ORTH"].to_numpy(), index=idx, name="sentiment")
    vix = pd.read_csv(snapshot / "vix.csv", parse_dates=["date"]).set_index("date")["vix"]
    return RealData(mpd=mpd, market=market, sentiment=sentiment, vix=vix)


if __name__ == "__main__":
    print(json.dumps(refresh(), indent=2))
