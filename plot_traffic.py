#!/usr/bin/env python3
"""
Plot 24 hours of mobile internet traffic for 5 squares of Milan
(Telecom Italia Big Data Challenge dataset, Nov-Dec 2013).

How to run (from your ran-energy-saver folder):
    python plot_traffic.py data/sms-call-internet-mi-2013-11-04_parsed.tar.gz

Choose your own squares instead of the automatic five:
    python plot_traffic.py data/sms-call-internet-mi-2013-11-04_parsed.tar.gz --squares 5161 4259 1150 8000 30

It reads the O-RAN SC "parsed" files (SquareID, Timestamp, InternetTraffic),
the original Harvard Dataverse files (8 tab-separated columns, no header),
a .tar.gz download as it is, an extracted file, or a folder of files.

Needs: python -m pip install pandas matplotlib
"""
import argparse
import io
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_SUFFIXES = {".txt", ".csv", ".tsv", ".dat", ""}


# ---------- reading ----------
def iter_sources(path: Path):
    """Yield (name, bytes) for every data file found at path."""
    if path.is_dir():
        for f in sorted(path.iterdir()):
            if f.is_file() and not f.name.startswith(".") and (f.suffix.lower() in DATA_SUFFIXES or f.name.endswith(".tar.gz")):
                yield from iter_sources(f)
    elif path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as tar:
            for m in tar.getmembers():
                name = Path(m.name).name
                if m.isfile() and not name.startswith(".") and Path(name).suffix.lower() in DATA_SUFFIXES:
                    yield m.name, tar.extractfile(m).read()
    elif path.is_file():
        yield path.name, path.read_bytes()
    else:
        sys.exit(f"Can't find {path}. Check the file name and that it is in the data folder.")


def first_line(raw: bytes) -> str:
    for line in raw[:65536].decode("utf-8", errors="replace").splitlines():
        if line.strip():
            return line
    return ""


def read_one(name: str, raw: bytes) -> pd.DataFrame:
    line = first_line(raw)
    if not line:
        return pd.DataFrame(columns=["square", "time", "internet"])
    if "\t" in line:
        sep = "\t"
    elif "," in line:
        sep = ","
    elif ";" in line:
        sep = ";"
    else:
        sep = r"\s+"
    tokens = [t.strip().strip('"') for t in (re.split(sep, line.strip()) if sep == r"\s+" else line.split(sep))]
    has_header = any(re.search(r"[A-Za-z]", t) for t in tokens)

    if has_header:
        low = [t.lower() for t in tokens]
        def find(*keys, default):
            for i, c in enumerate(low):
                if any(k in c for k in keys):
                    return i
            return default
        cols = (find("square", "cell", "grid", default=0),
                find("time", "date", "interval", default=1),
                find("internet", "traffic", default=len(tokens) - 1))
    else:
        cols = (0, 1, 7) if len(tokens) == 8 else (0, 1, len(tokens) - 1)

    df = pd.read_csv(io.BytesIO(raw), sep=sep, header=0 if has_header else None,
                     usecols=list(cols), engine="python" if sep == r"\s+" else "c")
    df = df.iloc[:, [list(sorted(cols)).index(c) for c in cols]]
    df.columns = ["square", "time", "internet"]
    print(f"  read {name}: {len(df):,} rows")
    return df


def to_local_time(t: pd.Series) -> pd.Series:
    """Return Milan local time without a timezone label."""
    num = pd.to_numeric(t, errors="coerce")
    if num.notna().mean() > 0.95:
        unit = "ms" if num.max() > 1e11 else "s"
        return pd.to_datetime(num, unit=unit, utc=True).dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    ts = pd.to_datetime(t, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    return ts  # the parsed files already use Milan local time


# ---------- choosing squares ----------
def pick_squares(totals: pd.Series, n: int = 5):
    """Busiest, 90th, 50th, 25th and 5th percentile squares by daily traffic."""
    totals = totals[totals > 0].sort_values()
    plan = [(1.00, "Busiest"), (0.90, "Busy (90th percentile)"), (0.50, "Median"),
            (0.25, "Quiet (25th percentile)"), (0.05, "Very quiet (5th percentile)")][:n]
    chosen, used = [], set()
    for q, label in plan:
        target = totals.quantile(q)
        for sq in (totals - target).abs().sort_values().index:
            if sq not in used:
                used.add(sq)
                chosen.append((sq, label))
                break
    return chosen


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot 24 h of internet traffic for 5 Milan squares.")
    ap.add_argument("path", type=Path, help="a .tar.gz download, an extracted file, or a folder of them")
    ap.add_argument("--squares", type=int, nargs="+", help="square IDs to plot (1-10000) instead of the automatic five")
    ap.add_argument("--out", type=Path, default=Path("traffic_24h.png"), help="where to save the chart")
    ap.add_argument("--no-show", action="store_true", help="save the chart without opening a window")
    args = ap.parse_args()

    print(f"Reading {args.path} ...")
    frames = [read_one(n, b) for n, b in iter_sources(args.path)]
    if not frames:
        sys.exit("No data files found there.")
    df = pd.concat(frames, ignore_index=True)

    df["square"] = pd.to_numeric(df["square"], errors="coerce")
    df["internet"] = pd.to_numeric(df["internet"], errors="coerce").fillna(0)
    df["ts"] = to_local_time(df["time"])
    df = df.dropna(subset=["square", "ts"])
    df["square"] = df["square"].astype(int)

    # keep the first 24 hours, then add up all country codes per square and 10-minute slot
    start = df["ts"].min()
    df = df[df["ts"] < start + pd.Timedelta(hours=24)]
    series = df.groupby(["square", "ts"])["internet"].sum().unstack("square").sort_index()

    if args.squares:
        missing = [s for s in args.squares if s not in series.columns]
        if missing:
            sys.exit(f"These squares have no data in the file: {missing}")
        chosen = [(s, "Chosen") for s in args.squares]
    else:
        chosen = pick_squares(series.sum())

    # chart: top = activity (log scale, so quiet squares stay visible); bottom = share of each square's own peak
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    for sq, label in chosen:
        s = series[sq].fillna(0)
        line, = top.plot(series.index, s.clip(lower=1e-3), linewidth=1.8)
        bottom.plot(series.index, 100 * s / s.max() if s.max() > 0 else s, linewidth=1.6, color=line.get_color(),
                    label=f"{label}: square {sq}")
    top.set_yscale("log")
    top.set_title(f"Mobile internet activity over 24 hours, Milan, {start:%a %d %b %Y}", fontsize=13, loc="left")
    top.set_ylabel("Internet activity\n(relative units, log scale)")
    bottom.legend(frameon=False, fontsize=9, loc="upper left")  # night hours sit low, so this corner stays clear
    bottom.axhline(30, color="grey", linestyle="--", linewidth=1)
    bottom.text(series.index[2], 32, "30% of peak: a possible sleep threshold", fontsize=9, color="dimgrey")
    bottom.set_ylim(0, 105)
    bottom.set_ylabel("% of the square's own peak")
    bottom.set_xlabel("Milan local time")
    bottom.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    # what the chart says, for your notes
    print(f"\nSaved chart: {args.out.resolve()}")
    print(f"{len(series.columns):,} squares, {len(series):,} ten-minute slots from {start:%d %b %Y %H:%M}\n")
    print(f"{'Square':>7}  {'Profile':<28}{'Peak':>7}{'Quietest':>10}{'Peak/quiet':>12}{'Hours < 30% of peak':>21}")
    for sq, label in chosen:
        s = series[sq].fillna(0)
        peak, low = s.max(), s.min()
        ratio = f"{peak / low:.1f}x" if low > 0 else "n/a"
        low_hours = (s < 0.3 * peak).sum() * 10 / 60
        peak_at, low_at = s.idxmax().strftime("%H:%M"), s.idxmin().strftime("%H:%M")
        print(f"{sq:>7}  {label:<28}{peak_at:>7}{low_at:>10}{ratio:>12}{low_hours:>19.1f} h")
    city = series.sum(axis=1)
    print(f"\nQuietest time across the whole city: {city.idxmin():%H:%M} (usually around 03:00-05:00).")
    print("Long stretches below 30% of peak are where a capacity cell could sleep to save energy.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
