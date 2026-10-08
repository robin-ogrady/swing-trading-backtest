"""
download_market_data.py
Downloads free daily (1d) and 4-hour (4h) OHLCV data for:
  - S&P 500 stocks   (Yahoo Finance via yfinance)
  - Forex pairs      (Yahoo Finance via yfinance)
  - Crypto USDT pairs (Binance public data dump, data.binance.vision, no key, no rate limit)

Output layout (parquet, one file per symbol):
  data/
    stocks/1d/AAPL.parquet     stocks/4h/AAPL.parquet
    forex/1d/EURUSD.parquet    forex/4h/EURUSD.parquet
    crypto/1d/BTCUSDT.parquet  crypto/4h/BTCUSDT.parquet
    index.csv                  (one row per file: asset class, symbol, timeframe, rows, start, end)

Every file has the same columns: timestamp (UTC index), open, high, low, close, volume

Install:  pip install yfinance pandas pyarrow requests lxml
Run:      python download_market_data.py                 (everything)
          python download_market_data.py --only stocks --tf 1d   (daily stocks only)
          python download_market_data.py --only crypto   (one asset class)
          python download_market_data.py --limit 5       (test run, 5 symbols per class)
Re-running skips files that already exist. Use --refresh to redo them.

Known limits (free data):
  - Stocks and forex 4h bars: Yahoo only serves ~2 years of 1h data, so 4h is built
    from that (about 2 years). Daily goes back as far as Yahoo has it.
  - Crypto has full history for both timeframes.
  - Stock prices are split/dividend adjusted. Forex volume from Yahoo is always 0.
  - Stock list is today's S&P 500, so delisted companies are missing (survivorship bias).
"""
import argparse
import io
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = Path("data")
COLS = ["open", "high", "low", "close", "volume"]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (market-data-downloader)"
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))


# ----------------------------------------------------------------- helpers
def save(df, asset, tf, symbol, index_rows):
    if df is None or df.empty:
        return
    df = df[COLS].dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        return
    d = OUT / asset / tf
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{symbol}.parquet")
    index_rows.append({"asset_class": asset, "symbol": symbol, "timeframe": tf,
                       "rows": len(df), "start": df.index[0], "end": df.index[-1]})


def exists(asset, tf, symbol):
    return (OUT / asset / tf / f"{symbol}.parquet").exists()


def write_index(rows):
    if not rows:
        return
    path = OUT / "index.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates(
            ["asset_class", "symbol", "timeframe"], keep="last")
    OUT.mkdir(exist_ok=True)
    new.sort_values(["asset_class", "symbol", "timeframe"]).to_csv(path, index=False)


def clean_yahoo(df):
    """Normalize a yfinance frame to lowercase OHLCV with a UTC index."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df:
        df["volume"] = 0
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df[COLS]


def resample_4h(df, session_tz=None):
    """Build 4h bars from 1h bars. For stocks, bins start at 9:30 and 13:30 New York time."""
    if df is None or df.empty:
        return None
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if session_tz:
        x = df.tz_convert(session_tz)
        out = x.resample("4h", offset="1h30min").agg(agg)
        out = out.dropna(subset=["open"]).tz_convert("UTC")
    else:
        out = df.resample("4h").agg(agg).dropna(subset=["open"])
    out.index.name = "timestamp"
    return out


# --------------------------------------------------- yahoo (stocks + forex)
def yahoo_batch(tickers, interval, period):
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval=interval, group_by="ticker",
                      auto_adjust=True, threads=False, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    if len(tickers) == 1:
        out[tickers[0]] = clean_yahoo(raw.droplevel(0, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw)
        return out
    for t in tickers:
        try:
            out[t] = clean_yahoo(raw[t].dropna(how="all"))
        except KeyError:
            pass
    return out


def run_yahoo(asset, symbol_map, refresh, index_rows, session_tz=None, batch=40, tfs=("1d", "4h")):
    """symbol_map: {file_symbol: yahoo_ticker}"""
    todo = {s: y for s, y in symbol_map.items()
            if refresh or not all(exists(asset, tf, s) for tf in tfs)}
    items = list(todo.items())
    print(f"[{asset}] {len(items)} symbols to download ({len(symbol_map) - len(items)} already done)")
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        tickers = [y for _, y in chunk]
        rev = {y: s for s, y in chunk}
        try:
            daily = yahoo_batch(tickers, "1d", "max") if "1d" in tfs else {}
            hourly = yahoo_batch(tickers, "1h", "729d") if "4h" in tfs else {}
        except Exception as e:
            print(f"  batch failed: {e}")
            continue
        for y in tickers:   # retry anything that failed (e.g. "database is locked")
            for res, iv, pr, tf in ((daily, "1d", "max", "1d"), (hourly, "1h", "729d", "4h")):
                if tf not in tfs:
                    continue
                d = res.get(y)
                if d is None or d.empty:
                    try:
                        res.update(yahoo_batch([y], iv, pr))
                    except Exception:
                        pass
        for y in tickers:
            s = rev[y]
            if "1d" in tfs:
                save(daily.get(y), asset, "1d", s, index_rows)
            if "4h" in tfs:
                save(resample_4h(hourly.get(y), session_tz), asset, "4h", s, index_rows)
        print(f"  [{asset}] {min(i + batch, len(items))}/{len(items)}")
        time.sleep(1)


def sp500_symbols():
    html = SESSION.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    syms = table["Symbol"].astype(str).str.strip().tolist()
    return {s: s.replace(".", "-") for s in syms}  # BRK.B -> BRK-B for Yahoo


def forex_symbols():
    # every pair among these currencies (Yahoo returns nothing for pairs it doesn't carry, those get skipped)
    # market convention order: earlier currency is the base (EURUSD, GBPJPY, USDJPY, AUDNZD...)
    ccys = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY",
            "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "TRY", "ZAR",
            "MXN", "BRL", "SGD", "HKD", "CNH", "INR", "KRW", "ILS", "THB"]
    majors = ccys[:8]
    pairs = {}
    for i, a in enumerate(ccys):
        for b in ccys[i + 1:]:
            if a in majors or b in majors:   # every pair with at least one major, Yahoo skips ones it lacks
                pairs[f"{a}{b}"] = f"{a}{b}=X"
    return pairs


# ------------------------------------------------------------ binance dump
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BAD = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")


def s3_list(prefix, delimiter=None):
    """Return (common_prefixes, keys) under a prefix, following pagination."""
    prefixes, keys, marker = [], [], ""
    while True:
        params = {"prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        r = SESSION.get(S3, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        prefixes += [e.text for e in root.findall(".//s:CommonPrefixes/s:Prefix", ns)]
        keys += [e.text for e in root.findall(".//s:Contents/s:Key", ns)]
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return prefixes, keys
        marker = root.findtext("s:NextMarker", namespaces=ns) or (keys[-1] if keys else prefixes[-1])


def binance_symbols(quote):
    prefixes, _ = s3_list("data/spot/monthly/klines/", delimiter="/")
    syms = [p.rstrip("/").split("/")[-1] for p in prefixes]
    return sorted(s for s in syms if s.endswith(quote) and not BAD.search(s))


def read_kline_zip(key):
    r = SESSION.get(f"{S3}/{key}", timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(df.iloc[0, 0]).isdigit():   # header row present
        df = df.iloc[1:]
    df = df.iloc[:, :6].astype(float)
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    ts = df["ts"]
    unit = pd.Series("ms", index=df.index)
    unit[ts > 1e14] = "us"                 # spot files from 2025 on use microseconds
    idx = pd.to_datetime(ts.where(ts <= 1e14, ts / 1000), unit="ms", utc=True)
    df.index = idx
    df.index.name = "timestamp"
    return df[COLS]


def binance_symbol(sym, tf):
    keys = []
    for kind in ("monthly", "daily"):
        _, k = s3_list(f"data/spot/{kind}/klines/{sym}/{tf}/")
        keys += [x for x in k if x.endswith(".zip")]
    # daily files for months already covered by a monthly file are redundant
    monthly_months = {re.search(r"(\d{4}-\d{2})\.zip$", k).group(1)
                      for k in keys if "/monthly/" in k and re.search(r"(\d{4}-\d{2})\.zip$", k)}
    keys = [k for k in keys if "/monthly/" in k or
            not (re.search(r"(\d{4}-\d{2})-\d{2}\.zip$", k) and
                 re.search(r"(\d{4}-\d{2})-\d{2}\.zip$", k).group(1) in monthly_months)]
    frames = [f for f in (read_kline_zip(k) for k in keys) if f is not None]
    return pd.concat(frames) if frames else None


def run_crypto(quote, refresh, limit, index_rows, tfs=("1d", "4h")):
    syms = binance_symbols(quote)
    if limit:
        syms = syms[:limit]
    todo = [s for s in syms if refresh or not all(exists("crypto", tf, s) for tf in tfs)]
    print(f"[crypto] {len(syms)} {quote} pairs on Binance, {len(todo)} to download")

    def job(sym):
        res = {}
        for tf in tfs:
            res[tf] = binance_symbol(sym, tf)
        return sym, res

    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(job, s) for s in todo]
        for f in as_completed(futs):
            try:
                sym, res = f.result()
                for tf, df in res.items():
                    save(df, "crypto", tf, sym, index_rows)
            except Exception as e:
                print(f"  failed: {e}")
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  [crypto] {done}/{len(todo)}")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["stocks", "forex", "crypto"], help="just one asset class")
    ap.add_argument("--limit", type=int, default=0, help="only N symbols per class (for testing)")
    ap.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    ap.add_argument("--tf", choices=["1d", "4h", "both"], default="both", help="timeframe to download (default both)")
    ap.add_argument("--quote", default="USDT", help="crypto quote currency (default USDT)")
    a = ap.parse_args()

    tfs = ("1d", "4h") if a.tf == "both" else (a.tf,)
    rows = []
    try:
        if a.only in (None, "stocks"):
            m = sp500_symbols()
            if a.limit:
                m = dict(list(m.items())[:a.limit])
            run_yahoo("stocks", m, a.refresh, rows, session_tz="America/New_York", tfs=tfs)
            write_index(rows); rows.clear()
        if a.only in (None, "forex"):
            m = forex_symbols()
            if a.limit:
                m = dict(list(m.items())[:a.limit])
            run_yahoo("forex", m, a.refresh, rows, tfs=tfs)
            write_index(rows); rows.clear()
        if a.only in (None, "crypto"):
            run_crypto(a.quote, a.refresh, a.limit, rows, tfs=tfs)
            write_index(rows); rows.clear()
    except KeyboardInterrupt:
        print("stopped, saving index")
    write_index(rows)
    print("done. files are in", OUT.resolve())


if __name__ == "__main__":
    main()
