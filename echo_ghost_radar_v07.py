# =========================================================
# ECHO GHOST COMMODITY CYCLE RADAR v0.7.0
# Commodity -> Theme -> Equity leadership engine
# Clean code + full features
# =========================================================

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

LOOKBACK_PERIOD = "1y"
TOP_PER_THEME = 8
GLOBAL_TOP_N = 15

MIN_PRICE = 5
MIN_DOLLAR_VOL = 5_000_000
MIN_HISTORY = 220
LEADER_MAX_DISTANCE_FROM_52W_HIGH = 0.25
ALLOW_DEEP_VALUE = True

OUT_DIR = "radar_output"
os.makedirs(OUT_DIR, exist_ok=True)

COMMODITY_HISTORY = os.path.join(OUT_DIR, "commodity_history.csv")
THEME_HISTORY     = os.path.join(OUT_DIR, "theme_history.csv")
EQUITY_HISTORY    = os.path.join(OUT_DIR, "equity_history.csv")
GLOBAL_HISTORY    = os.path.join(OUT_DIR, "global_history.csv")
THEME_SCORES_FILE = os.path.join(OUT_DIR, "theme_scores_history.csv")

# =========================================================
# COMMODITY DRIVERS
# =========================================================

commodities = {
    "Copper":     "HG=F",
    "Oil":        "CL=F",
    "NatGas":     "NG=F",
    "Corn":       "ZC=F",
    "Wheat":      "ZW=F",
    "UraniumETF": "URA",
    "Gold":       "GC=F",
    "Silver":     "SI=F",
}

# =========================================================
# THEMES
# =========================================================

themes = {
    "Steel Cycle": {
        "drivers": ["Copper"],
        "universe": ["CLF", "X", "NUE", "STLD", "FCX", "VALE", "BHP", "RIO", "AA", "SCCO", "MT", "CMC"],
    },
    "Energy Cycle": {
        "drivers": ["Oil", "NatGas"],
        "universe": ["XOM", "CVX", "COP", "EOG", "DVN", "FANG", "MRO", "APA", "OXY", "HES", "BTU", "AR", "EQT"],
    },
    "Uranium Cycle": {
        "drivers": ["UraniumETF"],
        "universe": ["CCJ", "NXE", "UEC", "DNN", "UUUU", "LEU", "URG", "UROY"],
    },
    "Food Inflation": {
        "drivers": ["Corn", "Wheat", "NatGas"],
        "universe": ["MOS", "NTR", "CF", "ADM", "BG", "DE", "AGCO", "CALM"],
    },
    "Precious Metals": {
        "drivers": ["Gold", "Silver"],
        "universe": ["NEM", "AEM", "WPM", "FNV", "AG", "PAAS", "GOLD", "KGC", "BTG", "HL", "MAG"],
    },
}

# =========================================================
# UTILITIES
# =========================================================

def safe_download(tickers, period=LOOKBACK_PERIOD):
    try:
        return yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()


def close_series(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                s = data[ticker]["Close"].dropna()
                return s if not s.empty else None
        if "Close" in data.columns:
            s = data["Close"].dropna()
            return s if not s.empty else None
        return None
    except Exception:
        return None


def volume_series(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                s = data[ticker]["Volume"].dropna()
                return s if not s.empty else None
        if "Volume" in data.columns:
            s = data["Volume"].dropna()
            return s if not s.empty else None
        return None
    except Exception:
        return None


def pct(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return (a / b) - 1


def fmt(x):
    return f"{x:,.2f}" if pd.notna(x) else "-"


def fmtp(x):
    return f"{x*100:,.1f}%" if pd.notna(x) else "-"


def append_csv(df, file):
    if df.empty:
        return
    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)


def load_csv(path):
    try:
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def last_run_rows(df, col="run_ts"):
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    try:
        max_ts = df[col].astype(str).max()
        return df[df[col].astype(str) == max_ts].copy()
    except Exception:
        return pd.DataFrame()


def slope_norm(series, n=20):
    s = series.dropna().tail(n)
    if len(s) < max(10, n // 2):
        return np.nan
    y = s.values.astype(float)
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)[0]
    denom = np.mean(np.abs(y)) or 1.0
    return coef / denom


# =========================================================
# COMMODITY ENGINE
# =========================================================

def commodity_score(close):
    if len(close) < 200:
        return None

    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    p      = float(close.iloc[-1])
    ma50v  = float(ma50.iloc[-1])
    ma200v = float(ma200.iloc[-1])

    r21 = pct(close.iloc[-1], close.iloc[-22]) if len(close) >= 22 else np.nan
    r63 = pct(close.iloc[-1], close.iloc[-64]) if len(close) >= 64 else np.nan

    ma50_gap  = pct(p, ma50v)
    ma200_gap = pct(p, ma200v)
    slope     = slope_norm(ma50, 20)

    # Continuous score
    score = 0.0
    if pd.notna(ma50_gap):  score += ma50_gap  * 100 * 0.9
    if pd.notna(ma200_gap): score += ma200_gap * 100 * 1.4
    if pd.notna(r21):       score += r21       * 100 * 1.2
    if pd.notna(r63):       score += r63       * 100 * 1.0
    if pd.notna(slope):     score += slope     * 1000 * 0.8

    # Acceleration
    accel = 0.0
    if pd.notna(r21) and pd.notna(r63):
        accel += (r21 * 3 - r63) * 100
    if pd.notna(slope):
        accel += slope * 1000

    # Categorical signal
    if p > ma50v and ma50v > ma200v:
        signal = "BULLISH"
    elif p > ma50v:
        signal = "WARMING"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "price":  p,
        "score":  score,
        "accel":  accel,
        "r21":    r21,
        "r63":    r63,
        "vs50":   ma50_gap,
        "vs200":  ma200_gap,
    }


# =========================================================
# EQUITY FACTORS
# =========================================================

def equity_score(close, vol, spy):
    close = close.dropna()
    if len(close) < MIN_HISTORY:
        return None

    # Volume alignment
    if vol is not None:
        aligned = pd.concat([close.rename("c"), vol.rename("v")], axis=1).dropna()
        if aligned.empty:
            vol = None
        else:
            close = aligned["c"]
            vol   = aligned["v"]

    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma100 = close.rolling(100).mean()
    ma200 = close.rolling(200).mean()

    p      = float(close.iloc[-1])
    ma20v  = float(ma20.iloc[-1])
    ma50v  = float(ma50.iloc[-1])
    ma100v = float(ma100.iloc[-1])
    ma200v = float(ma200.iloc[-1])

    high252 = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())

    r21  = pct(close.iloc[-1], close.iloc[-22])  if len(close) >= 22  else np.nan
    r63  = pct(close.iloc[-1], close.iloc[-64])  if len(close) >= 64  else np.nan
    r126 = pct(close.iloc[-1], close.iloc[-127]) if len(close) >= 127 else np.nan

    vol63 = float(close.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)) if len(close) >= 63 else np.nan

    # Dollar volume filter
    avg_dollar = np.nan
    if vol is not None and len(vol) >= 20:
        recent = pd.concat([close.tail(20), vol.tail(20)], axis=1).dropna()
        if not recent.empty:
            avg_dollar = (recent.iloc[:, 0] * recent.iloc[:, 1]).mean()

    if pd.isna(avg_dollar) or avg_dollar < MIN_DOLLAR_VOL:
        return None

    dist52 = (high252 - p) / high252 if high252 > 0 else np.nan

    # Leader qualification
    leader = (
        p >= MIN_PRICE and
        p > ma200v and
        pd.notna(dist52) and dist52 <= LEADER_MAX_DISTANCE_FROM_52W_HIGH
    )

    deep_value = p >= MIN_PRICE and not leader

    # Trend score
    trend = 0
    if p > ma20v:  trend += 1
    if p > ma50v:  trend += 1
    if p > ma100v: trend += 1
    if p > ma200v: trend += 2
    if ma50v > ma200v: trend += 2
    if ma20v > ma50v:  trend += 1

    # Momentum
    mom = 0.0
    if pd.notna(r21):  mom += r21  * 100
    if pd.notna(r63):  mom += r63  * 120
    if pd.notna(r126): mom += r126 * 80

    ma200_gap = pct(p, ma200v) * 100 if ma200v > 0 else 0
    vol_pen   = vol63 * 8 if pd.notna(vol63) else 0

    # Relative strength vs SPY
    rs63 = rs126 = rs_ratio_gap = np.nan
    rs_comp = 0.0
    if spy is not None and len(spy) >= 127:
        spy_r63  = pct(spy.iloc[-1], spy.iloc[-64])
        spy_r126 = pct(spy.iloc[-1], spy.iloc[-127])

        if pd.notna(r63)  and pd.notna(spy_r63):  rs63  = r63  - spy_r63
        if pd.notna(r126) and pd.notna(spy_r126): rs126 = r126 - spy_r126

        try:
            ratio = close / spy.reindex(close.index).dropna()
            ratio_ma50 = ratio.rolling(50).mean()
            if len(ratio_ma50.dropna()) > 0 and ratio_ma50.iloc[-1] > 0:
                rs_ratio_gap = float(ratio.iloc[-1] / ratio_ma50.iloc[-1] - 1)
        except Exception:
            pass

        if pd.notna(rs63):       rs_comp += rs63       * 100 * 1.3
        if pd.notna(rs126):      rs_comp += rs126      * 100 * 1.0
        if pd.notna(rs_ratio_gap): rs_comp += rs_ratio_gap * 100 * 1.2

    comp = trend + mom + ma200_gap - vol_pen + rs_comp

    return {
        "price":    p,
        "r21":      r21,
        "r63":      r63,
        "r126":     r126,
        "vol63":    vol63,
        "trend":    trend,
        "ma200_gap": ma200_gap,
        "avg_dollar": avg_dollar,
        "dist52":   dist52,
        "rs63":     rs63,
        "rs126":    rs126,
        "rs_ratio_gap": rs_ratio_gap,
        "rs_comp":  rs_comp,
        "comp":     comp,
        "leader":   leader,
        "deep_value": deep_value,
    }


# =========================================================
# ALERT BUILDERS
# =========================================================

def commodity_alerts(current_rows):
    alerts = []
    prev = last_run_rows(load_csv(COMMODITY_HISTORY))
    if prev.empty:
        return alerts
    prev_map = prev.set_index("commodity").to_dict("index")
    for row in current_rows:
        name = row["commodity"]
        p = prev_map.get(name)
        if not p:
            continue
        if row["signal"] != p.get("signal"):
            alerts.append(f"{name} flipped {p.get('signal')} -> {row['signal']}")
        try:
            delta = float(row["score"]) - float(p.get("score", 0))
            if delta >= 4:
                alerts.append(f"{name} accelerated +{delta:.1f}")
            elif delta <= -4:
                alerts.append(f"{name} cooled {delta:.1f}")
        except Exception:
            pass
    return list(dict.fromkeys(alerts))


def theme_alerts(current_rows):
    alerts = []
    prev = last_run_rows(load_csv(THEME_HISTORY))
    if prev.empty:
        return alerts
    prev_map = prev.set_index("theme").to_dict("index")
    for row in current_rows:
        name = row["theme"]
        p = prev_map.get(name)
        if not p:
            continue
        if row["status"] != p.get("status"):
            alerts.append(f"{name} flipped {p.get('status')} -> {row['status']}")
        try:
            delta = float(row["score"]) - float(p.get("score", 0))
            if delta >= 5:
                alerts.append(f"{name} cycle accelerating +{delta:.1f}")
            elif delta <= -5:
                alerts.append(f"{name} cycle fading {delta:.1f}")
        except Exception:
            pass
    return list(dict.fromkeys(alerts))


def global_alerts(current_df):
    alerts = []
    prev = last_run_rows(load_csv(GLOBAL_HISTORY))
    if prev.empty or current_df.empty:
        return alerts
    prev_ranks = {r["ticker"]: int(r["rank"]) for _, r in prev.iterrows()}
    prev_set   = set(prev["ticker"].astype(str))
    cur_set    = set(current_df["ticker"].astype(str))
    for _, row in current_df.iterrows():
        tk = row["ticker"]
        rk = int(row["rank"])
        if tk not in prev_set:
            alerts.append(f"{tk} entered GLOBAL TOP {GLOBAL_TOP_N} at #{rk}")
        elif tk in prev_ranks and rk < prev_ranks[tk] and (prev_ranks[tk] - rk) >= 3:
            alerts.append(f"{tk} jumped {prev_ranks[tk]-rk} spots to #{rk}")
    for tk in sorted(prev_set - cur_set)[:5]:
        alerts.append(f"{tk} dropped out of GLOBAL TOP {GLOBAL_TOP_N}")
    return list(dict.fromkeys(alerts))


# =========================================================
# CHART
# =========================================================

def plot_themes():
    try:
        df = pd.read_csv(THEME_SCORES_FILE, parse_dates=["run_ts"])
        if df.empty:
            return
        df = df.fillna(0).set_index("run_ts")
        cols = [c for c in df.columns if "score" in c.lower()]
        if not cols:
            return
        ax = df[cols].plot(figsize=(13, 6), title="Theme Score History")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "theme_chart.png"))
        plt.close()
    except Exception:
        pass


# =========================================================
# MAIN
# =========================================================

def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nECHO GHOST COMMODITY CYCLE RADAR v0.7.0  {now}\n")

    # --- Download core data ---
    core_tickers = list(commodities.values()) + ["SPY"]
    core_data    = safe_download(core_tickers)
    spy          = close_series(core_data, "SPY")

    # -------------------------------------------------------
    # COMMODITIES
    # -------------------------------------------------------
    results = {}
    comm_rows = []

    print("COMMODITIES")
    print("-" * 100)
    print(f"{'Name':<16} {'Signal':<10} {'Price':>10} {'21D':>8} {'63D':>8} {'Score':>8} {'Accel':>8} {'vs50':>8} {'vs200':>8}")
    print("-" * 100)

    for name, ticker in commodities.items():
        close = close_series(core_data, ticker)
        if close is None:
            results[name] = {"signal": "NO DATA", "score": 0, "accel": 0}
            continue

        res = commodity_score(close)
        if res is None:
            results[name] = {"signal": "NO DATA", "score": 0, "accel": 0}
            continue

        results[name] = res

        comm_rows.append({
            "run_ts":    now,
            "commodity": name,
            "ticker":    ticker,
            "signal":    res["signal"],
            "score":     res["score"],
            "accel":     res["accel"],
            "r21":       res["r21"],
            "r63":       res["r63"],
        })

        print(
            f"{name:<16} {res['signal']:<10} {fmt(res['price']):>10} "
            f"{fmtp(res['r21']):>8} {fmtp(res['r63']):>8} "
            f"{res['score']:>8.1f} {res['accel']:>8.1f} "
            f"{fmtp(res['vs50']):>8} {fmtp(res['vs200']):>8}"
        )

    append_csv(pd.DataFrame(comm_rows), COMMODITY_HISTORY)

    # -------------------------------------------------------
    # THEMES
    # -------------------------------------------------------
    theme_outputs = []
    theme_rows    = []
    scores_row    = {"run_ts": now}

    print("\nTHEMES")
    print("-" * 90)
    print(f"{'Theme':<22} {'Status':<10} {'Score':>8} {'Accel':>8}  Drivers")
    print("-" * 90)

    for t, config in themes.items():
        drivers = config["drivers"]

        score = sum(results.get(d, {}).get("score", 0) for d in drivers)
        accel = sum(results.get(d, {}).get("accel", 0) for d in drivers)

        status = "NEUTRAL"
        if score > 20:
            status = "BULLISH"
        elif score > 8:
            status = "WARMING"

        theme_outputs.append({"theme": t, "score": score, "accel": accel, "status": status})
        theme_rows.append({"run_ts": now, "theme": t, "score": score, "accel": accel, "status": status})

        scores_row[f"{t}|score"] = round(score, 2)
        scores_row[f"{t}|accel"] = round(accel, 2)

        print(f"{t:<22} {status:<10} {score:>8.1f} {accel:>8.1f}  {', '.join(drivers)}")

    append_csv(pd.DataFrame(theme_rows), THEME_HISTORY)

    df_scores = pd.DataFrame([scores_row])
    if os.path.exists(THEME_SCORES_FILE):
        df_scores.to_csv(THEME_SCORES_FILE, mode="a", header=False, index=False)
    else:
        df_scores.to_csv(THEME_SCORES_FILE, index=False)

    plot_themes()

    # -------------------------------------------------------
    # EQUITY SCANNER
    # -------------------------------------------------------
    active   = [t for t in theme_outputs if t["status"] != "NEUTRAL"]
    combined = []
    eq_rows  = []

    for t in active:
        theme_name = t["theme"]
        universe   = themes[theme_name]["universe"]
        data       = safe_download(universe)
        leaders    = []
        values     = []

        for ticker in universe:
            cl = close_series(data, ticker)
            vl = volume_series(data, ticker)
            if cl is None:
                continue
            fac = equity_score(cl, vl, spy)
            if fac is None:
                continue
            fac["ticker"] = ticker
            fac["theme"]  = theme_name
            fac["theme_status"] = t["status"]
            if fac["leader"]:
                leaders.append(fac)
            elif ALLOW_DEEP_VALUE and fac["deep_value"]:
                values.append(fac)

        leaders.sort(key=lambda x: x["comp"], reverse=True)
        values.sort(key=lambda x: x["comp"], reverse=True)
        top = leaders[:TOP_PER_THEME]

        print(f"\n{theme_name} | {t['status']}")
        print("-" * 130)
        print(f"{'#':<4} {'Ticker':<8} {'Price':>9} {'21D':>7} {'63D':>7} {'126D':>7} {'Trend':>6} {'vs200':>7} {'RS63':>7} {'RS126':>7} {'$Vol20':>12} {'Comp':>9}")
        print("-" * 130)

        if not top:
            print("  No leaders passed filter.")
        else:
            for i, r in enumerate(top, 1):
                print(
                    f"{i:<4} {r['ticker']:<8} {fmt(r['price']):>9} "
                    f"{fmtp(r['r21']):>7} {fmtp(r['r63']):>7} {fmtp(r['r126']):>7} "
                    f"{int(r['trend']):>6} {r['ma200_gap']:>6.1f}% "
                    f"{fmtp(r['rs63']):>7} {fmtp(r['rs126']):>7} "
                    f"{fmt(r['avg_dollar']):>12} {r['comp']:>9.2f}"
                )
                combined.append(r)
                eq_rows.append({"run_ts": now, **{k: v for k, v in r.items()}})

        if values:
            print(f"  Value bucket: {', '.join(v['ticker'] for v in values[:5])}")

    append_csv(pd.DataFrame(eq_rows), EQUITY_HISTORY)

    # -------------------------------------------------------
    # GLOBAL BEST IDEAS
    # -------------------------------------------------------
    if not combined:
        print("\nNo global ideas this run.")
    else:
        df = pd.DataFrame(combined)
        df = df.sort_values(["comp", "rs_comp", "trend"], ascending=False)
        df = df.drop_duplicates("ticker").head(GLOBAL_TOP_N).reset_index(drop=True)
        df["rank"] = df.index + 1

        print("\n" + "=" * 130)
        print("GLOBAL BEST IDEAS")
        print("=" * 130)
        print(f"{'#':<4} {'Ticker':<8} {'Theme':<24} {'Price':>9} {'63D':>7} {'126D':>7} {'RS63':>7} {'RS126':>7} {'Comp':>9}")
        print("-" * 130)

        for _, r in df.iterrows():
            print(
                f"{int(r['rank']):<4} {r['ticker']:<8} {r['theme'][:24]:<24} "
                f"{fmt(r['price']):>9} {fmtp(r['r63']):>7} {fmtp(r['r126']):>7} "
                f"{fmtp(r['rs63']):>7} {fmtp(r['rs126']):>7} {r['comp']:>9.2f}"
            )

        global_rows = [
            {"run_ts": now, "rank": int(r["rank"]), "ticker": r["ticker"],
             "theme": r["theme"], "comp": r["comp"]}
            for _, r in df.iterrows()
        ]
        append_csv(pd.DataFrame(global_rows), GLOBAL_HISTORY)

        g_alerts = global_alerts(df)

    if not combined:
        g_alerts = []

    # -------------------------------------------------------
    # ALERTS
    # -------------------------------------------------------
    c_alerts = commodity_alerts(comm_rows)
    t_alerts = theme_alerts(theme_rows)
    all_alerts = list(dict.fromkeys(c_alerts + t_alerts + g_alerts))

    print("\n" + "=" * 130)
    print("ALERTS")
    print("=" * 130)
    if all_alerts:
        for a in all_alerts:
            print(f"  - {a}")
    else:
        print("  No changes vs prior run.")

    print(f"\nDone. Output in {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
