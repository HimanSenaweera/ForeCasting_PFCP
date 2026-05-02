import itertools, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ── Metric functions ─────────────────────────────────────────────────────────
def smape(y_true, y_pred):
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100)

def mape_pct(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Outlier Treatment
# ══════════════════════════════════════════════════════════════════════════════

OUTLIER_REPLACEMENTS = {
    "2024-02-29": "2024-01-31",
    "2024-04-30": "2024-03-31",
    "2024-05-31": "2024-06-30",
}

# ── apply to train_series only ───────────────────────────────────────────────
train_clean = train_series.copy()

print(f"\n{'─'*60}")
print(f"  Outlier Treatment")
print(f"{'─'*60}")
print(f"  {'Date':<15} {'Original':>14} {'Source Date':>14} {'New Value':>14}")
print(f"  {'─'*60}")

for bad_date, good_date in OUTLIER_REPLACEMENTS.items():
    bad  = datetime.date.fromisoformat(bad_date)
    good = datetime.date.fromisoformat(good_date)
    if bad in train_clean.index and good in train_clean.index:
        orig = train_clean[bad]
        new  = train_series[good]
        train_clean[bad] = new
        print(f"  {bad_date:<15} {orig:>14,.0f} {good_date:>14} {new:>14,.0f}")
    else:
        print(f"  {bad_date:<15} — skipped (date not in train_series)")

print(f"{'─'*60}")
print(f"\n  Before — Min / Max / Mean : {train_series.min():,.0f} / {train_series.max():,.0f} / {train_series.mean():,.0f}")
print(f"  After  — Min / Max / Mean : {train_clean.min():,.0f} / {train_clean.max():,.0f} / {train_clean.mean():,.0f}")

# ── build full_clean — test window completely untouched ──────────────────────
full_clean       = data_df[TARGET_METRIC].copy()
full_clean.index = pd.to_datetime(full_clean.index).date

for bad_date in OUTLIER_REPLACEMENTS:
    d = datetime.date.fromisoformat(bad_date)
    if d in train_clean.index:
        full_clean[d] = train_clean[d]

# full series for plotting (train + test)
full_series_clean = full_clean[
    full_clean.index >= datetime.date.fromisoformat(first_input_date)
].dropna()

print(f"\n  train_clean       : {len(train_clean)} points  ({train_clean.index[0]} → {train_clean.index[-1]})")
print(f"  full_series_clean : {len(full_series_clean)} points  ({full_series_clean.index[0]} → {full_series_clean.index[-1]})")


# ── before vs after plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=False)

plot_config = [
    (axes[0], train_series,      '[PART A — TRAIN ONLY]  BEFORE',     PALETTE['raw']),
    (axes[1], train_clean,       '[PART A — TRAIN ONLY]  AFTER',      PALETTE['raw']),
    (axes[2], full_series,       '[PART B — TRAIN + TEST]  BEFORE',   PALETTE['actual']),
    (axes[3], full_series_clean, '[PART B — TRAIN + TEST]  AFTER',    PALETTE['actual']),
]

for ax, series, label, color in plot_config:
    is_after = 'AFTER' in label

    ax.plot(pd.to_datetime(series.index), series.values,
            marker='o', linewidth=2, color=color, markersize=5,
            label='Actual (treated)' if is_after else 'Actual')

    ax.axvline(pd.to_datetime(LAST_TRAIN_DATE), color='grey',
               linestyle='--', linewidth=1.2, label='Train/Test cutoff')

    for bad_date in OUTLIER_REPLACEMENTS:
        d = datetime.date.fromisoformat(bad_date)
        if d in series.index:
            ax.axvline(pd.to_datetime(bad_date),
                       color='green' if is_after else 'red',
                       linestyle=':', linewidth=1.2, alpha=0.8)

    y_min   = series.min()
    y_max   = series.max()
    padding = (y_max - y_min) * 0.05

    ax.set_ylim(y_min - padding, y_max + padding)
    ax.fill_between(pd.to_datetime(series.index), series.values,
                    y2=y_min - padding, alpha=0.12, color=color)
    ax.set_title(f"{label}  —  {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})")
    ax.set_ylabel('Outstanding Balances ($)')
    ax.set_xlabel('Date')
    ax.set_xticks(pd.to_datetime(series.index))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=9)
    ax.annotate(f"n = {len(series)}", xy=(0.01, 0.92),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.suptitle('Outlier Treatment — Before vs After', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ARIMA Grid Search on cleaned series
# ══════════════════════════════════════════════════════════════════════════════

def univariate_arima_grid_search(
    series,
    train_range,
    test_range,
    p_range=(0, 1, 2, 3),
    d_range=(0, 1),
    q_range=(0, 1, 2, 3),
    trend_options=("n",),
    rank_by="smape",
    top_k=20,
    min_train=20,
):
    s = series.copy()
    s.index = pd.to_datetime(s.index).date

    t0, t1 = map(datetime.date.fromisoformat, train_range)
    s0, s1 = map(datetime.date.fromisoformat, test_range)

    idx     = pd.Index(s.index)
    train_s = s[(idx >= t0) & (idx <= t1)].astype(float)
    test_s  = s[(idx >= s0) & (idx <= s1)].astype(float)

    if len(train_s) < min_train:
        raise ValueError(f"Only {len(train_s)} training points — need at least {min_train}")
    if len(test_s) == 0:
        raise ValueError("Test set is empty — check test_range dates")

    order_combos = list(itertools.product(p_range, d_range, q_range))
    total        = len(order_combos) * len(trend_options)

    # reasonable bounds to prevent explosive predictions
    reasonable_min = float(train_s.mean() - 4 * train_s.std())
    reasonable_max = float(train_s.mean() + 4 * train_s.std())

    print(f"\nTrain points    : {len(train_s)}  ({t0} → {t1})")
    print(f"Test  points    : {len(test_s)}   ({s0} → {s1})")
    print(f"Order combos    : {len(order_combos)}")
    print(f"Trend options   : {len(trend_options)}")
    print(f"Total fits      : {total * len(test_s):,}  ({total} combos × {len(test_s)} test steps)\n")

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="ARIMA grid search")

    for order in order_combos:
        for trend in trend_options:
            pbar.update(1)
            try:
                predictions = []
                history_y   = train_s.copy()

                for i in range(len(test_s)):
                    model = SARIMAX(
                        history_y,
                        order=order,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)

                    pred = model.get_forecast(steps=1).predicted_mean.iloc[0]

                    # clip explosive predictions
                    if (np.isnan(pred) or np.isinf(pred) or
                            pred < reasonable_min or pred > reasonable_max):
                        pred = float(history_y.iloc[-1])

                    predictions.append(pred)

                    # roll forward with prediction
                    history_y = pd.concat([
                        history_y,
                        pd.Series([pred], index=[test_s.index[i]])
                    ])

                actual = test_s.values
                fc     = np.array(predictions)
                mask   = actual != 0

                if not mask.any():
                    continue

                yt, yp = actual[mask], fc[mask]

                row = {
                    "smape":   smape(yt, yp),
                    "mape":    mape_pct(yt, yp),
                    "mae":     mae(yt, yp),
                    "rmse":    rmse(yt, yp),
                    "order":   order,
                    "trend":   trend,
                    "n_train": len(train_s),
                    "n_test":  len(test_s),
                }
                results.append(row)

                if row[rank_by] < best.get(rank_by, np.inf):
                    best = {
                        **row,
                        "forecast":   fc,
                        "actual":     actual,
                        "test_index": test_s.index,
                    }

            except Exception as e:
                results.append({
                    "smape": np.nan, "mape": np.nan,
                    "mae":   np.nan, "rmse": np.nan,
                    "order": order,  "trend": trend,
                    "error": str(e)[:120],
                })

    pbar.close()

    out = pd.DataFrame(results)

    if out.empty or out[rank_by].isna().all():
        print("⚠ All combinations failed. Errors:")
        if "error" in out.columns:
            print(out["error"].value_counts().head(5))
        return out, best

    out = out.sort_values(rank_by, na_position="last").reset_index(drop=True)

    print(f"\n✓ Best {rank_by.upper()} : {best.get(rank_by,  np.nan):.4f}")
    print(f"  MAPE           : {best.get('mape',  np.nan):.4f}%")
    print(f"  SMAPE          : {best.get('smape', np.nan):.4f}%")
    print(f"  MAE            : {best.get('mae',   np.nan):>14,.0f}")
    print(f"  RMSE           : {best.get('rmse',  np.nan):>14,.0f}")
    print(f"  Order ARIMA    : {best.get('order')}")
    print(f"  Trend          : {best.get('trend')}")

    return out.head(top_k), best


# ── Run ───────────────────────────────────────────────────────────────────────
top, best = univariate_arima_grid_search(
    series        = full_clean,                          # ← cleaned series
    train_range   = (first_input_date, last_input_date),
    test_range    = (first_test_date,  last_test_date),
    p_range       = (0, 1, 2, 3),
    d_range       = (0, 1),
    q_range       = (0, 1, 2, 3),
    trend_options = ("n", "c"),
    rank_by       = "smape",
    top_k         = 20,
)

top[["smape", "mape", "mae", "rmse", "order", "trend"]]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Inference using best model
# ══════════════════════════════════════════════════════════════════════════════

s = full_clean.copy()
s.index = pd.to_datetime(s.index).date
idx     = pd.Index(s.index)

t0, t1 = map(datetime.date.fromisoformat, (first_input_date, last_input_date))
s0, s1 = map(datetime.date.fromisoformat, (first_test_date,  last_test_date))

train_s = s[(idx >= t0) & (idx <= t1)].astype(float)
test_s  = s[(idx >= s0) & (idx <= s1)].astype(float)

reasonable_min = float(train_s.mean() - 4 * train_s.std())
reasonable_max = float(train_s.mean() + 4 * train_s.std())

predictions = []
history_y   = train_s.copy()
order       = best['order']
trend       = best['trend']

for i in range(len(test_s)):
    model = SARIMAX(
        history_y,
        order=order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    pred = model.get_forecast(steps=1).predicted_mean.iloc[0]

    if (np.isnan(pred) or np.isinf(pred) or
            pred < reasonable_min or pred > reasonable_max):
        pred = float(history_y.iloc[-1])

    predictions.append(pred)

    history_y = pd.concat([
        history_y,
        pd.Series([pred], index=[test_s.index[i]])
    ])

# ── results ───────────────────────────────────────────────────────────────────
arima_fc = pd.Series(predictions, index=test_s.index)
actual   = test_s.values
fc       = np.array(predictions)
mask     = actual != 0
yt, yp   = actual[mask], fc[mask]

print(f"\n{'='*50}")
print(f"  ARIMA Inference Results  (cleaned series)")
print(f"{'='*50}")
print(f"  Order  : {order}")
print(f"  Trend  : {trend}")
print(f"  MAPE   : {mape_pct(yt, yp):.4f}%")
print(f"  SMAPE  : {smape(yt, yp):.4f}%")
print(f"  MAE    : {mae(yt, yp):>14,.0f}")
print(f"  RMSE   : {rmse(yt, yp):>14,.0f}")
print(f"{'='*50}")

# ── plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(test_s.index, test_s.values,
         label='Actual', color='steelblue', marker='o', ms=4)
plt.plot(arima_fc.index, arima_fc.values,
         label='ARIMA forecast', color='red', marker='o', ms=4)
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.title(f"ARIMA{order} — {TARGET_METRIC}  (outlier treated)")
plt.legend()
plt.tight_layout()
plt.show()
