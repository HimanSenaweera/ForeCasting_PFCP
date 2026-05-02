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

OUTLIER_START = "2024-01-31"
OUTLIER_END   = "2024-04-30"

# build train series from your existing data_df
train_series = data_df.loc[
    (data_df.index >= datetime.date.fromisoformat(first_input_date)) &
    (data_df.index <= datetime.date.fromisoformat(last_input_date)),
    TARGET_METRIC
].dropna()

# ── visualise before ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(train_series.index, train_series.values, marker='o', ms=3, color='steelblue')
axes[0].axvspan(
    datetime.date.fromisoformat(OUTLIER_START),
    datetime.date.fromisoformat(OUTLIER_END),
    color='red', alpha=0.15, label='Outlier window'
)
axes[0].set_title("Before treatment")
axes[0].legend(fontsize=9)

# ── treat ────────────────────────────────────────────────────────────────────
train_clean = train_series.copy()

outlier_mask = (
    (train_clean.index >= datetime.date.fromisoformat(OUTLIER_START)) &
    (train_clean.index <= datetime.date.fromisoformat(OUTLIER_END))
)

train_clean[outlier_mask] = np.nan
train_clean = train_clean.interpolate(method='time')

assert train_clean.isna().sum() == 0, "NaNs still present after interpolation"

print(f"Points treated    : {outlier_mask.sum()}")
print(f"Dates affected    : {train_series[outlier_mask].index.tolist()}")
print("\nBefore vs After:")
print(pd.DataFrame({
    "original": train_series[outlier_mask],
    "treated":  train_clean[outlier_mask]
}))

# ── visualise after ──────────────────────────────────────────────────────────
axes[1].plot(train_clean.index, train_clean.values, marker='o', ms=3, color='green')
axes[1].axvspan(
    datetime.date.fromisoformat(OUTLIER_START),
    datetime.date.fromisoformat(OUTLIER_END),
    color='green', alpha=0.15, label='Treated window'
)
axes[1].set_title("After treatment")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

# ── build full_clean — substitute treated values, keep test untouched ────────
full_clean       = data_df[TARGET_METRIC].copy()
full_clean.index = pd.to_datetime(full_clean.index).date

for date in train_clean[outlier_mask].index:
    full_clean[date] = train_clean[date]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SARIMA Grid Search
# ══════════════════════════════════════════════════════════════════════════════

def univariate_sarima_grid_search(
    series,
    train_range,
    test_range,
    p_range=(0, 1, 2),
    d_range=(0, 1),
    q_range=(0, 1, 2),
    P_range=(0, 1),
    D_range=(0, 1),
    Q_range=(0, 1),
    m_values=(6, 12),
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

    order_combos    = list(itertools.product(p_range, d_range, q_range))
    seasonal_combos = list(itertools.product(P_range, D_range, Q_range))
    total           = (len(order_combos) * len(seasonal_combos) *
                       len(m_values) * len(trend_options))

    print(f"\nTrain points      : {len(train_s)}  ({t0} → {t1})")
    print(f"Test  points      : {len(test_s)}   ({s0} → {s1})")
    print(f"Order combos      : {len(order_combos)}")
    print(f"Seasonal combos   : {len(seasonal_combos)}")
    print(f"m values          : {list(m_values)}")
    print(f"Trend options     : {len(trend_options)}")
    print(f"Total fits        : {total * len(test_s):,}  ({total} combos × {len(test_s)} test steps)\n")

    # reasonable bounds for clipping explosive predictions
    reasonable_min = float(train_s.mean() - 4 * train_s.std())
    reasonable_max = float(train_s.mean() + 4 * train_s.std())

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="SARIMA grid search")

    for order in order_combos:
        for seasonal_pdq in seasonal_combos:
            for m in m_values:

                if len(train_s) < 2 * m:
                    pbar.update(len(trend_options))
                    continue

                seasonal_order = (*seasonal_pdq, m)

                for trend in trend_options:
                    pbar.update(1)
                    try:
                        predictions = []
                        history_y   = train_s.copy()

                        for i in range(len(test_s)):
                            model = SARIMAX(
                                history_y,
                                order=order,
                                seasonal_order=seasonal_order,
                                trend=trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)

                            pred = model.get_forecast(steps=1).predicted_mean.iloc[0]

                            # ── clip explosive predictions ────────────────────
                            if (np.isnan(pred) or np.isinf(pred) or
                                    pred < reasonable_min or pred > reasonable_max):
                                pred = float(history_y.iloc[-1])

                            predictions.append(pred)

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
                            "smape":          smape(yt, yp),
                            "mape":           mape_pct(yt, yp),
                            "mae":            mae(yt, yp),
                            "rmse":           rmse(yt, yp),
                            "order":          order,
                            "seasonal_order": seasonal_order,
                            "m":              m,
                            "trend":          trend,
                            "n_train":        len(train_s),
                            "n_test":         len(test_s),
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
                            "order": order,
                            "seasonal_order": seasonal_order,
                            "m": m, "trend": trend,
                            "error": str(e)[:120],
                        })

    pbar.close()

    out = pd.DataFrame(results)

    if out.empty or out[rank_by].isna().all():
        print("⚠ All combinations failed.")
        if "error" in out.columns:
            print(out["error"].value_counts().head(5))
        return out, best

    out = out.sort_values(rank_by, na_position="last").reset_index(drop=True)

    print(f"\n✓ Best {rank_by.upper()}   : {best.get(rank_by,       np.nan):.4f}")
    print(f"  MAPE             : {best.get('mape',           np.nan):.4f}%")
    print(f"  SMAPE            : {best.get('smape',          np.nan):.4f}%")
    print(f"  MAE              : {best.get('mae',            np.nan):>14,.0f}")
    print(f"  RMSE             : {best.get('rmse',           np.nan):>14,.0f}")
    print(f"  Order SARIMA     : {best.get('order')}")
    print(f"  Seasonal order   : {best.get('seasonal_order')}")
    print(f"  m (period)       : {best.get('m')}")
    print(f"  Trend            : {best.get('trend')}")

    return out.head(top_k), best


top, best = univariate_sarima_grid_search(
    series        = full_clean,                        # ← cleaned series
    train_range   = (first_input_date, last_input_date),
    test_range    = (first_test_date,  last_test_date),
    p_range       = (0, 1, 2),
    d_range       = (0, 1),
    q_range       = (0, 1, 2),
    P_range       = (0, 1),
    D_range       = (0, 1),
    Q_range       = (0, 1),
    m_values      = (6, 12),
    trend_options = ("n",),
    rank_by       = "smape",
    top_k         = 20,
)

top[["smape", "mape", "mae", "rmse", "order", "seasonal_order", "m", "trend"]]


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

order          = best['order']
seasonal_order = best['seasonal_order']
trend          = best['trend']

for i in range(len(test_s)):
    model = SARIMAX(
        history_y,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    pred = model.get_forecast(steps=1).predicted_mean.iloc[0]

    # clip explosive predictions — same logic as grid search
    if (np.isnan(pred) or np.isinf(pred) or
            pred < reasonable_min or pred > reasonable_max):
        pred = float(history_y.iloc[-1])

    predictions.append(pred)

    history_y = pd.concat([
        history_y,
        pd.Series([pred], index=[test_s.index[i]])
    ])

# ── results ──────────────────────────────────────────────────────────────────
sarima_fc = pd.Series(predictions, index=test_s.index)

actual    = test_s.values
fc        = np.array(predictions)
mask      = actual != 0
yt, yp    = actual[mask], fc[mask]

print(f"\n{'='*50}")
print(f"  SARIMA Inference Results")
print(f"{'='*50}")
print(f"  Order          : {order}")
print(f"  Seasonal order : {seasonal_order}")
print(f"  Trend          : {trend}")
print(f"  MAPE           : {mape_pct(yt, yp):.4f}%")
print(f"  SMAPE          : {smape(yt, yp):.4f}%")
print(f"  MAE            : {mae(yt, yp):>14,.0f}")
print(f"  RMSE           : {rmse(yt, yp):>14,.0f}")
print(f"{'='*50}")

# ── plot ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(test_s.index, test_s.values,  label='Actual',     color='steelblue', marker='o', ms=4)
plt.plot(sarima_fc.index, sarima_fc.values, label='SARIMA forecast', color='red',   marker='o', ms=4)
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.title(f"SARIMA{order}×{seasonal_order} — {TARGET_METRIC}")
plt.legend()
plt.tight_layout()
plt.show()
