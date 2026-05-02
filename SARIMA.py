import itertools, datetime, warnings
import numpy as np
import pandas as pd
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


def univariate_sarima_grid_search(
    series,
    train_range,
    test_range,
    p_range=(0, 1, 2),
    d_range=(0, 1),
    q_range=(0, 1, 2),
    P_range=(0, 1),           # seasonal AR order
    D_range=(0, 1),           # seasonal differencing
    Q_range=(0, 1),           # seasonal MA order
    m_values=(6, 12),         # seasonal periods to search — based on your periodogram: 6 and 12
    trend_options=("n",),
    rank_by="smape",
    top_k=20,
    min_train=20,
):
    # ── 1. Prepare series ────────────────────────────────────────────────────
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

    # ── 2. Search space ──────────────────────────────────────────────────────
    order_combos    = list(itertools.product(p_range, d_range, q_range))
    seasonal_combos = list(itertools.product(P_range, D_range, Q_range))
    total           = (len(order_combos) * len(seasonal_combos) *
                       len(m_values) * len(trend_options))

    print(f"Train points      : {len(train_s)}  ({t0} → {t1})")
    print(f"Test  points      : {len(test_s)}   ({s0} → {s1})")
    print(f"Order combos      : {len(order_combos)}")
    print(f"Seasonal combos   : {len(seasonal_combos)}")
    print(f"m values          : {list(m_values)}")
    print(f"Trend options     : {len(trend_options)}")
    print(f"Total fits        : {total * len(test_s):,}  ({total} combos × {len(test_s)} test steps)\n")

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="SARIMA grid search")

    for order in order_combos:
        for seasonal_order_pdq in seasonal_combos:
            for m in m_values:

                # ── skip if not enough data for this seasonal period ─────────
                # need at least 2 full seasonal cycles in training
                if len(train_s) < 2 * m:
                    pbar.update(len(trend_options))
                    continue

                seasonal_order = (*seasonal_order_pdq, m)   # (P, D, Q, m)

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
                            predictions.append(pred)

                            # roll forward with prediction
                            history_y = pd.concat([
                                history_y,
                                pd.Series([pred], index=[test_s.index[i]])
                            ])

                        # ── score ────────────────────────────────────────────
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
        print("⚠ All combinations failed. Errors:")
        if "error" in out.columns:
            print(out["error"].value_counts().head(5))
        return out, best

    out = out.sort_values(rank_by, na_position="last").reset_index(drop=True)

    print(f"\n✓ Best {rank_by.upper()}   : {best.get(rank_by,       np.nan):.4f}")
    print(f"  MAPE             : {best.get('mape',           np.nan):.4f}%")
    print(f"  SMAPE            : {best.get('smape',          np.nan):.4f}%")
    print(f"  MAE              : {best.get('mae',            np.nan):>14,.0f}")
    print(f"  RMSE             : {best.get('rmse',           np.nan):>14,.0f}")
    print(f"  Order ARIMA      : {best.get('order')}")
    print(f"  Seasonal order   : {best.get('seasonal_order')}")
    print(f"  m (period)       : {best.get('m')}")
    print(f"  Trend            : {best.get('trend')}")

    return out.head(top_k), best


# ── Run ──────────────────────────────────────────────────────────────────────
top, best = univariate_sarima_grid_search(
    series        = data_df[TARGET_METRIC],
    train_range   = (first_input_date, last_input_date),
    test_range    = (first_test_date,  last_test_date),
    p_range       = (0, 1, 2),
    d_range       = (0, 1),
    q_range       = (0, 1, 2),
    P_range       = (0, 1),
    D_range       = (0, 1),
    Q_range       = (0, 1),
    m_values      = (6, 12),   # periodogram said 6.6m dominant — test both
    trend_options = ("n",),
    rank_by       = "smape",
    top_k         = 20,
)

top[["smape", "mape", "mae", "rmse", "order", "seasonal_order", "m", "trend"]]
