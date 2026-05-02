import itertools, datetime, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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


def univariate_arima_cv(
    series,                 # pd.Series with datetime index
    train_range,            # (first_train_date, last_train_date) ISO strings
    test_range,             # (first_test_date,  last_test_date)  ISO strings
    n_splits=4,             # number of CV folds
    p_range=(0, 1, 2, 3),
    d_range=(0, 1),
    q_range=(0, 1, 2, 3),
    trend_options=("n",),
    rank_by="smape",        # "smape" | "mape" | "mae" | "rmse"
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

    # reasonable bounds to clip explosive predictions
    reasonable_min = float(train_s.mean() - 4 * train_s.std())
    reasonable_max = float(train_s.mean() + 4 * train_s.std())

    # ── 2. CV folds ──────────────────────────────────────────────────────────
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    order_combos = list(itertools.product(p_range, d_range, q_range))
    total        = len(order_combos) * len(trend_options)

    print(f"Train points    : {len(train_s)}  ({t0} → {t1})")
    print(f"Test  points    : {len(test_s)}   ({s0} → {s1})")
    print(f"CV folds        : {n_splits}")
    print(f"Order combos    : {len(order_combos)}")
    print(f"Trend options   : {len(trend_options)}")
    print(f"Total fits      : {total * n_splits:,}  ({total} combos × {n_splits} folds)\n")

    # ── print fold sizes ──────────────────────────────────────────────────────
    print("Fold breakdown:")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_s)):
        print(f"  Fold {fold+1}: train={len(tr_idx)} pts "
              f"({train_s.index[tr_idx[0]]} → {train_s.index[tr_idx[-1]]})  "
              f"val={len(val_idx)} pts "
              f"({train_s.index[val_idx[0]]} → {train_s.index[val_idx[-1]]})")
    print()

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="ARIMA CV grid search")

    for order in order_combos:
        for trend in trend_options:
            pbar.update(1)
            fold_scores = {
                "smape": [], "mape": [], "mae": [], "rmse": []
            }
            all_folds_ok = True

            # ── walk-forward across each CV fold ─────────────────────────────
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_s)):

                if len(tr_idx) < min_train:
                    all_folds_ok = False
                    continue

                fold_train = train_s.iloc[tr_idx]
                fold_val   = train_s.iloc[val_idx]

                try:
                    predictions = []
                    history_y   = fold_train.copy()

                    for i in range(len(fold_val)):
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
                            pd.Series([pred], index=[fold_val.index[i]])
                        ])

                    actual = fold_val.values
                    fc     = np.array(predictions)
                    mask   = actual != 0

                    if not mask.any():
                        continue

                    yt, yp = actual[mask], fc[mask]

                    fold_scores["smape"].append(smape(yt, yp))
                    fold_scores["mape"].append(mape_pct(yt, yp))
                    fold_scores["mae"].append(mae(yt, yp))
                    fold_scores["rmse"].append(rmse(yt, yp))

                except Exception:
                    all_folds_ok = False
                    continue

            if not fold_scores[rank_by]:
                continue

            # ── average across folds ─────────────────────────────────────────
            row = {
                "smape":      np.mean(fold_scores["smape"]),
                "mape":       np.mean(fold_scores["mape"]),
                "mae":        np.mean(fold_scores["mae"]),
                "rmse":       np.mean(fold_scores["rmse"]),
                "smape_std":  np.std(fold_scores["smape"]),
                "mape_std":   np.std(fold_scores["mape"]),
                "order":      order,
                "trend":      trend,
                "n_folds":    len(fold_scores[rank_by]),
                "fold_scores": fold_scores[rank_by],
                "n_train":    len(train_s),
                "n_test":     len(test_s),
            }
            results.append(row)

            if row[rank_by] < best.get(rank_by, np.inf):
                best = {**row}

    pbar.close()

    out = pd.DataFrame(results)

    if out.empty or out[rank_by].isna().all():
        print("⚠ All combinations failed.")
        return out, best

    out = out.sort_values(rank_by, na_position="last").reset_index(drop=True)

    print(f"\n✓ Best {rank_by.upper()}     : {best.get(rank_by,       np.nan):.4f}")
    print(f"  Std across folds : {best.get(f'{rank_by}_std',  np.nan):.4f}")
    print(f"  MAPE             : {best.get('mape',            np.nan):.4f}%")
    print(f"  SMAPE            : {best.get('smape',           np.nan):.4f}%")
    print(f"  MAE              : {best.get('mae',             np.nan):>14,.0f}")
    print(f"  RMSE             : {best.get('rmse',            np.nan):>14,.0f}")
    print(f"  Order ARIMA      : {best.get('order')}")
    print(f"  Trend            : {best.get('trend')}")
    print(f"  Folds completed  : {best.get('n_folds')}")
    print(f"  Fold scores      : {[f'{s:.2f}' for s in best.get('fold_scores', [])]}")

    return out.head(top_k), best


# ── Run ───────────────────────────────────────────────────────────────────────
top, best = univariate_arima_cv(
    series        = data_df_clean[TARGET_METRIC],    # ← cleaned series
    train_range   = (first_input_date, last_input_date),
    test_range    = (first_test_date,  last_test_date),
    n_splits      = 4,
    p_range       = (0, 1, 2, 3),
    d_range       = (0, 1),
    q_range       = (0, 1, 2, 3),
    trend_options = ("n", "c"),
    rank_by       = "smape",
    top_k         = 20,
)

top[["smape", "smape_std", "mape", "mae", "rmse", "order", "trend", "n_folds"]]
