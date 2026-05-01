def arimax_grid_search_rolling(
    base_df,
    features,               # exog feature names (raw, unlagged)
    target_col,
    train_range,
    test_range,
    lags=(0, 1, 2, 3),      # applies to BOTH features AND target lag
    target_lags=(1, 2, 3),  # ← now searchable, 0 excluded (can't use same-period target)
    p_range=(0, 1, 2, 3),
    d_range=(0, 1),
    q_range=(0, 1, 2, 3),
    trend_options=("n", "c"),
    rank_by="smape",
    top_k=20,
    min_train=20,
):
    # ── 1. Build all lag columns once ────────────────────────────────────────
    work = base_df.copy()
    work.index = pd.to_datetime(work.index).date

    for feat in features:
        for L in lags:
            work[f"{feat}__lag{L}"] = work[feat].shift(L) if L > 0 else work[feat]

    # build ALL target lag columns upfront
    for L in target_lags:
        work[f"{target_col}__lag{L}"] = work[target_col].shift(L)

    t0, t1 = map(datetime.date.fromisoformat, train_range)
    s0, s1 = map(datetime.date.fromisoformat, test_range)

    # ── 2. Search spaces ─────────────────────────────────────────────────────
    lag_combos        = list(itertools.product(lags, repeat=len(features)))
    order_combos      = list(itertools.product(p_range, d_range, q_range))
    total             = (len(lag_combos) * len(target_lags) *
                         len(order_combos) * len(trend_options))

    print(f"Feature lag combos : {len(lag_combos)}")
    print(f"Target lag options : {list(target_lags)}")
    print(f"Order combos       : {len(order_combos)}")
    print(f"Trend options      : {len(trend_options)}")
    print(f"Total combos       : {total:,}\n")

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="ARIMAX grid search")

    for lag_combo in lag_combos:
        feat_exog_cols = [f"{feat}__lag{L}" for feat, L in zip(features, lag_combo)]

        for target_lag in target_lags:                    # ← now loops over target lags
            target_lag_col = f"{target_col}__lag{target_lag}"
            exog_cols      = feat_exog_cols + [target_lag_col]

            clean = work.dropna(subset=exog_cols + [target_col])
            idx   = pd.Index(clean.index)
            train = clean[(idx >= t0) & (idx <= t1)]
            test  = clean[(idx >= s0) & (idx <= s1)]

            if len(train) < min_train or len(test) == 0:
                pbar.update(len(order_combos) * len(trend_options))
                continue

            for order in order_combos:
                for trend in trend_options:
                    pbar.update(1)
                    try:
                        predictions = []
                        history_y   = train[target_col].astype(float).copy()
                        history_df  = train.copy()

                        for i in range(len(test)):

                            model = SARIMAX(
                                history_y,
                                exog=history_df[exog_cols].astype(float),
                                order=order,
                                trend=trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)

                            new_row = test.iloc[[i]].copy()

                            for col in feat_exog_cols:
                                new_row[col] = history_df[col].iloc[-1]

                            # target lag = look back `target_lag` steps in history
                            new_row[target_lag_col] = history_y.iloc[-target_lag]

                            pred = model.get_forecast(
                                steps=1,
                                exog=new_row[exog_cols].astype(float),
                            ).predicted_mean.iloc[0]

                            predictions.append(pred)

                            history_y = pd.concat([
                                history_y,
                                pd.Series([pred], index=[test.index[i]])
                            ])

                            new_row[target_col]     = pred
                            new_row[target_lag_col] = history_y.iloc[-target_lag - 1]
                            history_df = pd.concat([history_df, new_row])

                        # ── score ────────────────────────────────────────────
                        actual = test[target_col].astype(float).values
                        fc     = np.array(predictions)
                        mask   = actual != 0

                        if not mask.any():
                            continue

                        yt, yp = actual[mask], fc[mask]

                        row = {
                            "smape":      smape(yt, yp),
                            "mape":       mape_pct(yt, yp),
                            "mae":        mae(yt, yp),
                            "rmse":       rmse(yt, yp),
                            "order":      order,
                            "trend":      trend,
                            "target_lag": target_lag,     # ← now recorded
                            "lags":       dict(zip(features, lag_combo)),
                            "exog_cols":  exog_cols,
                            "n_train":    len(train),
                            "n_test":     len(test),
                        }
                        results.append(row)

                        if row[rank_by] < best.get(rank_by, np.inf):
                            best = {
                                **row,
                                "forecast":   fc,
                                "actual":     actual,
                                "test_index": test.index,
                            }

                    except Exception as e:
                        results.append({
                            "smape": np.nan, "mape": np.nan,
                            "mae":   np.nan, "rmse": np.nan,
                            "order": order,  "trend": trend,
                            "target_lag": target_lag,
                            "lags":  dict(zip(features, lag_combo)),
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

    print(f"\n✓ Best {rank_by.upper()} : {best.get(rank_by,  np.nan):.4f}")
    print(f"  MAPE           : {best.get('mape',  np.nan):.4f}%")
    print(f"  SMAPE          : {best.get('smape', np.nan):.4f}%")
    print(f"  MAE            : {best.get('mae',   np.nan):>14,.0f}")
    print(f"  RMSE           : {best.get('rmse',  np.nan):>14,.0f}")
    print(f"  Order ARIMAX   : {best.get('order')}")
    print(f"  Trend          : {best.get('trend')}")
    print(f"  Target lag     : {best.get('target_lag')}")   # ← now printed
    print(f"  Lags           : {best.get('lags')}")

    return out.head(top_k), best


# ── Run ──────────────────────────────────────────────────────────────────────
features = [
    "90+ DQ Rate",
    "Payment Rate",
    "Finance Charges Rate",
    "Expected Loss Roll Ave 3M",
]

top, best = arimax_grid_search_rolling(
    base_df       = data_df,
    features      = features,
    target_col    = TARGET_METRIC,
    train_range   = (first_input_date, last_input_date),
    test_range    = (first_test_date,  last_test_date),
    lags          = (0, 1, 2, 3),
    target_lags   = (1, 2, 3),        # ← pass whatever lags you want to test
    p_range       = (0, 1, 2, 3),
    d_range       = (0, 1),
    q_range       = (0, 1, 2, 3),
    trend_options = ("n", "c"),
    rank_by       = "smape",
    top_k         = 20,
)

top[["smape", "mape", "mae", "rmse", "order", "trend", "target_lag", "lags"]]
