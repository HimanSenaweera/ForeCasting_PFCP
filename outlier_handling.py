# ══════════════════════════════════════════════════════════════════════════════
# Outlier Treatment + Replot
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Define replacement mapping ────────────────────────────────────────────
OUTLIER_REPLACEMENTS = {
    "2024-01-31": "2024-05-31",
    "2024-02-29": "2024-05-31",
    "2024-03-31": "2024-05-31",
    "2024-04-30": "2024-05-31",
}

# ── 2. Apply to train_series only ────────────────────────────────────────────
train_clean = train_series.copy()

print(f"\n{'─'*55}")
print(f"  Outlier Treatment")
print(f"{'─'*55}")
print(f"  {'Date':<15} {'Original':>14} {'Source Date':>14} {'New Value':>14}")
print(f"  {'─'*55}")

for bad_date, good_date in OUTLIER_REPLACEMENTS.items():
    bad  = datetime.date.fromisoformat(bad_date)
    good = datetime.date.fromisoformat(good_date)
    if bad in train_clean.index and good in train_clean.index:
        orig  = train_clean[bad]
        new   = train_series[good]
        train_clean[bad] = new
        print(f"  {bad_date:<15} {orig:>14,.0f} {good_date:>14} {new:>14,.0f}")
    else:
        print(f"  {bad_date:<15} — skipped (date not in train_series)")

print(f"{'─'*55}\n")

# ── 3. Build full_series_clean ───────────────────────────────────────────────
full_clean       = data_df[TARGET_METRIC].copy()
full_clean.index = pd.to_datetime(full_clean.index).date

for bad_date in OUTLIER_REPLACEMENTS:
    d = datetime.date.fromisoformat(bad_date)
    if d in train_clean.index:
        full_clean[d] = train_clean[d]

full_series_clean = full_clean[
    (full_clean.index >= cutoff_lower)
].dropna()

print(f"  train_clean       : {len(train_clean)} points  ({train_clean.index[0]} → {train_clean.index[-1]})")
print(f"  full_series_clean : {len(full_series_clean)} points  ({full_series_clean.index[0]} → {full_series_clean.index[-1]})")
print(f"  Min / Max / Mean  : {train_clean.min():,.0f} / {train_clean.max():,.0f} / {train_clean.mean():,.0f}")

# ── 4. Plot before vs after (2x2 matching your style) ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex=False)

plot_config = [
    # (ax,         series,             label,                                    color)
    (axes[0][0], train_series,       '[PART A — TRAIN ONLY]  BEFORE',           PALETTE['raw']),
    (axes[0][1], full_series,        '[PART B — TRAIN + TEST]  BEFORE',         PALETTE['actual']),
    (axes[1][0], train_clean,        '[PART A — TRAIN ONLY]  AFTER',            PALETTE['raw']),
    (axes[1][1], full_series_clean,  '[PART B — TRAIN + TEST]  AFTER',          PALETTE['actual']),
]

for ax, series, label, color in plot_config:
    is_after = 'AFTER' in label

    ax.plot(pd.to_datetime(series.index), series.values,
            marker='o', linewidth=2, color=color, markersize=5,
            label='Actual (treated)' if is_after else 'Actual')

    ax.axvline(pd.to_datetime(LAST_TRAIN_DATE), color='grey',
               linestyle='--', linewidth=1.2, label='Train/Test cutoff')

    # mark outlier dates
    for bad_date in OUTLIER_REPLACEMENTS:
        d = datetime.date.fromisoformat(bad_date)
        if d in series.index:
            ax.axvline(pd.to_datetime(bad_date),
                       color='green' if is_after else 'red',
                       linestyle=':', linewidth=1.2, alpha=0.8)

    y_min    = series.min()
    y_max    = series.max()
    padding  = (y_max - y_min) * 0.05

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
