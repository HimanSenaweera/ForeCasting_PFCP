# ══════════════════════════════════════════════════════════════════════════════
# Outlier Treatment + Replot
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Define replacement mapping ────────────────────────────────────────────
# Use 2024-05-31 post-recovery value for all dip dates
OUTLIER_REPLACEMENTS = {
    "2024-01-31": "2024-05-31",
    "2024-02-29": "2024-05-31",
    "2024-03-31": "2024-05-31",
    "2024-04-30": "2024-05-31",
}

# ── 2. Apply to train_series only ────────────────────────────────────────────
train_clean = train_series.copy()

print("Outlier Treatment:")
print(f"{'Date':<15} {'Original':>15} {'Replaced With':>15} {'New Value':>15}")
print("-" * 60)

for bad_date, good_date in OUTLIER_REPLACEMENTS.items():
    bad  = datetime.date.fromisoformat(bad_date)
    good = datetime.date.fromisoformat(good_date)

    if bad in train_clean.index and good in train_clean.index:
        original_val = train_clean[bad]
        new_val      = train_series[good]
        train_clean[bad] = new_val
        print(f"{bad_date:<15} {original_val:>15,.0f} {good_date:>15} {new_val:>15,.0f}")
    else:
        print(f"{bad_date:<15} — date not found in train_series, skipping")

# ── 3. Build full_clean — only train window modified, test untouched ─────────
full_clean       = data_df[TARGET_METRIC].copy()
full_clean.index = pd.to_datetime(full_clean.index).date

for bad_date in OUTLIER_REPLACEMENTS:
    d = datetime.date.fromisoformat(bad_date)
    if d in train_clean.index:
        full_clean[d] = train_clean[d]

# rebuild full_series_clean for plotting (train + test)
idx = pd.Index(full_clean.index)
full_series_clean = full_clean[
    (idx >= datetime.date.fromisoformat(first_input_date)) &
    (idx <= datetime.date.fromisoformat(last_test_date))
].dropna()

print(f"\ntrain_clean  : {len(train_clean)} points")
print(f"full_clean   : {len(full_series_clean)} points")

# ── 4. Plot before vs after ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex=False)

# row 0 — BEFORE
for ax, series, label, color in zip(
    [axes[0][0], axes[0][1]],
    [train_series, full_series],
    ['[PART A — TRAIN ONLY]  BEFORE', '[PART B — TRAIN + TEST]  BEFORE'],
    [PALETTE['raw'], PALETTE['actual']],
):
    ax.plot(pd.to_datetime(series.index), series.values,
            marker='o', linewidth=2, color=color, markersize=5, label='Actual')
    ax.axvline(pd.to_datetime(LAST_TRAIN_DATE), color='grey',
               linestyle='--', linewidth=1.2, label='Train/Test cutoff')
    for bad_date in OUTLIER_REPLACEMENTS:
        ax.axvline(pd.to_datetime(bad_date), color='red',
                   linestyle=':', linewidth=1, alpha=0.7)
    y_min, y_max = series.min(), series.max()
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.fill_between(pd.to_datetime(series.index), series.values,
                    y2=y_min - padding, alpha=0.12, color=color)
    ax.set_title(f"{label}  —  {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})")
    ax.set_ylabel('Outstanding Balances ($)')
    ax.set_xlabel('Date')
    ax.set_xticks(pd.to_datetime(series.index))
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.annotate(f"n = {len(series)}", xy=(0.01, 0.92),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# row 1 — AFTER
for ax, series, label, color in zip(
    [axes[1][0], axes[1][1]],
    [train_clean, full_series_clean],
    ['[PART A — TRAIN ONLY]  AFTER', '[PART B — TRAIN + TEST]  AFTER'],
    [PALETTE['raw'], PALETTE['actual']],
):
    ax.plot(pd.to_datetime(series.index), series.values,
            marker='o', linewidth=2, color=color, markersize=5, label='Actual (treated)')
    ax.axvline(pd.to_datetime(LAST_TRAIN_DATE), color='grey',
               linestyle='--', linewidth=1.2, label='Train/Test cutoff')
    for bad_date in OUTLIER_REPLACEMENTS:
        ax.axvline(pd.to_datetime(bad_date), color='green',
                   linestyle=':', linewidth=1, alpha=0.7)
    y_min, y_max = series.min(), series.max()
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.fill_between(pd.to_datetime(series.index), series.values,
                    y2=y_min - padding, alpha=0.12, color=color)
    ax.set_title(f"{label}  —  {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})")
    ax.set_ylabel('Outstanding Balances ($)')
    ax.set_xlabel('Date')
    ax.set_xticks(pd.to_datetime(series.index))
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.annotate(f"n = {len(series)}", xy=(0.01, 0.92),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.suptitle('Outlier Treatment — Before vs After', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
