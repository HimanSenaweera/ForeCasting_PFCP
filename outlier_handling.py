OUTLIER_REPLACEMENTS = {
    "2024-02-29": "2024-01-31",
    "2024-04-30": "2024-03-31",
    "2024-05-31": "2024-06-30",
}

data_df_clean = data_df.copy()
data_df_clean.index = pd.to_datetime(data_df_clean.index).date

print(f"\n{'─'*55}")
print(f"  {'Date':<15} {'Original':>14} {'Source':>14} {'New Value':>14}")
print(f"  {'─'*55}")

for bad_date, good_date in OUTLIER_REPLACEMENTS.items():
    bad  = datetime.date.fromisoformat(bad_date)
    good = datetime.date.fromisoformat(good_date)

    if bad in data_df_clean.index and good in data_df_clean.index:
        orig = data_df_clean.loc[bad, TARGET_METRIC]
        new  = data_df_clean.loc[good, TARGET_METRIC]
        data_df_clean.loc[bad, TARGET_METRIC] = new
        print(f"  {bad_date:<15} {orig:>14,.0f} {good_date:>14} {new:>14,.0f}")
    else:
        print(f"  {bad_date:<15} — skipped (date not found)")

print(f"{'─'*55}")

# restore DatetimeIndex
data_df_clean.index = pd.to_datetime(data_df_clean.index)
print(f"\n✅ Done — use data_df_clean going forward")

# ── Plot ──────────────────────────────────────────────────────────────────────
orig_series  = data_df[TARGET_METRIC].dropna()
clean_series = data_df_clean[TARGET_METRIC].dropna()

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

for ax, series, label, color in zip(
    axes,
    [orig_series, clean_series],
    ['BEFORE treatment', 'AFTER treatment'],
    ['steelblue', 'green'],
):
    ax.plot(pd.to_datetime(series.index), series.values,
            marker='o', linewidth=2, color=color,
            markersize=4, label='Actual')

    # mark outlier dates
    for bad_date in OUTLIER_REPLACEMENTS:
        ax.axvline(pd.to_datetime(bad_date),
                   color='red' if 'BEFORE' in label else 'green',
                   linestyle=':', linewidth=1.5, alpha=0.8)

    y_min   = series.min()
    y_max   = series.max()
    padding = (y_max - y_min) * 0.05

    ax.set_ylim(y_min - padding, y_max + padding)
    ax.fill_between(pd.to_datetime(series.index), series.values,
                    y2=y_min - padding, alpha=0.08, color=color)
    ax.set_title(f"[{label}]  —  {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})",
                 fontsize=12)
    ax.set_ylabel('Outstanding Balances ($)')
    ax.legend(fontsize=9)
    ax.annotate(f"n = {len(series)}", xy=(0.01, 0.90),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

axes[1].set_xlabel('Date')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle(f'Outlier Treatment — {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
