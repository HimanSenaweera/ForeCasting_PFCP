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

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=False)

plot_config = [
    (axes[0], data_df[TARGET_METRIC].dropna(),       'BEFORE',  PALETTE['raw']),
    (axes[1], data_df_clean[TARGET_METRIC].dropna(), 'AFTER',   PALETTE['raw']),
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
        if d in pd.to_datetime(series.index).date:
            ax.axvline(pd.to_datetime(bad_date),
                       color='green' if is_after else 'red',
                       linestyle=':', linewidth=1.2, alpha=0.8)

    y_min   = series.min()
    y_max   = series.max()
    padding = (y_max - y_min) * 0.05

    ax.set_ylim(y_min - padding, y_max + padding)
    ax.fill_between(pd.to_datetime(series.index), series.values,
                    y2=y_min - padding, alpha=0.12, color=color)

    ax.set_title(f"[{label}]  —  {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})")
    ax.set_ylabel('Outstanding Balances ($)')
    ax.set_xlabel('Date')
    ax.set_xticks(pd.to_datetime(series.index))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=9)
    ax.annotate(f"n = {len(series)}", xy=(0.01, 0.92),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.suptitle(f'Outlier Treatment — {TARGET_METRIC}  ({PORTFOLIO} / {SUB_PORTFOLIO})',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
