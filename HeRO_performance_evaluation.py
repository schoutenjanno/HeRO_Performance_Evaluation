import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================
# PARAMETERS
# ==============================
FIXED_THRESHOLD        = 2
REFERENCE_HOURS        = 24
LAG_HOURS              = 12
TRIGGER_DELTA          = 2
DETECTION_WINDOW_HOURS = 24
ABX_DAYS               = 2
BURNIN_HOURS           = 12
START_DATE             = pd.Timestamp('2024-01-01 00:00')
np.random.seed(42)

MAX_THRESHOLD          = 7    # thresholds cannot exceed this value

# ==============================
# BUILD TIMESTAMPS (10 days, hourly)
# ==============================
timestamps = pd.date_range(start=START_DATE, periods=240, freq='1h')
t = np.arange(240)  # hours since start

BURNIN_CUTOFF = START_DATE + pd.Timedelta(hours=BURNIN_HOURS)

# Culture draw at hour 168 (end of day 7)
CULTURE_TIME = START_DATE + pd.Timedelta(hours=168)
WINDOW_START = CULTURE_TIME - pd.Timedelta(hours=DETECTION_WINDOW_HOURS)
# Clip detection window so it does not reach into the burn-in period
WINDOW_START = max(WINDOW_START, BURNIN_CUTOFF)
ABX_END      = CULTURE_TIME + pd.Timedelta(days=ABX_DAYS)

# ==============================
# SIGNAL 1: fixed threshold demo
# FP bump on day 5 (hour 102), TP bump on day 7 (hour 156)
# ==============================
fixed_scores = (
    1.0
    + 0.4 * np.sin(2 * np.pi * t / 12)
    + 1.4 * np.exp(-0.5 * ((t -  102) /  6) ** 2)   # FP: exceeds 2, outside exclusion window
    + 1.8 * np.exp(-0.5 * ((t - 156) /  8) ** 2)   # TP: exceeds 2 inside detection window
    + 0.15 * np.sin(t * 7.3)
)
fixed_scores = np.clip(fixed_scores, 0.1, 6.5)

# Cap fixed threshold at MAX_THRESHOLD
effective_fixed_threshold = min(FIXED_THRESHOLD, MAX_THRESHOLD)

# ==============================
# SIGNAL 2: dynamic threshold demo
# Stable plateau ~3.0; FP spike day 5 (hour 102), TP sigmoid jump day 7 (hour 156)
# ==============================
dynamic_scores = (
    3.0
    + 0.3 * np.sin(2 * np.pi * t / 16)
    + 2.6 * np.exp(-0.5 * ((t -  102) /  5) ** 2)   # FP: spike outside exclusion window
    + 2.5 / (1 + np.exp(-1.2 * (t - 158)))          # TP: sigmoid jump inside detection window
    + np.where(t > 168, -0.5 * (1 - np.exp(-(t - 168) / 24)), 0.0)
    + 0.12 * np.sin(t * 5.1)
)
dynamic_scores = np.clip(dynamic_scores, 0.1, 6.5)

# ==============================
# LAGGED DYNAMIC THRESHOLD
# All data (including burn-in) is used for threshold computation so the
# baseline is well-calibrated by the time performance evaluation starts.
# ==============================
dyn_series      = pd.Series(dynamic_scores, index=timestamps)
trailing_median = dyn_series.rolling(f'{REFERENCE_HOURS}h', min_periods=1).median()

dyn_thresh = trailing_median + TRIGGER_DELTA
dyn_thresh.index = trailing_median.index + pd.Timedelta(hours=LAG_HOURS)

first_valid = dyn_thresh.iloc[0]
filler = pd.Series(
    first_valid,
    index=pd.date_range(start=timestamps[0], periods=LAG_HOURS, freq='1h')
)
dyn_thresh = pd.concat([filler, dyn_thresh])
dyn_thresh = dyn_thresh[~dyn_thresh.index.duplicated(keep='last')]
dyn_thresh = dyn_thresh.reindex(timestamps, method='nearest')

# Cap dynamic threshold at MAX_THRESHOLD
dyn_thresh = dyn_thresh.clip(upper=MAX_THRESHOLD)

# ==============================
# MASKS
# ==============================
inside_abx   = (timestamps >= CULTURE_TIME) & (timestamps < ABX_END)
in_det_win   = (timestamps >= WINDOW_START) & (timestamps <= CULTURE_TIME)
# outside_excl excludes burn-in, detection window, and antibiotic window
outside_excl = (
    (timestamps >= BURNIN_CUTOFF) &
    ((timestamps < WINDOW_START) | (timestamps >= ABX_END))
)

# ==============================
# DOT COLORS
# ==============================
def dot_colors(scores, thresh_vals, inside_abx_mask):
    above = scores > thresh_vals
    return np.where(inside_abx_mask, 'grey', np.where(above, 'red', 'blue'))

fixed_colors   = dot_colors(fixed_scores,   effective_fixed_threshold, inside_abx)
dynamic_colors = dot_colors(dynamic_scores, dyn_thresh.values,         inside_abx)

# ==============================
# DETECTION (TP / FN)
# Culture at hour 168 is well after burn-in (hour 72), so it is included.
# ==============================
fixed_detected = (fixed_scores[in_det_win] > effective_fixed_threshold).any()
dyn_detected   = (dynamic_scores[in_det_win] > dyn_thresh.values[in_det_win]).any()

# ==============================
# TN / FP DAYS (outside exclusion window, outside burn-in)
# ==============================
dates = pd.Series(pd.to_datetime(timestamps).date, index=timestamps)

def tn_fp_days(scores, thresh_vals, outside_mask):
    df = pd.DataFrame({'score': scores, 'thresh': thresh_vals,
                       'date': dates, 'outside': outside_mask},
                      index=timestamps)
    df = df[df['outside']]
    df['above'] = df['score'] > df['thresh']
    day_above = df.groupby('date')['above'].any()
    return day_above.index[~day_above], day_above.index[day_above]

fixed_TN,   fixed_FP   = tn_fp_days(fixed_scores,   np.full(240, effective_fixed_threshold), outside_excl)
dynamic_TN, dynamic_FP = tn_fp_days(dynamic_scores, dyn_thresh.values,                       outside_excl)

# ==============================
# METRICS
# Day-level confusion matrix values:
#   TP = 1 if detected within detection window, else 0  (one event per simulation)
#   FN = 1 - TP
#   FP / TN = counts of false-positive / true-negative days outside exclusion window
# ==============================
def compute_metrics(detected, TN_days, FP_days):
    TP = int(detected)
    FN = 1 - TP
    FP = len(FP_days)
    TN = len(TN_days)

    precision   = TP / (TP + FP)  if (TP + FP) > 0 else float('nan')
    recall      = TP / (TP + FN)  # always defined (denom = 1)
    specificity = TN / (TN + FP)  if (TN + FP) > 0 else float('nan')
    npv         = TN / (TN + FN)  if (TN + FN) > 0 else float('nan')

    return dict(TP=TP, FN=FN, FP=FP, TN=TN,
                precision=precision, recall=recall,
                specificity=specificity, npv=npv)

fixed_metrics   = compute_metrics(fixed_detected,   fixed_TN,   fixed_FP)
dynamic_metrics = compute_metrics(dyn_detected,     dynamic_TN, dynamic_FP)

def print_metrics(label, m):
    print(f"\n{'='*46}")
    print(f"  {label}")
    print(f"{'='*46}")
    print(f"  Confusion matrix:  TP={m['TP']}  FN={m['FN']}  FP={m['FP']}  TN={m['TN']}")
    print(f"  {'Metric':<30} {'Value':>8}")
    print(f"  {'-'*38}")
    for name, key in [
        ('Precision (PPV)',            'precision'),
        ('Recall (Sensitivity)',       'recall'),
        ('Specificity',                'specificity'),
        ('Negative Predictive Value',  'npv'),
    ]:
        val = m[key]
        if val != val:   # NaN check
            print(f"  {name:<30} {'N/A':>8}")
        else:
            print(f"  {name:<30} {val:>8.3f}")
    print()

print_metrics(f"Fixed threshold  (score > {effective_fixed_threshold})", fixed_metrics)
print_metrics(f"Dynamic threshold  ({REFERENCE_HOURS}h lagged median + {TRIGGER_DELTA}, capped at {MAX_THRESHOLD})", dynamic_metrics)

# ==============================
# PLOT
# ==============================
fig, (ax_fixed, ax_dyn) = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True,
    gridspec_kw={'hspace': 0.42}
)

for ax, scores, colors, thresh, TN_days, FP_days, detected, title in [
    (ax_fixed,
     fixed_scores, fixed_colors, None,
     fixed_TN, fixed_FP, fixed_detected,
     f'Fixed threshold  (score > {effective_fixed_threshold})'),
    (ax_dyn,
     dynamic_scores, dynamic_colors, dyn_thresh,
     dynamic_TN, dynamic_FP, dyn_detected,
     f'Dynamic threshold  ({REFERENCE_HOURS} h lagged median + {TRIGGER_DELTA}, lag = {LAG_HOURS} h, capped at {MAX_THRESHOLD})'),
]:
    # --- burn-in shading ---
    ax.axvspan(START_DATE, BURNIN_CUTOFF,
               color='lightgrey', alpha=0.4, zorder=0,
               label=f'Burn-in ({BURNIN_HOURS} h, excluded from analysis)')

    # --- detection window shading ---
    shade = 'green' if detected else 'red'
    ax.axvspan(WINDOW_START, CULTURE_TIME,
               color=shade, alpha=0.15, zorder=0,
               label='Detection window (TP)' if detected else 'Detection window (FN)')

    # --- scatter dots ---
    non_abx_mask = ~inside_abx
    ax.scatter(timestamps[non_abx_mask], scores[non_abx_mask],
               s=10, c=np.array(colors)[non_abx_mask], zorder=3, label='HeRO score')
    ax.scatter(timestamps[inside_abx], scores[inside_abx],
               s=10, color='grey', zorder=3, label='HeRO score (antibiotic window)')

    # --- threshold line ---
    if thresh is None:
        ax.axhline(y=effective_fixed_threshold, color='black', linestyle='--',
                   linewidth=1.4, label=f'Threshold = {effective_fixed_threshold}', zorder=4)
    else:
        ax.plot(thresh.index, thresh.values, color='black', linestyle='--',
                linewidth=1.4, label='Dynamic threshold (lagged, capped at 7)', zorder=4)

    # --- culture line ---
    ax.axvline(x=CULTURE_TIME, color='darkblue', linestyle=':',
               linewidth=2, label='Culture draw', zorder=5)

    # --- TN day markers (green | at y=0) ---
    if len(TN_days):
        tn_x = pd.to_datetime(TN_days) + pd.Timedelta(hours=12)
        ax.scatter(tn_x, np.zeros(len(tn_x)),
                   color='green', marker='|', s=300, linewidths=3,
                   zorder=6)
        ax.plot([], [], color='green', marker='|', markersize=8,
                linewidth=0, label='True negative day')

    # --- FP day markers (red | at y=0) ---
    if len(FP_days):
        fp_x = pd.to_datetime(FP_days) + pd.Timedelta(hours=12)
        ax.scatter(fp_x, np.zeros(len(fp_x)),
                   color='red', marker='|', s=300, linewidths=3,
                   zorder=6)
        ax.plot([], [], color='red', marker='|', markersize=8,
                linewidth=0, label='False positive day')

    # --- formatting ---
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=True, loc='upper left')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('HeRO score')
    ax.set_ylim(0, 7.1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.setp(ax_dyn.xaxis.get_majorticklabels(), rotation=45, ha='right')
fig.suptitle('Simulated HeRO score — fixed vs dynamic threshold (lagged baseline)', fontsize=12)
plt.tight_layout()
plt.savefig('hero_simulation_plot.png', dpi=150, bbox_inches='tight')
plt.show()
