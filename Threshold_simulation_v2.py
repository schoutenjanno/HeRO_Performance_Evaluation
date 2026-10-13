"""
HeRO monitoring pipeline, illustrated on synthetic data.

Purpose: a self-contained, reproducible supplement showing how the fixed and
patient specific (dynamic) thresholds, the detection window logic, and the
TN/FP day counting work in the main validation pipeline. This is NOT a
validation run, all scores are synthetic and the "patients" below are
illustrative constructs, not real HeRO data.

Constants that intentionally match the production scripts (main validation
and subgroup analysis): FIXED_THRESHOLD, REFERENCE_HOURS, TRIGGER_DELTA,
LAG_HOURS, MAX_THRESHOLD, DETECTION_WINDOW_HOURS.

Deliberate simplifications versus the production scripts, called out here so
a reader does not mistake this for the full inclusion logic:
- Burn-in only reflects BURNIN_HERO_HOURS. The production burn-in is the
  later of that and 72h postnatal age, which requires a birth time this
  single synthetic stay does not have.
- Antibiotic therapy duration (ABX_DAYS) is fixed, the production pipeline
  reads a per-episode duration from the culture dataset.
- Panels A and B use different synthetic signals, chosen to illustrate each
  threshold's behavior. They are not the same patient and should not be
  compared directly. In the production pipeline both thresholds are applied
  to the same HRC index per patient.
- The final section is a schematic illustration of the stratified bootstrap
  mechanics on a small sim population, it is not a real validation bootstrap
  and its numbers are not meaningful, only the procedure is.
- The AUC calculations below (single-stay and schematic-population) are
  likewise illustrative only, showing how the raw-score AUC in the main
  validation script is constructed, not a real discrimination estimate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# ==============================
# PARAMETERS
# ==============================
FIXED_THRESHOLD        = 2
REFERENCE_HOURS        = 24
LAG_HOURS              = 12
TRIGGER_DELTA          = 2
DETECTION_WINDOW_HOURS = 24
ABX_DAYS               = 5      # minimum antibiotic course length, fixed for illustration
BURNIN_HOURS           = 12     # illustrates BURNIN_HERO_HOURS only, see module docstring
START_DATE             = pd.Timestamp('2024-01-01 00:00')
MAX_THRESHOLD          = 7

SEED = 42
rng = np.random.default_rng(SEED)  # used below for signal noise and the bootstrap illustration

# Output goes next to this script by default, no institutional paths in a
# reproducibility supplement
OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = OUTPUT_DIR / 'hero_simulation_plot.png'

# ==============================
# BUILD TIMESTAMPS (20 days, hourly)
# ==============================
timestamps = pd.date_range(start=START_DATE, periods=480, freq='1h')
t = np.arange(480)

BURNIN_CUTOFF = START_DATE + pd.Timedelta(hours=BURNIN_HOURS)

# Two culture events on the same synthetic stay: one the alarm catches
# (true positive), one it misses (false negative), so the figure shows both
# detection outcomes rather than only the true positive case. With a 5-day
# antibiotic course, the first episode's treatment runs from 168h to 288h;
# the second culture sits at 336h, giving a clean 24h gap (288h-312h) before
# its own detection window opens, so the FN detection window is not itself
# sitting inside the first episode's grey antibiotic shading.
CULTURES = [
    {'label': 'TP', 'time': START_DATE + pd.Timedelta(hours=168), 'abx_days': ABX_DAYS},
    {'label': 'FN', 'time': START_DATE + pd.Timedelta(hours=336), 'abx_days': ABX_DAYS},
]
for c in CULTURES:
    c['window_start'] = max(c['time'] - pd.Timedelta(hours=DETECTION_WINDOW_HOURS), BURNIN_CUTOFF)
    c['abx_end'] = c['time'] + pd.Timedelta(days=c['abx_days'])

# ==============================
# SIGNAL 1: fixed threshold demo
# A small amount of Gaussian noise (seeded) is added so the illustration
# looks like real, noisy HeRO data rather than a clean idealised curve.
# ==============================
fixed_scores = (
    1.0
    + 0.4 * np.sin(2 * np.pi * t / 12)
    + 1.4 * np.exp(-0.5 * ((t -  60) /  6) ** 2)   # early bump, no culture drawn here (produces the false positive day)
    + 1.8 * np.exp(-0.5 * ((t - 156) /  8) ** 2)    # TP episode: clear rise into the first detection window
    + 0.5 * np.exp(-0.5 * ((t - 336) /  3) ** 2)    # FN episode: brief, sub-threshold bump
    + 0.15 * np.sin(t * 7.3)
    + rng.normal(0, 0.12, size=t.shape)
)
fixed_scores = np.clip(fixed_scores, 0.1, 6.5)

effective_fixed_threshold = min(FIXED_THRESHOLD, MAX_THRESHOLD)

# ==============================
# SIGNAL 2: dynamic threshold demo
# The TP episode is a single rise-and-fall bump (rather than a sigmoid that
# stays elevated indefinitely), so the index climbs into the detection
# window, then genuinely comes back down toward baseline once treatment has
# had time to act, matching a resolving septic episode.
# ==============================
dynamic_scores = (
    3.0
    + 0.3 * np.sin(2 * np.pi * t / 16)
    + 2.6 * np.exp(-0.5 * ((t -  60) /  5) ** 2)   # early noisy bump, no culture drawn here
    + 3.2 * np.exp(-0.5 * ((t - 156) / 14) ** 2)    # TP episode: rises into the window, decays after
    + 0.4 * np.exp(-0.5 * ((t - 336) / 3) ** 2)     # FN episode: brief, sub-threshold bump
    + 0.12 * np.sin(t * 5.1)
    + rng.normal(0, 0.12, size=t.shape)
)
dynamic_scores = np.clip(dynamic_scores, 0.1, 6.5)

# ==============================
# LAGGED DYNAMIC THRESHOLD
# Mirrors determine_threshold_dynamic() + the reindex/fallback logic in
# is_detected() and compute_tn_fp_counts() from the main script: nearest
# match within 1h, missing values fall back to the series median. The
# earlier version of this demo forward-filled the first LAG_HOURS instead,
# which is not what the production pipeline does.
# ==============================
dyn_series      = pd.Series(dynamic_scores, index=timestamps)
trailing_median = dyn_series.rolling(f'{REFERENCE_HOURS}h', min_periods=1).median()

dyn_thresh_raw = trailing_median + TRIGGER_DELTA
dyn_thresh_raw = dyn_thresh_raw.clip(upper=MAX_THRESHOLD)
dyn_thresh_raw.index = trailing_median.index + pd.Timedelta(hours=LAG_HOURS)
dyn_thresh_raw = dyn_thresh_raw[~dyn_thresh_raw.index.duplicated(keep='last')]

dyn_thresh = dyn_thresh_raw.reindex(timestamps, method='nearest', tolerance=pd.Timedelta('1h'))
dyn_thresh = dyn_thresh.fillna(dyn_thresh_raw.median())

# ==============================
# MASKS
# inside_abx / outside_excl are unions across both culture events, matching
# how the production script accumulates masks across all of a patient's
# culture rows.
# ==============================
inside_abx = np.zeros(len(timestamps), dtype=bool)
outside_excl = timestamps >= BURNIN_CUTOFF
for c in CULTURES:
    inside_abx |= (timestamps >= c['time']) & (timestamps <= c['abx_end'])
    outside_excl &= ~((timestamps >= c['window_start']) & (timestamps <= c['abx_end']))

# ==============================
# DOT COLORS
# ==============================
def dot_colors(scores, thresh_vals, inside_abx_mask):
    above = scores > thresh_vals
    return np.where(inside_abx_mask, 'grey', np.where(above, 'red', 'blue'))

fixed_colors   = dot_colors(fixed_scores,   effective_fixed_threshold, inside_abx)
dynamic_colors = dot_colors(dynamic_scores, dyn_thresh.values,         inside_abx)

# ==============================
# DETECTION (per culture event)
# ==============================
for c in CULTURES:
    in_win = (timestamps >= c['window_start']) & (timestamps <= c['time'])
    c['fixed_detected'] = bool((fixed_scores[in_win] > effective_fixed_threshold).any())
    c['dyn_detected']   = bool((dynamic_scores[in_win] > dyn_thresh.values[in_win]).any())

# ==============================
# TN / FP DAYS
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

fixed_TN,   fixed_FP   = tn_fp_days(fixed_scores,   np.full(len(timestamps), effective_fixed_threshold), outside_excl)
dynamic_TN, dynamic_FP = tn_fp_days(dynamic_scores, dyn_thresh.values,                       outside_excl)

# ==============================
# METRICS
# TP/FN are summed across both culture events, so the confusion matrix
# reflects one true positive and one false negative rather than a single
# outcome.
# ==============================
def compute_metrics(detections, TN_days, FP_days):
    TP = sum(detections)
    FN = len(detections) - TP
    FP = len(FP_days)
    TN = len(TN_days)
    precision   = TP / (TP + FP)  if (TP + FP) > 0 else float('nan')
    recall      = TP / (TP + FN)  if (TP + FN) > 0 else float('nan')
    specificity = TN / (TN + FP)  if (TN + FP) > 0 else float('nan')
    npv         = TN / (TN + FN)  if (TN + FN) > 0 else float('nan')
    return dict(TP=TP, FN=FN, FP=FP, TN=TN,
                precision=precision, recall=recall,
                specificity=specificity, npv=npv)

fixed_metrics   = compute_metrics([c['fixed_detected'] for c in CULTURES], fixed_TN,   fixed_FP)
dynamic_metrics = compute_metrics([c['dyn_detected']   for c in CULTURES], dynamic_TN, dynamic_FP)

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
        if val != val:
            print(f"  {name:<30} {'N/A':>8}")
        else:
            print(f"  {name:<30} {val:>8.3f}")
    print()

print_metrics(f"Fixed threshold  (score > {effective_fixed_threshold})", fixed_metrics)
print_metrics(f"Dynamic threshold  ({REFERENCE_HOURS}h lagged median + {TRIGGER_DELTA}, capped at {MAX_THRESHOLD})", dynamic_metrics)

# ==============================================================
# AUC (single-stay illustration)
# Threshold-free discrimination of the raw HRC index, matching how the main
# validation script's raw-score AUC is constructed (raw HRC index, not a
# threshold or baseline-adjusted margin). Illustrated here on the
# fixed-threshold panel's synthetic signal (fixed_scores), since that is the
# unadjusted raw index in this demo. With only two culture events on a
# single synthetic stay (one TP, one FN), this is a worked illustration of
# the calculation, not a meaningful discrimination estimate, see the
# schematic bootstrap section below for a (still illustrative) population-
# level version with a bootstrap CI.
#
# Positive class: max raw score within each culture event's detection
# window. Negative class: max raw score per outside (non-excluded,
# non-burn-in) day, the same day set used for the TN/FP counts above.
# ==============================================================
def window_max_score(scores, times, window_start, t_culture):
    """Max score within [window_start, t_culture]; NaN if the window is empty."""
    mask = (times >= window_start) & (times <= t_culture)
    if not mask.any():
        return np.nan
    return scores[mask].max()


def compute_auc(scores, labels):
    """
    AUC = P(score of random positive > score of random negative), via
    rank-sum (Mann-Whitney U). Ties are handled with average ranking
    (pandas' default rank method), equivalent to sklearn's roc_auc_score.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method='average').values
    sum_ranks_pos = ranks[labels == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_auc_ci(pos_scores, neg_scores, n_bootstrap, seed, alpha=0.05):
    """
    Illustrative 95% CI for the single-stay AUC: resamples the positive and
    negative scores themselves, with replacement, and recomputes the AUC
    each iteration. This is NOT the patient-level, sepsis-stratified
    bootstrap the real validation script uses (there is only one synthetic
    stay here, with two events), it only demonstrates the resampling-and-
    percentile-CI mechanics; with n_pos=2 the interval will be wide.
    """
    boot_rng = np.random.default_rng(seed)
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    n_pos, n_neg = len(pos_scores), len(neg_scores)

    aucs = []
    for _ in range(n_bootstrap):
        pos_sample = boot_rng.choice(pos_scores, size=n_pos, replace=True) if n_pos > 0 else np.array([])
        neg_sample = boot_rng.choice(neg_scores, size=n_neg, replace=True) if n_neg > 0 else np.array([])
        scores = np.concatenate([pos_sample, neg_sample])
        labels = np.concatenate([np.ones(len(pos_sample)), np.zeros(len(neg_sample))])
        aucs.append(compute_auc(scores, labels))

    aucs = np.asarray(aucs, dtype=float)
    valid = aucs[~np.isnan(aucs)]
    if valid.size == 0:
        return np.nan, np.nan, 0
    lower = np.percentile(valid, 100 * (alpha / 2))
    upper = np.percentile(valid, 100 * (1 - alpha / 2))
    return lower, upper, valid.size


single_stay_pos_scores = [
    window_max_score(fixed_scores, timestamps, c['window_start'], c['time'])
    for c in CULTURES
]
single_stay_pos_scores = [s for s in single_stay_pos_scores if not np.isnan(s)]

single_stay_day_df = pd.DataFrame({'score': fixed_scores, 'date': dates}, index=timestamps)
single_stay_neg_scores = (
    single_stay_day_df[outside_excl].groupby('date')['score'].max().dropna().tolist()
)

single_stay_auc_scores = single_stay_pos_scores + single_stay_neg_scores
single_stay_auc_labels = [1] * len(single_stay_pos_scores) + [0] * len(single_stay_neg_scores)
single_stay_auc = compute_auc(single_stay_auc_scores, single_stay_auc_labels)
single_stay_auc_lower, single_stay_auc_upper, single_stay_auc_n_valid = bootstrap_auc_ci(
    single_stay_pos_scores, single_stay_neg_scores, n_bootstrap=500, seed=SEED
)

print(f"\n{'='*46}")
print("  AUC (single-stay illustration, raw HRC index)")
print(f"{'='*46}")
auc_str = f"{single_stay_auc:.3f}" if not np.isnan(single_stay_auc) else "N/A"
auc_ci_str = (f"[{single_stay_auc_lower:.3f}, {single_stay_auc_upper:.3f}]"
              if not np.isnan(single_stay_auc_lower) else "[N/A]")
print(f"  AUC: {auc_str}   95% CI {auc_ci_str}   (n positive={len(single_stay_pos_scores)}, "
      f"n negative={len(single_stay_neg_scores)}, n_valid={single_stay_auc_n_valid}/500)")
print("  Note: illustrative only. A single synthetic stay with two events")
print("  is far too small a sample for a meaningful discrimination estimate;")
print("  see the schematic bootstrap section below for how a population-level")
print("  AUC with a bootstrap CI is derived in the real validation scripts.")
print()

# ==============================
# PLOT
# ==============================
fig, (ax_fixed, ax_dyn) = plt.subplots(
    2, 1, figsize=(16, 10), sharex=True,
)
fig.subplots_adjust(top=0.93, bottom=0.28, hspace=0.35, left=0.06, right=0.98)

for ax, scores, colors, thresh, TN_days, FP_days, detect_key, panel_label in [
    (ax_fixed,
     fixed_scores, fixed_colors, None,
     fixed_TN, fixed_FP, 'fixed_detected',
     'A   Fixed threshold'),
    (ax_dyn,
     dynamic_scores, dynamic_colors, dyn_thresh,
     dynamic_TN, dynamic_FP, 'dyn_detected',
     'B   Patient specific threshold'),
]:
    # --- burn-in shading ---
    ax.axvspan(START_DATE, BURNIN_CUTOFF,
               color='lightgrey', alpha=0.5, zorder=0)

    # --- detection window shading + culture line, one per culture event ---
    for c in CULTURES:
        shade = 'green' if c[detect_key] else 'red'
        ax.axvspan(c['window_start'], c['time'], color=shade, alpha=0.15, zorder=0)
        ax.axvline(x=c['time'], color='darkblue', linestyle=':', linewidth=2, zorder=5)

    # --- scatter dots ---
    non_abx_mask = ~inside_abx
    ax.scatter(timestamps[non_abx_mask], scores[non_abx_mask],
               s=10, c=np.array(colors)[non_abx_mask], zorder=3)
    ax.scatter(timestamps[inside_abx], scores[inside_abx],
               s=10, color='grey', zorder=3)

    # --- threshold line ---
    if thresh is None:
        ax.axhline(y=effective_fixed_threshold, color='black', linestyle='--',
                   linewidth=1.4, zorder=4)
    else:
        ax.plot(thresh.index, thresh.values, color='black', linestyle='--',
                linewidth=1.4, zorder=4)

    # --- TN day markers ---
    if len(TN_days):
        tn_x = pd.to_datetime(TN_days) + pd.Timedelta(hours=12)
        ax.scatter(tn_x, np.zeros(len(tn_x)),
                   color='green', marker='|', s=300, linewidths=3, zorder=6)

    # --- FP day markers ---
    if len(FP_days):
        fp_x = pd.to_datetime(FP_days) + pd.Timedelta(hours=12)
        ax.scatter(fp_x, np.zeros(len(fp_x)),
                   color='red', marker='|', s=300, linewidths=3, zorder=6)

    # --- formatting ---
    ax.set_title(panel_label, fontsize=11, fontweight='bold', loc='left', pad=6)
    ax.set_ylabel('HRC index', fontsize=10)
    ax.set_ylim(0, 7.1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.setp(ax_dyn.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ==============================
# SHARED LEGEND
# ==============================
legend_elements = [
    mlines.Line2D([], [], color='blue',  marker='o', linestyle='None',
                  markersize=5, label='HRC index: below threshold'),
    mlines.Line2D([], [], color='red',   marker='o', linestyle='None',
                  markersize=5, label='HRC index: above threshold'),
    mlines.Line2D([], [], color='grey',  marker='o', linestyle='None',
                  markersize=5, label='HRC index: antibiotic treatment period'),
    mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.4,
                  label='Alarm threshold'),
    mlines.Line2D([], [], color='darkblue', linestyle=':', linewidth=2,
                  label='Blood culture draw'),
    mpatches.Patch(color='lightgrey', alpha=0.6,
                   label='Initialisation period (excluded from analysis)'),
    mpatches.Patch(color='green',     alpha=0.3,
                   label='Detection window: alarm raised (true positive)'),
    mpatches.Patch(color='red',       alpha=0.3,
                   label='Detection window: no alarm raised (false negative)'),
    mlines.Line2D([], [], color='green', marker='|', linestyle='None',
                  markersize=10, markeredgewidth=2.5, label='True negative'),
    mlines.Line2D([], [], color='red',   marker='|', linestyle='None',
                  markersize=10, markeredgewidth=2.5, label='False positive'),
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=2,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01),
)

fig.suptitle(
    'HRC monitoring: fixed versus patient specific threshold',
    fontsize=13,
)

plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================
# SCHEMATIC BOOTSTRAP ILLUSTRATION
#
# The two panels above show a single synthetic stay, which is not enough
# patients to run a real stratified bootstrap. This section instead builds a
# small sim population (a handful of synthetic case and control "patients")
# and runs the exact same stratified, patient-level bootstrap procedure used
# in the validation scripts, so a reader can see the resampling and CI
# mechanics end to end. The resulting numbers are illustrative only, they
# say nothing about real model performance. This includes the AUC below:
# per-patient positive/negative scores are drawn from separated synthetic
# distributions purely so the illustration AUC reads as "better than
# chance", not because they reflect any real HeRO behavior.
# ==============================================================
N_SIM_CASES    = 6
N_SIM_CONTROLS = 6
N_BOOTSTRAP    = 500   # smaller than the real validation's N_BOOTSTRAP, this is a demo, not a result
CI_ALPHA       = 0.05


def calc_metrics(TP, FN, FP, TN):
    precision   = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall      = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    npv         = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    return precision, recall, specificity, npv


def build_sim_population(rng):
    """Synthetic patient records in the same shape used by the validation
    scripts: sepsis_positive, episode_outcomes (fixed/dynamic detection per
    episode), TN/FP day counts, and per-patient raw-score AUC inputs
    (auc_pos_scores / auc_neg_scores). Detection is drawn so the dynamic
    method performs somewhat better than the fixed one, and positive-class
    scores are drawn higher than negative-class scores, purely to make the
    two threshold columns and the AUC visibly informative in the printed
    example."""
    records = []
    for _ in range(N_SIM_CASES):
        n_episodes = rng.choice([1, 1, 1, 2])  # mostly one episode, occasionally two
        episode_outcomes = [
            (bool(rng.random() < 0.55), bool(rng.random() < 0.75))
            for _ in range(n_episodes)
        ]
        # Synthetic per-episode and per-day max raw scores, for the AUC
        # illustration only, not linked to episode_outcomes above.
        auc_pos_scores = list(rng.normal(4.5, 1.0, size=n_episodes))
        auc_neg_scores = list(rng.normal(2.5, 1.0, size=int(rng.integers(3, 10))))
        records.append({
            'sepsis_positive': True,
            'episode_outcomes': episode_outcomes,
            'fixed_TN': int(rng.integers(3, 10)),
            'fixed_FP': int(rng.integers(0, 2)),
            'dyn_TN':   int(rng.integers(3, 10)),
            'dyn_FP':   int(rng.integers(0, 2)),
            'auc_pos_scores': auc_pos_scores,
            'auc_neg_scores': auc_neg_scores,
        })
    for _ in range(N_SIM_CONTROLS):
        auc_neg_scores = list(rng.normal(2.0, 1.0, size=int(rng.integers(8, 16))))
        records.append({
            'sepsis_positive': False,
            'episode_outcomes': [],
            'fixed_TN': int(rng.integers(8, 16)),
            'fixed_FP': int(rng.integers(0, 3)),
            'dyn_TN':   int(rng.integers(8, 16)),
            'dyn_FP':   int(rng.integers(0, 3)),
            'auc_pos_scores': [],
            'auc_neg_scores': auc_neg_scores,
        })
    return records


def run_stratified_bootstrap(patient_records, n_bootstrap, seed):
    """Same procedure as the validation scripts: resample patients with
    replacement, separately within the sepsis-positive and sepsis-negative
    strata, reuse each resample for both threshold methods."""
    boot_rng = np.random.default_rng(seed)
    positive = [r for r in patient_records if r['sepsis_positive']]
    negative = [r for r in patient_records if not r['sepsis_positive']]
    n_pos, n_neg = len(positive), len(negative)

    results = {
        'fixed':   {'precision': [], 'recall': [], 'specificity': [], 'npv': []},
        'dynamic': {'precision': [], 'recall': [], 'specificity': [], 'npv': []},
    }
    for _ in range(n_bootstrap):
        pos_idx = boot_rng.integers(0, n_pos, n_pos) if n_pos > 0 else np.array([], dtype=int)
        neg_idx = boot_rng.integers(0, n_neg, n_neg) if n_neg > 0 else np.array([], dtype=int)
        sample = [positive[i] for i in pos_idx] + [negative[i] for i in neg_idx]

        for key in ('fixed', 'dynamic'):
            TP = FN = FP = TN = 0
            for record in sample:
                for fixed_det, dyn_det in record['episode_outcomes']:
                    detected = fixed_det if key == 'fixed' else dyn_det
                    TP += int(detected)
                    FN += int(not detected)
                TN += record['fixed_TN'] if key == 'fixed' else record['dyn_TN']
                FP += record['fixed_FP'] if key == 'fixed' else record['dyn_FP']
            precision, recall, specificity, npv = calc_metrics(TP, FN, FP, TN)
            results[key]['precision'].append(precision)
            results[key]['recall'].append(recall)
            results[key]['specificity'].append(specificity)
            results[key]['npv'].append(npv)

    for key in ('fixed', 'dynamic'):
        for metric in ('precision', 'recall', 'specificity', 'npv'):
            results[key][metric] = np.array(results[key][metric], dtype=float)
    return results


def run_auc_bootstrap(patient_records, n_bootstrap, seed):
    """Same patient-level, stratified resampling as run_stratified_bootstrap,
    applied to the pooled positive/negative raw-score lists to produce a
    bootstrap distribution of the AUC."""
    boot_rng = np.random.default_rng(seed)
    positive = [r for r in patient_records if r['sepsis_positive']]
    negative = [r for r in patient_records if not r['sepsis_positive']]
    n_pos, n_neg = len(positive), len(negative)

    aucs = []
    for _ in range(n_bootstrap):
        pos_idx = boot_rng.integers(0, n_pos, n_pos) if n_pos > 0 else np.array([], dtype=int)
        neg_idx = boot_rng.integers(0, n_neg, n_neg) if n_neg > 0 else np.array([], dtype=int)
        sample = [positive[i] for i in pos_idx] + [negative[i] for i in neg_idx]

        scores, labels = [], []
        for record in sample:
            scores.extend(record['auc_pos_scores'])
            labels.extend([1] * len(record['auc_pos_scores']))
            scores.extend(record['auc_neg_scores'])
            labels.extend([0] * len(record['auc_neg_scores']))
        aucs.append(compute_auc(scores, labels))
    return np.array(aucs, dtype=float)


def percentile_ci(values, alpha):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    lower = np.percentile(vals, 100 * (alpha / 2))
    upper = np.percentile(vals, 100 * (1 - alpha / 2))
    return lower, upper, vals.size


def sim_point_estimate(patient_records, key):
    """
    Point estimate from the original, non-resampled sim population (all
    patients counted once), aggregated the same way each bootstrap
    iteration aggregates its resample. Mirrors the point-estimate step in
    the main validation script, which is computed separately from its
    bootstrap CI.
    """
    TP = FN = FP = TN = 0
    for record in patient_records:
        for fixed_det, dyn_det in record['episode_outcomes']:
            detected = fixed_det if key == 'fixed' else dyn_det
            TP += int(detected)
            FN += int(not detected)
        TN += record['fixed_TN'] if key == 'fixed' else record['dyn_TN']
        FP += record['fixed_FP'] if key == 'fixed' else record['dyn_FP']
    return calc_metrics(TP, FN, FP, TN)


sim_records = build_sim_population(rng)
sim_bootstrap = run_stratified_bootstrap(sim_records, n_bootstrap=N_BOOTSTRAP, seed=SEED)

print(f"\n{'='*55}")
print("SCHEMATIC BOOTSTRAP ILLUSTRATION (sim population, not a validation result)")
print(f"  {N_SIM_CASES} synthetic cases, {N_SIM_CONTROLS} synthetic controls, "
      f"n_bootstrap={N_BOOTSTRAP}, seed={SEED}")
print(f"{'='*55}")
for key, label in [('fixed', 'Fixed threshold'), ('dynamic', 'Dynamic threshold')]:
    point = sim_point_estimate(sim_records, key)
    print(f"\n  -- {label} --")
    for metric_name, point_val in zip(('precision', 'recall', 'specificity', 'npv'), point):
        lower, upper, n_valid = percentile_ci(sim_bootstrap[key][metric_name], CI_ALPHA)
        point_str = f"{point_val:.3f}" if not np.isnan(point_val) else "N/A"
        ci_str = f"[{lower:.3f}, {upper:.3f}]" if not np.isnan(lower) else "[N/A]"
        print(f"    {metric_name:<12} {point_str:>8}   95% CI {ci_str}   (n_valid={n_valid}/{N_BOOTSTRAP})")

# --- AUC (threshold-free), same sim population and bootstrap procedure ---
sim_auc_scores, sim_auc_labels = [], []
for r in sim_records:
    sim_auc_scores.extend(r['auc_pos_scores'])
    sim_auc_labels.extend([1] * len(r['auc_pos_scores']))
    sim_auc_scores.extend(r['auc_neg_scores'])
    sim_auc_labels.extend([0] * len(r['auc_neg_scores']))

sim_auc_point = compute_auc(sim_auc_scores, sim_auc_labels)
sim_auc_bootstrap = run_auc_bootstrap(sim_records, n_bootstrap=N_BOOTSTRAP, seed=SEED)
sim_auc_lower, sim_auc_upper, sim_auc_n_valid = percentile_ci(sim_auc_bootstrap, CI_ALPHA)

print(f"\n  -- AUC (threshold-free, raw score) --")
sim_auc_point_str = f"{sim_auc_point:.3f}" if not np.isnan(sim_auc_point) else "N/A"
sim_auc_ci_str = f"[{sim_auc_lower:.3f}, {sim_auc_upper:.3f}]" if not np.isnan(sim_auc_lower) else "[N/A]"
print(f"    AUC: {sim_auc_point_str}   95% CI {sim_auc_ci_str}   "
      f"(n_valid={sim_auc_n_valid}/{N_BOOTSTRAP})")
print(f"{'='*55}")