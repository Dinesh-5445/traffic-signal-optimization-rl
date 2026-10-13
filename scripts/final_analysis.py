import pandas as pd
import numpy as np
from scipy import stats
import json

# Configuration
DATA_PATH = "results/phase4_final/raw_episodes.csv"
CONTROLLERS = ["fixed_time", "random", "rl_marl_gnn"]
SCENARIOS = ["balanced", "ns_heavy", "ew_heavy", "rush_hour", "congested", "random"]
METRICS = ["mean_queue", "mean_wait", "total_flow"]

def cohens_d_paired(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def compute_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h

def analyze_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error reading {DATA_PATH}: {e}")
        return

    # Data validation
    print("=== Data Validation ===")
    print(f"Total rows: {len(df)}")

    # Check completeness
    for scenario in SCENARIOS:
        for controller in CONTROLLERS:
            sub = df[(df['scenario'] == scenario) & (df['controller'] == controller)]
            if len(sub) != 10:
                print(f"Warning: {scenario} - {controller} has {len(sub)} episodes instead of 10.")

    # Descriptive Stats
    print("\n=== Descriptive Statistics ===")
    desc_results = []
    for scenario in SCENARIOS:
        for controller in CONTROLLERS:
            sub = df[(df['scenario'] == scenario) & (df['controller'] == controller)]
            for metric in METRICS:
                data = sub[metric].values
                m = np.mean(data)
                std = np.std(data, ddof=1)
                ci_low, ci_high = compute_ci(data)
                desc_results.append({
                    "scenario": scenario,
                    "controller": controller,
                    "metric": metric,
                    "mean": m,
                    "std": std,
                    "ci_95": (ci_low, ci_high)
                })

    # Paired comparisons
    print("\n=== Controller Comparisons (Paired t-tests) ===")
    comparisons = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            # RL vs Fixed Time
            rl_data = df[(df['scenario'] == scenario) & (df['controller'] == 'rl_marl_gnn')].sort_values('seed')[metric].values
            ft_data = df[(df['scenario'] == scenario) & (df['controller'] == 'fixed_time')].sort_values('seed')[metric].values

            if len(rl_data) == 10 and len(ft_data) == 10:
                diff_rl_ft = rl_data - ft_data
                t_ft, p_ft = stats.ttest_rel(rl_data, ft_data)
                ci_low_ft, ci_high_ft = compute_ci(diff_rl_ft)
                d_ft = cohens_d_paired(rl_data, ft_data)

                # relative diff: (RL - Baseline) / Baseline
                rel_diff_ft = np.mean(diff_rl_ft) / np.mean(ft_data) * 100

                comparisons.append({
                    "scenario": scenario,
                    "metric": metric,
                    "comparison": "RL vs Fixed-Time",
                    "abs_diff": np.mean(diff_rl_ft),
                    "rel_diff_pct": rel_diff_ft,
                    "ci_95_diff": (ci_low_ft, ci_high_ft),
                    "p_value": p_ft,
                    "effect_size_d": d_ft
                })

            # RL vs Random
            rand_data = df[(df['scenario'] == scenario) & (df['controller'] == 'random')].sort_values('seed')[metric].values
            if len(rl_data) == 10 and len(rand_data) == 10:
                diff_rl_rand = rl_data - rand_data
                t_rand, p_rand = stats.ttest_rel(rl_data, rand_data)
                ci_low_rand, ci_high_rand = compute_ci(diff_rl_rand)
                d_rand = cohens_d_paired(rl_data, rand_data)

                rel_diff_rand = np.mean(diff_rl_rand) / np.mean(rand_data) * 100

                comparisons.append({
                    "scenario": scenario,
                    "metric": metric,
                    "comparison": "RL vs Random",
                    "abs_diff": np.mean(diff_rl_rand),
                    "rel_diff_pct": rel_diff_rand,
                    "ci_95_diff": (ci_low_rand, ci_high_rand),
                    "p_value": p_rand,
                    "effect_size_d": d_rand
                })

    # Multiple Comparison Correction (Holm-Bonferroni) per comparison family
    print("\n=== Adjusted p-values ===")
    from statsmodels.stats.multitest import multipletests

    # We will adjust p-values for RL vs FT across all scenarios and metrics (18 tests)
    p_ft_list = [c['p_value'] for c in comparisons if c['comparison'] == 'RL vs Fixed-Time']
    adj_p_ft = multipletests(p_ft_list, method='holm')[1]

    idx = 0
    for c in comparisons:
        if c['comparison'] == 'RL vs Fixed-Time':
            c['adj_p_value'] = adj_p_ft[idx]
            idx += 1

    # Adjust for RL vs Random (18 tests)
    p_rand_list = [c['p_value'] for c in comparisons if c['comparison'] == 'RL vs Random']
    adj_p_rand = multipletests(p_rand_list, method='holm')[1]

    idx = 0
    for c in comparisons:
        if c['comparison'] == 'RL vs Random':
            c['adj_p_value'] = adj_p_rand[idx]
            idx += 1

    # Output nicely
    for c in comparisons:
        print(f"{c['comparison']} | {c['scenario']} | {c['metric']}")
        print(f"  Abs Diff: {c['abs_diff']:.2f}, Rel: {c['rel_diff_pct']:.2f}%")
        print(f"  95% CI of Diff: [{c['ci_95_diff'][0]:.2f}, {c['ci_95_diff'][1]:.2f}]")
        print(f"  p-value: {c['p_value']:.4e} (adj: {c['adj_p_value']:.4e}), Cohen's d: {c['effect_size_d']:.2f}\n")

    # Overall Means
    print("\n=== Overall Means ===")
    for controller in CONTROLLERS:
        print(f"\nController: {controller}")
        sub = df[df['controller'] == controller]
        for metric in METRICS:
            print(f"  {metric}: {sub[metric].mean():.2f}")

if __name__ == "__main__":
    analyze_data()
