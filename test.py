import pandas as pd
from scipy import stats

# Load the CSVs
df_orig = pd.read_csv('eval_orig.csv')
df_abl = pd.read_csv('eval_ablated.csv')

# 1. Global Averages
print("=== GLOBAL AVERAGES (SDR) ===")
print(f"Original TDANet: {df_orig['sdr_avg'].mean():.3f} dB")
print(f"Ablated TDANet:  {df_abl['sdr_avg'].mean():.3f} dB")
diff = df_orig['sdr_avg'].mean() - df_abl['sdr_avg'].mean()
print(f"Improvement from Attention: +{diff:.3f} dB\n")

# 2. Stem Breakdown
print("=== STEM BREAKDOWN (SDR) ===")
stems = ['vocals_sdr', 'drums_sdr', 'bass_sdr', 'other_sdr']
for stem in stems:
    orig_val = df_orig[stem].mean()
    abl_val = df_abl[stem].mean()
    print(f"{stem.split('_')[0].capitalize()}: Orig = {orig_val:.3f} | Abl = {abl_val:.3f} | Diff = {orig_val - abl_val:+.3f} dB")

# 3. Statistical Significance
stat, p_val = stats.ttest_rel(df_orig['sdr_avg'].dropna(), df_abl['sdr_avg'].dropna())
print(f"\n=== STATISTICAL SIGNIFICANCE ===")
print(f"Paired t-test p-value: {p_val:.5f}")
if p_val < 0.05:
    print("Conclusion: The improvement IS statistically significant!")
else:
    print("Conclusion: The improvement is not statistically significant.")