import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors

# initialize colormap
spectral_cmap = sns.color_palette("Spectral_r", as_cmap=True)

# linear regression and analysis functions
def manual_linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return m, b, r2, y_pred

def permutation_test(observed_data, n_permutations=120):
    obs_slope = manual_linear_regression(x, observed_data)[0]
    extreme_count = 0

    for perm in all_perms[:n_permutations]:
        perm_slope = manual_linear_regression(x, observed_data[list(perm)])[0]
        if abs(perm_slope) >= abs(obs_slope):
            extreme_count += 1

    return extreme_count / n_permutations

def residual_bootstrap_ci(x, y, n_bootstrap=1000, ci=95):
    """Word-specific residual bootstrap for evolution % CIs"""
    m, b, _, y_pred = manual_linear_regression(x, y)
    residuals = y - y_pred

    evolutions = []
    for _ in range(n_bootstrap):
        # Resample residuals with replacement
        boot_residuals = np.random.choice(residuals, size=len(x), replace=True)
        y_boot = y_pred + boot_residuals

        # calculate evolution percentage
        if y_boot[0] > 0:  # only calculate if starting value > 0
            evolution = ((y_boot[-1] - y_boot[0]) / y_boot[0]) * 100
            evolutions.append(evolution)

    if len(evolutions) == 0:
        return np.nan, np.nan

    lower = np.percentile(evolutions, (100 - ci) / 2)
    upper = np.percentile(evolutions, ci + (100 - ci) / 2)
    return lower, upper

# data loading and prep
df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
years = np.array([2015, 2016, 2017, 2018, 2019])
x = years - years[0]  # [0,1,2,3,4]
all_perms = list(permutations(range(5)))

# now we process the data
results = []
for row_idx in tqdm(range(2, 41), desc="Processing words"):
    english_word = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()
    queries = df.iloc[row_idx, 2:7].values.astype(float)

    slope, intercept, r2, y_pred = manual_linear_regression(x, queries)
    p_value = permutation_test(queries)

    first_value = queries[0]
    last_value = queries[-1]
    evolution_pct = ((last_value - first_value) / first_value) * 100 if first_value != 0 else np.nan

    # get word-specific bootstrap CI
    ci_lower, ci_upper = residual_bootstrap_ci(x, queries)

    results.append({
        'Word': english_word,
        'R2': r2,
        'p_value': p_value,
        'Slope': slope,
        '2015_Value': first_value,
        'Evolution_2015_2019 (%)': evolution_pct,
        'Evolution_CI_Lower': ci_lower,
        'Evolution_CI_Upper': ci_upper
    })

# create and Style Results DataFrame
output_df = pd.DataFrame(results)
output_df['Significant'] = (output_df['R2'] >= 0.7) & (output_df['p_value'] <= 0.05)
num_significant = output_df['Significant'].sum()

# styling functions
def threshold_coloring(val, column):
    if pd.isna(val):
        return ''

    if column == 'R2':
        if val >= 0.7:
            normalized = 1-(val - 0.7) / 0.3
            color_value = 1.0 - (0.3 * normalized)
            color = spectral_cmap(color_value)
        else:
            normalized = val / 0.7
            color = spectral_cmap(0.25 * normalized)
    elif column == 'p_value':
        if val <= 0.05:
            normalized = 1-(0.05 - val) / 0.05
            color_value = 1.0 - (0.3 * normalized)
            color = spectral_cmap(color_value)
        else:
            normalized = (val - 0.05) / 0.95
            color = spectral_cmap(0.25 * (1 - normalized))
    return f'background-color: {matplotlib.colors.rgb2hex(color)}; color: white'

def color_significant(val):
    if isinstance(val, bool):
        return ('color: white; background-color: #4dac26' if val
                else 'color: white; background-color: #d01c8b')
    return ''

# Sort and style
output_df_sorted = output_df.sort_values(by='R2', ascending=False).reset_index(drop=True)

styled_df = (
    output_df_sorted.style
    .format({
        'R2': '{:.3f}',
        'p_value': '{:.4f}',
        'Slope': '{:.3f}',
        '2015_Value': '{:,.0f}',
        'Evolution_2015_2019 (%)': '{:+.1f}%',
        'Evolution_CI_Lower': '{:+.1f}%',
        'Evolution_CI_Upper': '{:+.1f}%'
    })
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid #ddd'
    })
    .map(lambda x: threshold_coloring(x, 'R2'), subset=['R2'])
    .map(lambda x: threshold_coloring(x, 'p_value'), subset=['p_value'])
    .map(color_significant, subset=['Significant'])
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#404040'),
            ('color', 'white'),
            ('font-weight', 'bold')
        ]},
        {'selector': 'tr:hover', 'props': [('background-color', '#ffff99')]}
    ])
    .set_caption(f"<b>French Trends Analysis</b> - {num_significant} significant words (R²≥0.7, p≤0.05)")
)

# visualization
plt.figure(figsize=(12, 8))
significant_df = output_df[output_df['Significant']].sort_values('Evolution_2015_2019 (%)')

# calculate eror bars (absolute distances)
lower_errors = abs(significant_df['Evolution_2015_2019 (%)'] - significant_df['Evolution_CI_Lower'])
upper_errors = abs(significant_df['Evolution_CI_Upper'] - significant_df['Evolution_2015_2019 (%)'])

# create plot
plt.errorbar(
    significant_df['Evolution_2015_2019 (%)'],
    significant_df['Word'],
    xerr=[lower_errors, upper_errors],
    fmt='o',
    capsize=5,
    color='#4dac26',
    alpha=0.7
)

plt.axvline(0, color='gray', linestyle='--')
plt.title('Percentage Change 2015-2019 with 95% Confidence Intervals\n(Significant Trends Only)')
plt.xlabel('Percentage Change (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# display results
display(styled_df)
plt.show()


# print significant words
if num_significant > 0:
    print("\nSignificant words:")
    for word in output_df[output_df['Significant']]['Word']:
        print(f"- {word}")
