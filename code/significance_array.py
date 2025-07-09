import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors

# initialize colormap (use of seaborn)
spectral_cmap = sns.color_palette("Spectral_r", as_cmap=True)

# 1. Regression and Analysis Functions (unchanged)
def manual_linear_regression(x, y):
    '''
    Compute the linear regression for a given dataset of coordinates (x,y).
    
    Parameters
    ---
    x : numpy.ndarray
        1D array of values (used as indexes)
    y : numpy.ndarray
        1D array values 
        
    Returns
    ---
    tuple
        A tuple containing four elements:
        - m (float): Slope of the regression line (coefficient)
        - b (float): Y-intercept of the regression line
        - r2 (float): R-squared value (coefficient of determination)
        - y_pred (numpy.ndarray): Predicted y-values based on the regression line
    '''
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
  '''
  Make a permutation test for a given dataset and returns the p-value associated.

  Parameters
  ---
  observed_data : numpy.ndarray
    1D array of dependent variable value

  Returns
  ---
  extreme_count / n_permutations (float) : the p-value
  '''
    obs_slope = manual_linear_regression(x, observed_data)[0]
    extreme_count = 0

    for perm in all_perms[:n_permutations]:
        perm_slope = manual_linear_regression(x, observed_data[list(perm)])[0]
        if abs(perm_slope) >= abs(obs_slope):
            extreme_count += 1

    return extreme_count / n_permutations

# 2. Data Loading - SPANISH TERMS (Columns W-AA)
df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
years = np.array([2015, 2016, 2017, 2018, 2019])
x = years - years[0]  # reset the numbers to [0,1,2,3,4] for easily deal with indexes
all_perms = list(permutations(range(5)))

# process SPANISH terms (columns 22-26 = W-AA) with ENGLISH labels
results = []
for row_idx in tqdm(range(2, 41), desc="Processing Spanish trends"):
    # extract English term name for labeling
    english_term = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()

    # get Spanish data values (columns W-AA = indexes 22 to 27)
    queries = df.iloc[row_idx, 22:27].values.astype(float)

    # skip rows with invalid data
    if len(queries) != 5 or np.isnan(queries).any():
        continue

    slope, intercept, r2, y_pred = manual_linear_regression(x, queries)
    p_value = permutation_test(queries)
  

    first_value = queries[0]

    results.append({
        'Term': english_term, 
        'R\u00B2': r2,
        'p-value': p_value,
        'Slope': slope,
        '2015_Value': first_value,
        'Significant': (r2 >= 0.7) & (p_value <= 0.05)
    })

# create dataFrame
output_df = pd.DataFrame(results)
num_significant = output_df['Significant'].sum()

# styling functions 
def threshold_coloring(val, column):
  '''
  It colors the columns R2 and p-value, centering the colormap on the respective thresholds (0.7 for the R2 and 0.05 for the p-value)
  Blue (cold) colors do not respect the threshold and red (hot) colors respect it.

  '''
    if pd.isna(val):
        return ''

    if column == 'R\u00B2':
        if val >= 0.7:
            normalized = 1-(val - 0.7) / 0.3
            color_value = 1.0 - (0.3 * normalized)
            color = spectral_cmap(color_value)
        else:
            normalized = val / 0.7
            color = spectral_cmap(0.25 * normalized)
    elif column == 'p-value':
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
output_df_sorted = output_df.sort_values(by='R\u00B2', ascending=False).reset_index(drop=True)

styled_df = (
    output_df_sorted.style
    .format({
        'R\u00B2': '{:.3f}',
        'p-value': '{:.4f}',
        'Slope': '{:.3f}',
        '2015_Value': '{:,.0f}'
    })
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid #ddd'
    })
    .map(lambda x: threshold_coloring(x, 'R\u00B2'), subset=['R\u00B2'])
    .map(lambda x: threshold_coloring(x, 'p-value'), subset=['p-value'])
    .map(color_significant, subset=['Significant'])
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#404040'),
            ('color', 'white'),
            ('font-weight', 'bold')
        ]},
        {'selector': 'tr:hover', 'props': [('background-color', '#ffff99')]}
    ])
    .set_caption(f"<b>Spanish Trends Analysis</b> - {num_significant} significant terms (R²≥0.7, p≤0.05)")
)

# display results
display(styled_df)
styled_df.to_html('array_spanish.html')
print("\nSignificant terms (Spanish search data):")
for term in output_df[output_df['Significant']]['Term']:
    print(f"- {term}")

'''
additionnal script for the other languages :

# data Loading - ENGLISH TERMS (Columns M-Q)
df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
years = np.array([2015, 2016, 2017, 2018, 2019])
x = years - years[0]  # [0,1,2,3,4]
all_perms = list(permutations(range(5)))

# process ENGLISH terms (columns 12-16 = M-Q)
results = []
for row_idx in tqdm(range(2, 41), desc="Processing English words"):
    english_term = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()
    queries = df.iloc[row_idx, 12:17].values.astype(float)  # M-Q columns

    slope, intercept, r2, y_pred = manual_linear_regression(x, queries)
    p_value = permutation_test(queries)
    first_value = queries[0]

    results.append({
        'Term': english_term,
        'R\u00B2': r2,
        'p-value': p_value,
        'Slope': slope,
        '2015_Value': first_value,
        'Significant': (r2 >= 0.7) & (p_value <= 0.05)
    })

# create DataFrame
output_df = pd.DataFrame(results)
num_significant = output_df['Significant'].sum()

output_df_sorted = output_df.sort_values(by='R\u00B2', ascending=False).reset_index(drop=True)

styled_df = (
    output_df_sorted.style
    .format({
        'R\u00B2': '{:.3f}',
        'p-value': '{:.4f}',
        'Slope': '{:.3f}',
        '2015_Value': '{:,.0f}'
    })
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid #ddd'
    })
    .map(lambda x: threshold_coloring(x, 'R\u00B2'), subset=['R\u00B2'])
    .map(lambda x: threshold_coloring(x, 'p-value'), subset=['p-value'])
    .map(color_significant, subset=['Significant'])
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#404040'),
            ('color', 'white'),
            ('font-weight', 'bold')
        ]},
        {'selector': 'tr:hover', 'props': [('background-color', '#ffff99')]}
    ])
    .set_caption(f"<b>English Trends Analysis</b> - {num_significant} significant terms (R²≥0.7, p≤0.05)")
)

# display results
display(styled_df)
styled_df.to_html('array_english.html')
print("\nSignificant English terms:")
for term in output_df[output_df['Significant']]['Term']:
    print(f"- {term}")


# data Loading and preparation
df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
years = np.array([2015, 2016, 2017, 2018, 2019])
x = years - years[0]  # [0,1,2,3,4]
all_perms = list(permutations(range(5)))

# process data
results = []
for row_idx in tqdm(range(2, 41), desc="Processing words"):
    english_word = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()
    queries = df.iloc[row_idx, 2:7].values.astype(float)

    slope, intercept, r2, y_pred = manual_linear_regression(x, queries)
    p_value = permutation_test(queries)
    first_value = queries[0]

    results.append({
        'Term': english_word,
        'R\u00B2': r2,
        'p-value': p_value,
        'Slope': slope,
        '2015_Value': first_value,
        'Significant': (r2 >= 0.7) & (p_value <= 0.05)
    })

# create and styling results DataFrame
output_df = pd.DataFrame(results)
num_significant = output_df['Significant'].sum()

# sort and style
output_df_sorted = output_df.sort_values(by='R\u00B2', ascending=False).reset_index(drop=True)

styled_df = (
    output_df_sorted.style
    .format({
        'R\u00B2': '{:.3f}',
        'p-value': '{:.4f}',
        'Slope': '{:.3f}',
        '2015_Value': '{:,.0f}'
    })
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid #ddd'
    })
    .map(lambda x: threshold_coloring(x, 'R\u00B2'), subset=['R\u00B2'])
    .map(lambda x: threshold_coloring(x, 'p-value'), subset=['p-value'])
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

# display results
display(styled_df)
styled_df.to_html('array_french.html')
# print significant words
if num_significant > 0:
    print("\nSignificant words:")
    for word in output_df[output_df['Significant']]['Term']:
        print(f"- {word}")
