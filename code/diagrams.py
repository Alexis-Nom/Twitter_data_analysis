###-------------------------------- LIBRARIES --------------------------------------###

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors
from itertools import permutations
from matplotlib.colors import Normalize
from matplotlib.ticker import SymmetricalLogLocator
import pandas as pd
from tqdm import tqdm
from significance_array import linear_regression
from significance_array import permutation_test


###-------------------------------- ANALYSIS FUNCTIONS ----------------------------###

def residual_bootstrap_ci(x, y, n_bootstrap=1000, ci=95):
    '''Word-specific bootstrap for evolution % CIs, '''
    m, b, _, y_pred = linear_regression(x, y)
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



###------------------------------- STYLING FUNCTIONS -------------------------------###


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



###-------------------------------- TABLES CREATION INCLUDING CIs --------------------###



def table_french():

  # initialize colormap
  spectral_cmap = sns.color_palette("Spectral_r", as_cmap=True)

  # data loading and prep
  df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
  years = np.array([2015, 2016, 2017, 2018, 2019])
  x = years - years[0]  # [0,1,2,3,4]
  global all_perms
  all_perms = list(permutations(range(5)))

  # now we process the data
  results = []
  for row_idx in tqdm(range(2, 41), desc="Processing words"):
      english_word = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()
      queries = df.iloc[row_idx, 2:7].values.astype(float)

      slope, intercept, r2, y_pred = linear_regression(x, queries)
      p_value = permutation_test(x, queries, all_perms)

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

  # create the dataframe
  output_df = pd.DataFrame(results)
 

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
    .set_caption(f"<b>French Trends Analysis</b> -  significant words (R²≥0.7, p≤0.05)")
  )

  # create and Style Results DataFrame
  output_df = pd.DataFrame(results)
  output_df['Significant'] = (output_df['R2'] >= 0.7) & (output_df['p_value'] <= 0.05)
  num_significant = output_df['Significant'].sum()
  return output_df

def table_spanish():

  # initialize colormap
  spectral_cmap = sns.color_palette("Spectral_r", as_cmap=True)

  # data loading and prep
  df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
  years = np.array([2015, 2016, 2017, 2018, 2019])
  x = years - years[0]  # [0,1,2,3,4]
  global all_perms
  all_perms = list(permutations(range(5)))

  # now we process the data
  results = []
  for row_idx in tqdm(range(2, 41), desc="Processing words"):
      english_word = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()
      queries = df.iloc[row_idx, 22:27].values.astype(float)

      slope, intercept, r2, y_pred = linear_regression(x, queries)
      p_value = permutation_test(x, queries, all_perms)

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

  # create the dataframe
  output_df = pd.DataFrame(results)
 

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
    .set_caption(f"<b>Spanish Trends Analysis</b> -  significant words (R²≥0.7, p≤0.05)")
  )

  # create and Style Results DataFrame
  output_df = pd.DataFrame(results)
  output_df['Significant'] = (output_df['R2'] >= 0.7) & (output_df['p_value'] <= 0.05)
  num_significant = output_df['Significant'].sum()
  return output_df


def table_english():

  # initialize colormap
  spectral_cmap = sns.color_palette("Spectral_r", as_cmap=True)

  # data loading and prep
  df = pd.read_excel("Data_adapted.xlsx", sheet_name="Chiffres 2015-2024", header=None)
  years = np.array([2015, 2016, 2017, 2018, 2019])
  x = years - years[0]  # [0,1,2,3,4]
  global all_perms
  all_perms = list(permutations(range(5)))

  # now we process the data
  results = []
  for row_idx in tqdm(range(2, 41), desc="Processing words"):
      english_word = df.iloc[row_idx, 0].split('—')[1].strip().split('-')[0].strip()
      queries = df.iloc[row_idx, 12:17].values.astype(float)

      slope, intercept, r2, y_pred = linear_regression(x, queries)
      p_value = permutation_test(x, queries, all_perms)

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

  # create the dataframe
  output_df = pd.DataFrame(results)
 

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
    .set_caption(f"<b>English Trends Analysis</b> -  significant words (R²≥0.7, p≤0.05)")
  )

  # create and Style Results DataFrame
  output_df = pd.DataFrame(results)
  output_df['Significant'] = (output_df['R2'] >= 0.7) & (output_df['p_value'] <= 0.05)
  num_significant = output_df['Significant'].sum()
  return output_df



###-------------------------------------DIAGRAM CREATION------------------------------------###



def diagram_french(output_df):

  '''Build the bar diagram for the words in French'''



  # prepare the data we computed 
  sig_df = output_df[output_df['Significant']].copy()

  # convert percentage columns if needed
  if output_df['Evolution_2015_2019 (%)'].dtype == object:
      sig_df['Evolution_pct'] = sig_df['Evolution_2015_2019 (%)'].str.rstrip('%').astype(float)
      sig_df['CI_lower'] = sig_df['Evolution_CI_Lower'].str.rstrip('%').astype(float)
      sig_df['CI_upper'] = sig_df['Evolution_CI_Upper'].str.rstrip('%').astype(float)
  else:
      sig_df['Evolution_pct'] = sig_df['Evolution_2015_2019 (%)']
      sig_df['CI_lower'] = sig_df['Evolution_CI_Lower']
      sig_df['CI_upper'] = sig_df['Evolution_CI_Upper']

  # sort by evolution percentage (biggest increase to biggest decrease)
  sig_df = sig_df.sort_values('Evolution_pct', ascending=False)

  # create figure with adjusted size
  fig, ax = plt.subplots(figsize=(16, 10))

  # create color gradient using YlOrBr palette (seaborn)
  ylorbr_cmap = sns.color_palette("YlOrBr", as_cmap=True)
  norm = Normalize(vmin=np.log10(sig_df['2015_Value'].min()),
                  vmax=np.log10(sig_df['2015_Value'].max()))

  # calculate error bar lengths
  lower_errors = sig_df['Evolution_pct'] - sig_df['CI_lower']
  upper_errors = sig_df['CI_upper'] - sig_df['Evolution_pct']
  error_bars = [lower_errors.values, upper_errors.values]

  # create bars with error bars
  bars = ax.bar(sig_df['Word'],
              sig_df['Evolution_pct'],
              color=ylorbr_cmap(norm(np.log10(sig_df['2015_Value']))),
              yerr=error_bars,
              capsize=5,
              error_kw={'elinewidth': 1.5, 'capthick': 1.5})

  # set symmetric log scale with custom range
  ax.set_yscale('symlog', linthresh=100)  # Linear threshold of ±100%
  ax.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=100, base=10))

  # set custom y-axis limits (-1000% to +20000%)
  ax.set_ylim(-1000, 20000)

  # custom grid lines
  ax.grid(True, which='both', axis='y', color='lightgrey', linestyle='--', alpha=0.6)

  # add zero line
  ax.axhline(0, color='black', linewidth=0.8)

  # customize axes
  ax.set_ylabel('Evolution between 2015 and 2019 (%) - Log Scale', fontsize=12)
  ax.set_title('Evolution of Search Interest for Significant Alternative Medicine Terms in French\nwith 95% Confidence Intervals (Ordered by Evolution Magnitude)',
              fontsize=14, pad=20)
  plt.xticks(rotation=45, ha='right', fontsize=10)

  # create colorbar
  sm = plt.cm.ScalarMappable(cmap=ylorbr_cmap, norm=norm)
  sm.set_array([])
  cbar = plt.colorbar(sm, ax=ax, pad=0.01)
  cbar.set_ticks(np.log10([100, 1000, 10000, 100000]))
  cbar.set_ticklabels(['100', '1,000', '10,000', '100,000'])
  cbar.set_label('2015 Search Volume (log scale)', fontsize=10)

  # add value labels on bars with background boxes
  for bar in bars:
      height = bar.get_height()
      # determine label position
      if abs(height) < 100:
          # small values - place inside bar
          va = 'bottom' if height > 0 else 'top'
          y_pos = height/2 if height > 0 else height*0.6
          color = 'white'
      else:
          # large values - place outside
          va = 'bottom' if height > 0 else 'top'
          y_pos = height * 1.05 if height > 0 else height * 0.95
          color = 'black'

      # add text with background box (we place a background box so the numbers are visible on the bars)
      ax.text(bar.get_x() + bar.get_width()/2.,
              y_pos,
              f'{height:.0f}%',
              ha='center',
              va=va,
              color='black',
              fontsize=9,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

  # add horizontal markers for CI bounds
  for bar, (lower, upper) in zip(bars, zip(sig_df['CI_lower'], sig_df['CI_upper'])):
      # Upper CI marker
      ax.hlines(upper,
                bar.get_x() + bar.get_width()/2 - 0.2,
                bar.get_x() + bar.get_width()/2 + 0.2,
                color='black', linewidth=1)
      # lower CI marker
      ax.hlines(lower,
                bar.get_x() + bar.get_width()/2 - 0.2,
                bar.get_x() + bar.get_width()/2 + 0.2,
                color='black', linewidth=1)
  fig.savefig(
      'diagram_french.png',
      dpi=100,  # matches figsize (11.87in*100dpi=1187px) (in order to have a better quality)
      bbox_inches='tight',  #removes extra whitespace
      pad_inches=0.1,  # small padding to prevent clipping
      facecolor='white',  # background color
      transparent=False  
  )
 
 

def diagram_spanish(output_df):

  '''Build the bar diagram for the words in Spanish'''

  # prepare the data we computed 
  sig_df = output_df[output_df['Significant']].copy()

  # convert percentage columns if needed
  if output_df['Evolution_2015_2019 (%)'].dtype == object:
      sig_df['Evolution_pct'] = sig_df['Evolution_2015_2019 (%)'].str.rstrip('%').astype(float)
      sig_df['CI_lower'] = sig_df['Evolution_CI_Lower'].str.rstrip('%').astype(float)
      sig_df['CI_upper'] = sig_df['Evolution_CI_Upper'].str.rstrip('%').astype(float)
  else:
      sig_df['Evolution_pct'] = sig_df['Evolution_2015_2019 (%)']
      sig_df['CI_lower'] = sig_df['Evolution_CI_Lower']
      sig_df['CI_upper'] = sig_df['Evolution_CI_Upper']

  # sort by evolution percentage (biggest increase to biggest decrease)
  sig_df = sig_df.sort_values('Evolution_pct', ascending=False)

  # create figure with adjusted size
  fig, ax = plt.subplots(figsize=(16, 10))

  # create color gradient using YlOrBr palette (seaborn)
  ylorbr_cmap = sns.color_palette("YlOrBr", as_cmap=True)
  norm = Normalize(vmin=np.log10(sig_df['2015_Value'].min()),
                  vmax=np.log10(sig_df['2015_Value'].max()))

  # calculate error bar lengths
  lower_errors = sig_df['Evolution_pct'] - sig_df['CI_lower']
  upper_errors = sig_df['CI_upper'] - sig_df['Evolution_pct']
  error_bars = [lower_errors.values, upper_errors.values]

  # create bars with error bars
  bars = ax.bar(sig_df['Word'],
              sig_df['Evolution_pct'],
              color=ylorbr_cmap(norm(np.log10(sig_df['2015_Value']))),
              yerr=error_bars,
              capsize=5,
              error_kw={'elinewidth': 1.5, 'capthick': 1.5})

  # set symmetric log scale with custom range
  ax.set_yscale('symlog', linthresh=100)  # Linear threshold of ±100%
  ax.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=100, base=10))

  # set custom y-axis limits (-1000% to +20000%)
  ax.set_ylim(-1000, 20000)

  # custom grid lines
  ax.grid(True, which='both', axis='y', color='lightgrey', linestyle='--', alpha=0.6)

  # add zero line
  ax.axhline(0, color='black', linewidth=0.8)

  # customize axes
  ax.set_ylabel('Evolution between 2015 and 2019 (%) - Log Scale', fontsize=12)
  ax.set_title('Evolution of Search Interest for Significant Alternative Medicine Terms in Spanish\nwith 95% Confidence Intervals (Ordered by Evolution Magnitude)',
              fontsize=14, pad=20)
  plt.xticks(rotation=45, ha='right', fontsize=10)

  # create colorbar
  sm = plt.cm.ScalarMappable(cmap=ylorbr_cmap, norm=norm)
  sm.set_array([])
  cbar = plt.colorbar(sm, ax=ax, pad=0.01)
  cbar.set_ticks(np.log10([100, 1000, 10000, 100000]))
  cbar.set_ticklabels(['100', '1,000', '10,000', '100,000'])
  cbar.set_label('2015 Search Volume (log scale)', fontsize=10)

  # add value labels on bars with background boxes
  for bar in bars:
      height = bar.get_height()
      # determine label position
      if abs(height) < 100:
          # small values - place inside bar
          va = 'bottom' if height > 0 else 'top'
          y_pos = height/2 if height > 0 else height*0.6
          color = 'white'
      else:
          # large values - place outside
          va = 'bottom' if height > 0 else 'top'
          y_pos = height * 1.05 if height > 0 else height * 0.95
          color = 'black'

      # add text with background box (we place a background box so the numbers are visible on the bars)
      ax.text(bar.get_x() + bar.get_width()/2.,
              y_pos,
              f'{height:.0f}%',
              ha='center',
              va=va,
              color='black',
              fontsize=9,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

  # add horizontal markers for CI bounds
  for bar, (lower, upper) in zip(bars, zip(sig_df['CI_lower'], sig_df['CI_upper'])):
      # Upper CI marker
      ax.hlines(upper,
                bar.get_x() + bar.get_width()/2 - 0.2,
                bar.get_x() + bar.get_width()/2 + 0.2,
                color='black', linewidth=1)
      # lower CI marker
      ax.hlines(lower,
                bar.get_x() + bar.get_width()/2 - 0.2,
                bar.get_x() + bar.get_width()/2 + 0.2,
                color='black', linewidth=1)
  fig.savefig(
      'diagram_spanish.png',
      dpi=100,  # matches figsize (11.87in*100dpi=1187px) (in order to have a better quality)
      bbox_inches='tight',  #removes extra whitespace
      pad_inches=0.1,  # small padding to prevent clipping
      facecolor='white',  # background color
      transparent=False  
  )
 


 
def diagram_english(output_df):

  '''Build the bar diagram for the words in English'''

  # prepare the data we computed 
  sig_df = output_df[output_df['Significant']].copy()

  # convert percentage columns if needed
  if output_df['Evolution_2015_2019 (%)'].dtype == object:
      sig_df['Evolution_pct'] = sig_df['Evolution_2015_2019 (%)'].str.rstrip('%').astype(float)
      sig_df['CI_lower'] = sig_df['Evolution_CI_Lower'].str.rstrip('%').astype(float)
      sig_df['CI_upper'] = sig_df['Evolution_CI_Upper'].str.rstrip('%').astype(float)
  else:
      sig_df['Evolution_pct'] = sig_df['Evolution_2015_2019 (%)']
      sig_df['CI_lower'] = sig_df['Evolution_CI_Lower']
      sig_df['CI_upper'] = sig_df['Evolution_CI_Upper']

  # sort by evolution percentage (biggest increase to biggest decrease)
  sig_df = sig_df.sort_values('Evolution_pct', ascending=False)

  # create figure with adjusted size
  fig, ax = plt.subplots(figsize=(16, 10))

  # create color gradient using YlOrBr palette (seaborn)
  ylorbr_cmap = sns.color_palette("YlOrBr", as_cmap=True)
  norm = Normalize(vmin=np.log10(sig_df['2015_Value'].min()),
                  vmax=np.log10(sig_df['2015_Value'].max()))

  # calculate error bar lengths
  lower_errors = sig_df['Evolution_pct'] - sig_df['CI_lower']
  upper_errors = sig_df['CI_upper'] - sig_df['Evolution_pct']
  error_bars = [lower_errors.values, upper_errors.values]

  # create bars with error bars
  bars = ax.bar(sig_df['Word'],
              sig_df['Evolution_pct'],
              color=ylorbr_cmap(norm(np.log10(sig_df['2015_Value']))),
              yerr=error_bars,
              capsize=5,
              error_kw={'elinewidth': 1.5, 'capthick': 1.5})

  # set symmetric log scale with custom range
  ax.set_yscale('symlog', linthresh=100)  # Linear threshold of ±100%
  ax.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=100, base=10))

  # set custom y-axis limits (-1000% to +20000%)
  ax.set_ylim(-1000, 20000)

  # custom grid lines
  ax.grid(True, which='both', axis='y', color='lightgrey', linestyle='--', alpha=0.6)

  # add zero line
  ax.axhline(0, color='black', linewidth=0.8)

  # customize axes
  ax.set_ylabel('Evolution between 2015 and 2019 (%) - Log Scale', fontsize=12)
  ax.set_title('Evolution of Search Interest for Significant Alternative Medicine Terms in English\nwith 95% Confidence Intervals (Ordered by Evolution Magnitude)',
              fontsize=14, pad=20)
  plt.xticks(rotation=45, ha='right', fontsize=10)

  # create colorbar
  sm = plt.cm.ScalarMappable(cmap=ylorbr_cmap, norm=norm)
  sm.set_array([])
  cbar = plt.colorbar(sm, ax=ax, pad=0.01)
  cbar.set_ticks(np.log10([100, 1000, 10000, 100000]))
  cbar.set_ticklabels(['100', '1,000', '10,000', '100,000'])
  cbar.set_label('2015 Search Volume (log scale)', fontsize=10)

  # add value labels on bars with background boxes
  for bar in bars:
      height = bar.get_height()
      # determine label position
      if abs(height) < 100:
          # small values - place inside bar
          va = 'bottom' if height > 0 else 'top'
          y_pos = height/2 if height > 0 else height*0.6
          color = 'white'
      else:
          # large values - place outside
          va = 'bottom' if height > 0 else 'top'
          y_pos = height * 1.05 if height > 0 else height * 0.95
          color = 'black'

      # add text with background box (we place a background box so the numbers are visible on the bars)
      ax.text(bar.get_x() + bar.get_width()/2.,
              y_pos,
              f'{height:.0f}%',
              ha='center',
              va=va,
              color='black',
              fontsize=9,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

  # add horizontal markers for CI bounds
  for bar, (lower, upper) in zip(bars, zip(sig_df['CI_lower'], sig_df['CI_upper'])):
      # Upper CI marker
      ax.hlines(upper,
                bar.get_x() + bar.get_width()/2 - 0.2,
                bar.get_x() + bar.get_width()/2 + 0.2,
                color='black', linewidth=1)
      # lower CI marker
      ax.hlines(lower,
                bar.get_x() + bar.get_width()/2 - 0.2,
                bar.get_x() + bar.get_width()/2 + 0.2,
                color='black', linewidth=1)
  fig.savefig(
      'diagram_english.png',
      dpi=100,  # matches figsize (11.87in*100dpi=1187px) (in order to have a better quality)
      bbox_inches='tight',  #removes extra whitespace
      pad_inches=0.1,  # small padding to prevent clipping
      facecolor='white',  # background color
      transparent=False  
  )
  
 

###--------------------------------------- MAIN FUNCTION ------------------------------------------###


def diagrams_combined():
  table_fr = table_french()
  diag_fr = diagram_french(table_fr)
  table_sp = table_spanish()
  diag_sp = diagram_spanish(table_sp)
  table_en = table_english()
  diag_en = diagram_english(table_en)


  # load images with different dimensions
  image_paths = ["diagram_french.png", "diagram_english.png", "diagram_spanish.png"]  
  images = [mpimg.imread(path) for path in image_paths]
  dimensions = [img.shape[:2][::-1] for img in images]  # get (width, height) for each

  # calculate scaling factors to normalize heights
  ref_height = 1001  # ref height (choose one or use min/max)
  scaling_factors = [ref_height / h for w, h in dimensions]

  # create figure with dynamic width
  px = 1/plt.rcParams['figure.dpi']
  total_width = sum(w * scale for (w, h), scale in zip(dimensions, scaling_factors)) * px
  total_height = (ref_height + 200) * px  # +200px for title (finally not essential as the title is outside the figure)

  fig = plt.figure(figsize=(total_width, total_height), dpi=100)
  gs = fig.add_gridspec(2, 3, height_ratios=[ref_height, 200], hspace=0, wspace=0.05)

  # plot images with proportional scaling
  for i, (img, scale) in enumerate(zip(images, scaling_factors)):
      ax = fig.add_subplot(gs[0, i])
      ax.imshow(img, aspect='auto')  # 'auto' prevents forced equal aspect ratio
      ax.axis('off')




  # save with exact dimensions
  output_width = int(sum(w * scale for (w, h), scale in zip(dimensions, scaling_factors)))
  output_height = ref_height + 200
  plt.savefig('combined_figure.png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor='white')
  plt.show()

