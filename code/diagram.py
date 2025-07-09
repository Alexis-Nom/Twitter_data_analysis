import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.ticker import SymmetricalLogLocator

# prepare the data we computed with array_with_CI.py
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
plt.tight_layout()
plt.show()

#the logic is the same for English and Spanish
