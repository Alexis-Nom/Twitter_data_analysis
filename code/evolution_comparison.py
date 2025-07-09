import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# we adapt the shape of the data so we can easily handle it for comparison
french_df = pd.DataFrame({
    'word': ['Shiatsu', 'Reflexology', 'Yoga', 'Ketogenic diet', 'Complementary medicine',
             'Homeopathy', 'Chiropractic', 'Reiki', 'Ayurveda', 'Shamanism', 'Tai chi'],
    'slope': [-1694, -5676.8, -19738, 432.8, 70.5, 33281.8, 2454.7, -3629.1, -144.5, 281, -1804.3]
})

english_df = pd.DataFrame({
    'word': ['Yoga', 'Shiatsu', 'Reiki', 'Meditation', 'Reflexology', 'Hypnotherapy',
             'Osteopathy', 'Acupuncture', 'Qigong', 'Intermittent fasting', 'Moxibustion',
             'Ayurveda', 'Integrative medicine'],
    'slope': [-2113203.1, -21331, -95214.4, -786175.5, -23404.3, -34822.2, -6909, -89451.9,
              -12733, 46095.5, -1772.9, -27278.3, -1827.7]
})

spanish_df = pd.DataFrame({
    'word': ['Energy healing', 'Reflexology', 'Forest bathing / Sylvotherapy', 'Neurofeedback',
             'Integrative medicine', 'Meditation', 'Intermittent fasting', 'Reiki'],
    'slope': [-85.5, -3015.7, 35.9, -514.3, 760.5, -63200, 6815.5, -30248.2]
})

# combine all unique words (union)
all_words = list(set(french_df['word']).union(english_df['word']).union(spanish_df['word']))

# build the unified data structure
data = []
for word in all_words:
    entry = {'word': word}

    # French slope (if exists)
    french_match = french_df[french_df['word'] == word]
    entry['french'] = french_match['slope'].values[0] if not french_match.empty else None

    # English slope (if exists)
    english_match = english_df[english_df['word'] == word]
    entry['english'] = english_match['slope'].values[0] if not english_match.empty else None

    # Spanish slope (if exists)
    spanish_match = spanish_df[spanish_df['word'] == word]
    entry['spanish'] = spanish_match['slope'].values[0] if not spanish_match.empty else None

    data.append(entry)

# sort alphabetically for cleaner Y-axis
data = sorted(data, key=lambda x: x['word'])
print(data)


# create figure with external legend space
plt.figure(figsize=(12, 8), dpi=100)
ax = plt.gca()

# settings
languages = ['French', 'Spanish', 'English']
colors = {'increase': '#fc0808', 'decrease': '#0f67cd'}  # Red/Green
y_pos = np.arange(len(data))

# plot points
for lang_idx, language in enumerate(languages):
    lang_key = language.lower()
    for word_idx, word_data in enumerate(data):
        slope = word_data.get(lang_key)
        if slope is not None:
            color = colors['increase'] if slope > 0 else colors['decrease']
            ax.scatter(
                lang_idx,
                word_idx,
                s=100,  # Fixed size
                c=color,
                edgecolor='white',
                linewidth=1,
                zorder=3
            )

ax.set_xticks(range(len(languages)))
ax.set_xticklabels(languages, fontsize=12)
ax.set_yticks(y_pos)
ax.set_yticklabels([d['word'] for d in data], fontsize=10)

# grid settings
ax.grid(True,
        axis='both',  # both x and y grid
        color='lightgray',
        linestyle='-',
        linewidth=0.7,
        alpha=0.4)
ax.set_axisbelow(True)  

# top-right legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Increasing',
               markerfacecolor=colors['increase'], markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Decreasing',
               markerfacecolor=colors['decrease'], markersize=8)
]
plt.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(1.25, 1),
    frameon=False,
    title="Trend Direction",
    title_fontsize=10,
    fontsize=9
)

plt.title("Significant Trends in Alternative Medicine Terms", pad=20, fontsize=14)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for external legend
plt.show()
