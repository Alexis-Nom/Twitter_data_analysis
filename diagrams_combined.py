import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# load images with different dimensions
image_paths = ["diagram_french.png", "diagram_english.png", "diagram_spanish.png"]  # Replace with your files
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
plt.close()
