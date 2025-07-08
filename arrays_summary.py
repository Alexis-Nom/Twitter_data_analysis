import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the images
image_paths = ["array_french.png", "array_english.png", "array_spanish.png"]  # Replace with your files
images = [mpimg.imread(path) for path in image_paths]

# calculate dimensions (this part was added to ensure better image quality)
px = 1/plt.rcParams['figure.dpi']  # Pixel-to-inch conversion
img_width = 1187 * px
img_height = 1988 * px

# add 10% extra height for the title below (finally not mandatory as the main title is outside the figure)
total_height = img_height * 1.10  # 90% images, 10% title
fig = plt.figure(figsize=(3*img_width, total_height), dpi=100)

# create grid layout (1 row for images, 1 row for title)
gs = fig.add_gridspec(2, 3, height_ratios=[img_height, total_height-img_height],
                     hspace=0, wspace=0)

# display images in top row
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(images[i])
    ax.axis('off')


# remove all margins
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

plt.savefig('arrays.png',
           dpi=300,
           bbox_inches='tight',
           pad_inches=0.1)
plt.show()
