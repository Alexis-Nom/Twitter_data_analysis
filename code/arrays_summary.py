###----------------------------------- LIBRARIES ------------------------------------###

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


###----------------------------------- MAIN FUNCTION --------------------------------###

def create_tables_figure():
  '''
  Create the final figure gathering the 3 tables for French, English, and Spanish. 
  It adapts the dimensions of the figures to ensure a good output quality.
  
  '''
  # Load the images
  image_paths = ["table_french.png", "table_english.png", "table_spanish.png"]
  images = [mpimg.imread(path) for path in image_paths]
  img_sizes = [(1167, 1988), (1197, 1988), (1182, 1988)]  # Actual image sizes

  # Calculate dimensions based on the average width and common height
  px = 1/plt.rcParams['figure.dpi']  # Pixel-to-inch conversion
  avg_width = sum(size[0] for size in img_sizes) / len(img_sizes) * px
  common_height = img_sizes[0][1] * px  # Using first image's height as reference

  # Figure dimensions (width accommodates 3 images side by side)
  fig_width = 3 * avg_width
  fig_height = common_height  # No extra space for title as per your note

  # Create figure with proper dimensions
  fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)

  # Create grid layout (1 row for images)
  gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)

  # Display images with their original aspect ratios
  for i, (img, size) in enumerate(zip(images, img_sizes)):
      ax = fig.add_subplot(gs[0, i])
      ax.imshow(img)
      ax.axis('off')

      # Set extent to maintain correct aspect ratio
      ax.set_xlim(0, size[0])
      ax.set_ylim(size[1], 0)  # Inverted y-axis for images

  # Remove all margins
  plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

  # Save with high quality
  plt.savefig('arrays.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0)  # Changed to 0 since we want no padding

  plt.show()
