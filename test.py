from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def test_nearest_neighbor_resize():
  try:
    from A1 import nearest_neighbor_resize
  
  except Exception as e:
    print(f"Error in test_nearest_neighbor_resize: {e}")

def test_lbp():
  try:
    from A1 import lbp

  except Exception as e:
    print(f"Error in test_lbp: {e}")

from A1 import nearest_neighbor_resize, lbp

def test_dog_image():
  # 1. Load an image
  # If you don't have one, this loads a sample image from matplotlib
  try:
    # Load a standard test image
    img = mpimg.imread('dog.png')

    # Convert to grayscale if it's RGB (Average the channels)
    if len(img.shape) == 3:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    # If the image is in 0-1 range, scale it to 0-255
    if img.max() <= 1.0:
      img = (img * 255).astype(np.uint8)

    # 2. Run your functions
    resized = nearest_neighbor_resize(img, 200, 200)
    lbp_img = lbp(img)

    # 3. Visualize the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original")

    ax[1].imshow(resized, cmap='gray')
    ax[1].set_title("Resized (200x200)")

    ax[2].imshow(lbp_img, cmap='gray')
    ax[2].set_title("LBP Encoded")

    plt.show()

  except FileNotFoundError:
    print("Please place a test in the folder or change the filename.")

def main():
    test_nearest_neighbor_resize()
    test_lbp()
    test_dog_image()
    print("Done!")

if __name__ == "__main__":
    main()
