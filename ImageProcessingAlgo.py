import numpy as np

def nearest_neighbor_resize(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resizes an image using nearest-neighbor interpolation.

    Args:
        image (np.ndarray): Input grayscale image as a 2D array.
        new_height (int): Desired height of the resized image.
        new_width (int): Desired width of the resized image.

    Returns:
        np.ndarray: Resized image as a 2D array.
    """

    # Compute the scaling factors
    scaleRow = image.shape[0]/new_height
    scaleCol = image.shape[1]/new_width

    # Create an empty array for the resized image
    resizedImage = np.zeros((new_height, new_width), dtype=int)

    # Map each pixel in the output image to the nearest pixel in the input image
    for row in range(new_height):
        for col in range(new_width):
            # Find the nearest neighbor in the original image
            orgRow = int(round(row*scaleRow))
            orgCol = int(round(col*scaleCol))

            if orgRow >= image.shape[0]:
                orgRow = image.shape[0] - 1
            if orgCol >= image.shape[1]:
                orgCol = image.shape[1] - 1

            # Assign the value from the nearest neighbor
            resizedImage[row][col] = image[orgRow][orgCol]

    # return resized_image
    return resizedImage

def lbp(image):
    """
    Compute the LBP encoding of an image using convolution.
    Args:
        image (numpy.ndarray): Input grayscale image.
    Returns:
        numpy.ndarray: LBP-encoded image.
    """
    # Define the LBP weight kernel 
    kernel = [(2,2),(2,1),(2,0),(1,0),(0,0),(0,1),(0,2),(1,2)]
    
    # Create a padded version of the image for sliding window
    paddedVersion = np.pad(image, (1,1), 'constant', constant_values=(0, 0))

    # Initialize binary image for LBP decisions
    binaryImage = np.zeros((image.shape[0], image.shape[1]), dtype=int)

    # Iterate over each pixel in the image
    for row in range(paddedVersion.shape[0]-2):
        for col in range(paddedVersion.shape[1]-2):
            # Compare neighbors to center pixel and build binary pattern
            binaryToDec = ""
            centerValue = paddedVersion[row+1][col+1]
        
            for p in kernel:
                if paddedVersion[row+p[0]][col+p[1]] >= centerValue:
                    binaryToDec+="1"
                else:
                    binaryToDec+="0"

            # Compute the LBP value
            binaryImage[row][col] = int(binaryToDec, 2)        

    # return binary_image
    return binaryImage

# DONES