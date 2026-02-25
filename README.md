# Image Processing: Nearest Neighbor & Local Binary Patterns (LBP)

A Python implementation of fundamental image processing algorithms from scratch using NumPy. This project demonstrates spatial transformations (resizing) and texture analysis (feature extraction) without the use of high-level computer vision libraries.

---

## 🛠 Features

### 1. Nearest-Neighbor Resizing
A spatial transformation algorithm that resizes grayscale images by mapping output pixels to the nearest corresponding pixel in the input image.
* **Mathematical Approach:** Uses scaling factors for rows and columns to determine the inverse mapping.
* **Robustness:** Includes boundary handling to prevent index-out-of-bounds errors.

### 2. Local Binary Patterns (LBP) Encoding
A powerful feature extraction method used for texture classification and facial recognition.
* **Mechanism:** Compares each pixel to its 8 neighbors in a $3 \times 3$ neighborhood.
* **Encoding:** Generates an 8-bit binary string based on intensity comparisons, which is then converted to a decimal value.
* **Padding:** Uses constant zero-padding to ensure the output image maintains the same dimensions as the input.



---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* NumPy

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/image-processing-scratch.git](https://github.com/your-username/image-processing-scratch.git)