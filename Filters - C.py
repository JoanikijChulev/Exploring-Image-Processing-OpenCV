import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread("/Users/DAVID/Desktop/ship.jpeg", cv2.IMREAD_GRAYSCALE)

#TASK 1


# Apply different 3x3 smoothing filters
box_filtered = cv2.boxFilter(image, -1, (3, 3))
gaussian_filtered = cv2.GaussianBlur(image, (3, 3), 0)
median_filtered = cv2.medianBlur(image, 3)

# Display results
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(box_filtered, cmap='gray')
plt.title('Box Filter')
plt.subplot(2, 2, 3)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('Gaussian Filter')
plt.subplot(2, 2, 4)
plt.imshow(median_filtered, cmap='gray')
plt.title('Median Filter')
plt.tight_layout()
plt.show()

#TASK 2


# Add random noise by setting intensity of random pixels to zero
num_noise_pixels = 2000
noise_indices = np.random.choice(image.size, num_noise_pixels, replace=False)
noisy_image = image.copy()
noisy_image.flat[noise_indices] = 0

# Apply median filters to repair the noisy image
median_3x3 = cv2.medianBlur(noisy_image, 3)
median_5x5 = cv2.medianBlur(noisy_image, 5)

# Display results
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.subplot(2, 2, 3)
plt.imshow(median_3x3, cmap='gray')
plt.title('Median Filter (3x3)')
plt.subplot(2, 2, 4)
plt.imshow(median_5x5, cmap='gray')
plt.title('Median Filter (5x5)')
plt.tight_layout()
plt.show()


# Function to add noise by setting intensity of 2x2 clusters of pixels to zero
def add_cluster_noise(image, num_clusters):
    noisy_image = image.copy()
    height, width = image.shape
    cluster_size = 2
    cluster_indices = np.random.randint(0, min(height, width) // cluster_size, size=(num_clusters, 2))
    for index in cluster_indices:
        row_index, col_index = index * cluster_size
        noisy_image[row_index:row_index+cluster_size, col_index:col_index+cluster_size] = 0
    return noisy_image

# Add noise to the original grayscale image
num_clusters = 2000
noisy_cluster_image = add_cluster_noise(image, num_clusters)

# Apply 3x3 and 5x5 median filters to repair the noisy image
median_3x3 = cv2.medianBlur(noisy_cluster_image, 3)
median_5x5 = cv2.medianBlur(noisy_cluster_image, 5)

# Display original image, noisy image, and filtered images
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 3, 2)
plt.imshow(noisy_cluster_image, cmap='gray')
plt.title('Noisy Image with Cluster Noise')
plt.subplot(2, 3, 4)
plt.imshow(median_3x3, cmap='gray')
plt.title('3x3 Median Filter')
plt.subplot(2, 3, 5)
plt.imshow(median_5x5, cmap='gray')
plt.title('5x5 Median Filter')

plt.tight_layout()
plt.show()

#TASK 3

# Apply unsharp masking
blurred = cv2.GaussianBlur(image, (5, 5), 10)
unsharp_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# Display results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(unsharp_mask, cmap='gray')
plt.title('Unsharp Masking')
plt.tight_layout()
plt.show()
