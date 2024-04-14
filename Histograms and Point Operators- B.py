import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread("/Users/DAVID/Desktop/ship.jpeg")


#TASK 1

# Calculate histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calculate cumulative histogram
cumulative_histogram = np.cumsum(histogram)

# Invert the grayscale image
inverted_image = 255 - image

# Plot original image, histogram, and cumulative histogram
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.plot(histogram)
plt.title('Histogram')
plt.subplot(2, 2, 3)
plt.plot(cumulative_histogram)
plt.title('Cumulative Histogram')
plt.subplot(2, 2, 4)
plt.imshow(inverted_image, cmap='gray')
plt.title('Inverted Image')
plt.tight_layout()
plt.show()

#TASK 2

# Load dark and light grayscale images
dark_image = cv2.imread("/Users/DAVID/Desktop/dark.jpg", cv2.IMREAD_GRAYSCALE)
light_image = cv2.imread("/Users/DAVID/Desktop/li.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate histograms
dark_histogram = cv2.calcHist([dark_image], [0], None, [256], [0, 256])
light_histogram = cv2.calcHist([light_image], [0], None, [256], [0, 256])

# Calculate cumulative histograms
dark_cumulative_histogram = np.cumsum(dark_histogram)
light_cumulative_histogram = np.cumsum(light_histogram)

# Plot histograms and cumulative histograms separately
plt.figure(figsize=(10, 8))

# Dark Image
plt.subplot(2, 3, 1)
plt.imshow(dark_image, cmap='gray')
plt.title('Dark Image')
plt.subplot(2, 3, 2)
plt.plot(dark_histogram, color='blue')
plt.title('Dark Image Histogram')
plt.subplot(2, 3, 3)
plt.plot(dark_cumulative_histogram, color='red')
plt.title('Dark Image Cumulative Histogram')

# Light Image
plt.subplot(2, 3, 4)
plt.imshow(light_image, cmap='gray')
plt.title('Light Image')
plt.subplot(2, 3, 5)
plt.plot(light_histogram, color='blue')
plt.title('Light Image Histogram')
plt.subplot(2, 3, 6)
plt.plot(light_cumulative_histogram, color='red')
plt.title('Light Image Cumulative Histogram')

plt.show()

#TASK 3

# Perform histogram equalization
dark_equalized = cv2.equalizeHist(dark_image)
light_equalized = cv2.equalizeHist(light_image)

# Calculate histograms of equalized images
dark_equalized_hist = cv2.calcHist([dark_equalized], [0], None, [256], [0, 256])
light_equalized_hist = cv2.calcHist([light_equalized], [0], None, [256], [0, 256])

# Calculate cumulative histograms of equalized images
dark_equalized_cumulative = np.cumsum(dark_equalized_hist)
light_equalized_cumulative = np.cumsum(light_equalized_hist)

# Plot equalized images, histograms, and cumulative histograms
plt.figure(figsize=(10, 8))

# Dark Image Equalized
plt.subplot(2, 3, 1)
plt.imshow(dark_equalized, cmap='gray')
plt.title('Dark Image Equalized')
plt.subplot(2, 3, 2)
plt.plot(dark_equalized_hist, color='blue')
plt.title('Dark Image Equalized Histogram')
plt.subplot(2, 3, 3)
plt.plot(dark_equalized_cumulative, color='red')
plt.title('Dark Image Equalized Cumulative Histogram')

# Light Image Equalized
plt.subplot(2, 3, 4)
plt.imshow(light_equalized, cmap='gray')
plt.title('Light Image Equalized')
plt.subplot(2, 3, 5)
plt.plot(light_equalized_hist, color='blue')
plt.title('Light Image Equalized Histogram')
plt.subplot(2, 3, 6)
plt.plot(light_equalized_cumulative, color='red')
plt.title('Light Image Equalized Cumulative Histogram')

plt.tight_layout()
plt.show()



#TASK 4

# Load the original image
original_image = cv2.imread("/Users/DAVID/Desktop/sea.jpg")

# Increase brightness (add a constant value)
increased_brightness = cv2.convertScaleAbs(original_image, alpha=1.5, beta=50)

# Increase contrast (multiply by a constant value)
increased_contrast = cv2.convertScaleAbs(original_image, alpha=1.5, beta=0)

# Display results
cv2.imshow('Original Image', original_image)
cv2.imshow('Increased Brightness', increased_brightness)
cv2.imshow('Increased Contrast', increased_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
