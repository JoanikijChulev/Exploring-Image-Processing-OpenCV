import cv2
import numpy as np

# Load image
image = cv2.imread("/Users/DAVID/Desktop/oh.jpg")


#TASK 1

# Halving the R values
halved_red = image.copy()
halved_red[:, :, 2] = halved_red[:, :, 2] // 2

# Setting R and G values to zero
red_and_green_zero = image.copy()
red_and_green_zero[:, :, 2] = 0
red_and_green_zero[:, :, 1] = 0

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Halved R Values', halved_red)
cv2.imshow('R and G Values Zero', red_and_green_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()


#TASK 2

# Convert to grayscale using different weighted sums
gray_average = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_weighted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_weighted[:, :] = 0.3 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0]

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale - Average', gray_average)
cv2.imshow('Grayscale - Weighted', gray_weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

#TASK 3

# Reduce resolution
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
reduced_resolution = cv2.resize(gray_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Display results
cv2.imshow('Original Image', gray_image)
cv2.imshow('Reduced Resolution', reduced_resolution)
cv2.waitKey(0)
cv2.destroyAllWindows()

#TASK 4

# Reduce resolution
reduced_resolution = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Reduced Resolution', reduced_resolution)
cv2.waitKey(0)
cv2.destroyAllWindows()