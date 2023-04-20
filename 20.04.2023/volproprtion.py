import cv2
import numpy as np

# Load the image and crop to the region of interest
img = cv2.imread('C:/Users/rajas/OneDrive/Desktop/ScaDS/20230414_154740.jpg')

# Define the desired output size
new_width = 640
new_height = 640

# Resize the image
img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

cv2.imshow('LIQUID', img)

glass_roi = cv2.selectROI(img)
glass_img = img[int(glass_roi[1]):int(glass_roi[1]+glass_roi[3]),
                int(glass_roi[0]):int(glass_roi[0]+glass_roi[2])]

# Convert to grayscale and preprocess the image
gray = cv2.cvtColor(glass_img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

hsv = cv2.cvtColor(glass_img, cv2.COLOR_BGR2HSV)
#lower_red = np.array([0, 50, 50])
upper_orange = np.array([20, 255, 255])
lower_orange = np.array([5, 0, 0])
mask = cv2.inRange(hsv, lower_orange, upper_orange)
cv2.imshow('maskimage', mask)
#cv2.imwrite('maskimage', mask)

# Find the contours of the liquid mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)

# Compute the area of the liquid mask and the area of the entire glass
liquid_area = cv2.contourArea(max_contour)
glass_area = glass_img.shape[0] * glass_img.shape[1]

# Calculate the proportion of liquid in the glass
liquid_proportion = liquid_area / glass_area
print('Liquid area',liquid_area)
print('Glass area', glass_area)

# Display the results
cv2.drawContours(glass_img, contours, -1, (0, 255, 0), 2)
#cv2.imshow('Liquid mask', thresh)
#cv2.imwrite('Liquid mask', thresh)
cv2.imshow('Glass with liquid contour', glass_img)
#cv2.imwrite('Glass with liquid contour', glass_img)
print("Proportion of liquid in the glass:", liquid_proportion)

cv2.waitKey()
#cv2.destroyAllWindows()
