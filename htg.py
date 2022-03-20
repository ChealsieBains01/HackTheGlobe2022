# import the necessary packages
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2
import pandas as pd
import openpyxl

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# load the input image, convert it to grayscale, and grab its
# dimensions
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(H, W) = gray.shape
# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
# smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morpholigical operator to find dark regions on a light
# background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# cv2.imshow("Blackhat", blackhat)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
# cv2.imshow("Gradient", grad)

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("Rect Close", thresh)
# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=2)
# cv2.imshow("Square Close", thresh)

# find contours in the thresholded image and sort them from bottom
# to top (since the MRZ will always be at the bottom of the passport)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="bottom-to-top")[0]
# initialize the bounding box associated with the MRZ
mrzBox = None

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then derive the
	# how much of the image the bounding box occupies in terms of
	# both width and height
	(x, y, w, h) = cv2.boundingRect(c)
	percentWidth = w / float(W)
	percentHeight = h / float(H)
	# if the bounding box occupies > 80% width and > 4% height of the
	# image, then assume we have found the MRZ
	if percentWidth > 0.8 and percentHeight > 0.04:
		mrzBox = (x, y, w, h)
		break

    # if the MRZ was not found, exit the script
if mrzBox is None:
	print("[INFO] MRZ could not be found")
	sys.exit(0)
# pad the bounding box since we applied erosions and now need to
# re-grow it
(x, y, w, h) = mrzBox
print(x, y, w, h)
pX = int((x + w) * 0.03)
pY = int((y + h) * 0.03)
(x, y) = (max(x - pX, 0), max(y - pY,0))
(w, h) = (w + (pX * 2), h + (pY * 2))
# extract the padded MRZ from the image
mrz = image[y:y + h, x:x + w]

mrz = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY )
mrz = cv2.resize(mrz, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

# OCR the MRZ region of interest using Tesseract, removing any
# occurrences of spaces
mrzText = pytesseract.image_to_string(mrz)
mrzText = mrzText.replace(" ", "")
print(mrzText)

if mrzText[0] == "P":
	print("document is passport")
country_code = mrzText[2:5] #index 2,3,4 gives country code
print(country_code)

index = 5
while mrzText[index] != "<":
	index += 1
last_name = mrzText[5:index] #index 5 to index-1 gives last name
print(last_name)
index += 2

first_name = ""
while mrzText[index] != "<" and mrzText[index] != '\n':
	#keep reading first name
	first_index = index
	while mrzText[index] != "<":
		index += 1
	first_name = first_name + " " + mrzText[first_index: index]
	index += 1
first_name = first_name[1:]
print(first_name)

while mrzText[index] != '\n':
	index += 1

index += 14 #skip to DoB
DoB = mrzText[index:index+6]
index += 7
print(DoB)
Gender = mrzText[index]
print(Gender)

df = pd.DataFrame([[country_code], [first_name], [last_name], [Gender], [DoB]],
                  index=['Origin Country', 'First Name', 'Last Name', 'Gender', 'Date of Birth'], columns=['Personal Info'])

df.to_excel('passport_data.xlsx', sheet_name='personal info')
# for element in range(, len(mrzText)):
#     print(mrzText[element])

# show the MRZ image
image = cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
cv2.imshow("MRZ", image)
cv2.waitKey(0)



