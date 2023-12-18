## code to plot histogram in python
import numpy as np

'''import cv2
from matplotlib import pyplot as plt
img = cv2.imread('static/upload/CF1card.png',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()'''

# import necessary packages
import cv2
import matplotlib.pyplot as plt
 
# load image
imageObj = cv2.imread('static/upload/E1.png')
im_size=imageObj.shape
w=im_size[1]
h=im_size[0]
# to avoid grid lines
plt.axis("off")
plt.title("Original Image")
plt.imshow(cv2.cvtColor(imageObj, cv2.COLOR_BGR2RGB))
#plt.show()
 
# Get RGB data from image
blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])
 
# Separate Histograms for each color
plt.subplot(3, 1, 1)
plt.title("histogram of Blue")
plt.hist(blue_color, color="blue")
plt.xlim([0,w])
plt.ylim([0,h])

plt.subplot(3, 1, 2)
plt.title("histogram of Green")
plt.hist(green_color, color="green")
plt.xlim([0,w])
plt.ylim([0,h])

plt.subplot(3, 1, 3)
plt.title("histogram of Red")
plt.hist(red_color, color="red")
plt.xlim([0,w])
plt.ylim([0,h])

# for clear view
plt.tight_layout()
plt.show()
 
# combined histogram
plt.title("Histogram of all RGB Colors")
plt.hist(blue_color, color="blue")
plt.hist(green_color, color="green")
plt.hist(red_color, color="red")
#plt.show()
