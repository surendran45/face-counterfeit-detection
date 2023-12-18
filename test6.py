'''# importing required libraries of opencv
import cv2
  
# importing library for plotting
from matplotlib import pyplot as plt
  
# reads an input image
img = cv2.imread('static/upload/CF1card.png',0)
  
# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.plot(histr)
plt.show()'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('static/upload/CF1card.png')
color = ('r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[2],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

'''import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
#import Image
import PIL.Image
from PIL import Image, ImageDraw, ImageFilter

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)

# Usig the original to get the same results
im = Image.open('static/upload/E2.png')

# image converted to double with max value of 1
data = np.array(im)/255.0
plot.i = 0
plot(data, 'Original')

# Create the filters
k3 = np.ones((3,3))/9.0
k5 = np.ones((5,5))/25.0

# Now the convolution
lp3 = ndimage.convolve(data, k3, mode='nearest')
lp5 = ndimage.convolve(data, k5, mode='nearest')

mse3 = np.sum(np.power(lp3-data,2))/np.size(data)
mse5 = np.sum(np.power(lp5-data,2))/np.size(data)

#~ PSNR = 10*np.log10(np.power(MAXi,2)/MSE);
psnr3 = 10*np.log10(np.power(1.0,2)/mse3);
psnr5 = 10*np.log10(np.power(1.0,2)/mse5);

print(psnr3,psnr5)

plot(lp3, '3x3 low-pass')
plot(lp5, '5x5 low-pass')
plt.show()
'''
