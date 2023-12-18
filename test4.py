'''import math
import cv2
import numpy as np

original = cv2.imread("static/upload/E2.png")
contrast = cv2.imread("static/upload/CF2card2.png", 1)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(original, contrast)
print(d)'''
#################
from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
#def main():
original = cv2.imread("static/upload/CF1card.png")
compressed = cv2.imread("static/upload/E1.png", 1)
value = PSNR(original, compressed)
print(f"PSNR value is {value} dB")
       
#if __name__ == "__main__":
#    main()
