from PIL import Image, ImageDraw, ImageFilter
import cv2
##embed img
im1 = Image.open('static/upload/F2card2.png')
im2 = Image.open('static/upload/E2.png')

# Detect the faces
image = cv2.imread("static/upload/F2card2.png")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

x=0
y=0
# Draw the rectangle around each face
j = 1
for (x, y, w, h) in faces:
    mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #rectface="S"+fnn
    #cv2.imwrite("static/upload/"+rectface, mm)
    #image = cv2.imread("static/upload/"+fnn)
    #cropped = image[y:y+h, x:x+w]
    #gg="C"+fnn
    #cv2.imwrite("static/upload/"+gg, cropped)

back_im = im1.copy()
back_im.paste(im2, (x, y))
back_im.save('static/upload/ex.png', quality=95)
