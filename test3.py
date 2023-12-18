from PIL import Image, ImageDraw, ImageFilter
import cv2
import piexif
import imagehash
##crop from embed
''''
# Detect the faces
image = cv2.imread("static/upload/EM2.png")
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
    image = cv2.imread("static/upload/EM2.png")
    cropped = image[y:y+h, x:x+w]
    gg="D.png"
    cv2.imwrite("static/upload/"+gg, cropped)
##################'''

'''im = Image.open("static/upload/D.png")
vv=piexif.load(im.info["exif"])["0th"]\
    [piexif.ImageIFD.ImageDescription].decode("utf-8")
print(vv)'''
################

cutoff=3
hash0 = imagehash.average_hash(Image.open("static/upload/E2.png")) 
hash1 = imagehash.average_hash(Image.open("static/upload/D.png"))
cc1=hash0 - hash1
print("cc="+str(cc1))
if cc1<=cutoff:
    print("yes")
else:
    print("no")





