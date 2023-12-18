from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import math
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
from flask_mail import Mail, Message
from flask import send_file
import smtplib
import socket

import numpy as np
from matplotlib import pyplot as plt
import cv2
import threading
import os
import time
import shutil
import hashlib
import imagehash
import PIL.Image
from PIL import Image, ImageDraw, ImageFilter
import piexif

import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="stegocard"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
##email
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": "stegofaceidissuer@gmail.com",
    "MAIL_PASSWORD": "pwxzxzkmnyygrakr"
}

app.config.update(mail_settings)
mail = Mail(app)
#######
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("")
    ff.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM sb_admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            result=" Your Logged in sucessfully**"
            return redirect(url_for('view_ins')) 
        else:
            result="Your logged in fail!!!"
        

    return render_template('index.html',msg=msg,act=act)

@app.route('/login',methods=['POST','GET'])
def login():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM sf_admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            result=" Your Logged in sucessfully**"
            return redirect(url_for('userhome')) 
        else:
            result="Your logged in fail!!!"
        

    return render_template('login.html',msg=msg,act=act)

@app.route('/login2',methods=['POST','GET'])
def login2():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM sf_register where uname=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            result=" Your Logged in sucessfully**"
            return redirect(url_for('home'))
            
        else:
            result="Your logged in fail!!!"
        

    return render_template('login2.html',msg=msg,act=act)



@app.route('/register',methods=['POST','GET'])
def register():
    result=""
    act=""
    mycursor = mydb.cursor()
    
    
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
       
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        #mycursor.execute("SELECT count(*) FROM sf_files where uname=%s",(uname, ))
        #cnt = mycursor.fetchone()[0]
        #if cnt==0:
        mycursor.execute("SELECT max(id)+1 FROM sf_files")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO sf_files(id, uname, mobile, email, rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, name, mobile, email, rdate)
        act="success"
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "record inserted.")
        return redirect(url_for('upload',fid=maxid))
            
        #else:
        #    act="wrong"
        #    result="Reg No. Already Exist!"
    return render_template('register.html',act=act,result=result)

@app.route('/add_verifier',methods=['POST','GET'])
def add_verifier():
    result=""
    act=""
    mycursor = mydb.cursor()
    
    
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
       
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM sf_register where uname=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM sf_register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO sf_register(id, name, mobile, email, uname, pass) VALUES (%s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, uname, pass1)
            act="success"
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            message="Dear "+name+", Document Identity Verifier account - Username:"+uname+", Password:"+pass1
            url="http://iotcloud.co.in/testmail/sendmail.php?email="+email+"&message="+message
            webbrowser.open_new(url)
            return redirect(url_for('add_verifier',fid=maxid))
                
        else:
            act="wrong"
            result="Already Exist!"

    mycursor.execute("SELECT * FROM sf_register")
    value = mycursor.fetchall()
    
    return render_template('add_verifier.html',act=act,value=value)

@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    uname=""
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    act = request.args.get('act')
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM sf_files")
    value = mycursor.fetchall()
    
    if act=="del":
        did = request.args.get('did')
        mycursor.execute("delete from sf_files where id=%s",(did, ))
        mydb.commit()
        return redirect(url_for('userhome'))
        
   
    
        
    return render_template('userhome.html',value=value)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    uname=""
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    fid = request.args.get('fid')
    
    mycursor = mydb.cursor()
    

    #mycursor.execute("SELECT max(id)+1 FROM sf_files")
    #maxid = mycursor.fetchone()[0]
    #if maxid is None:
    #    maxid=1
                        
    if request.method=='POST':
        
        file = request.files['file']
        print(file.filename)
        #try:
        #    if file.filename == '':
        #        flash('No selected file')
        #        return redirect(request.url)
        #if file:
        fn=file.filename
        fnn="F"+fid+fn
        #fnn = secure_filename(fn1)
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fnn))
        print("upload")
        filename2 = 'static/upload/'+fnn
        
        #sql = "INSERT INTO sf_files(id, uname, filename) VALUES (%s, %s, %s)"
        #val = (maxid, uid, fnn)
        #print(val)
        #mycursor.execute(sql, val)
        #mydb.commit()
        mycursor.execute("update sf_files set filename=%s where id=%s", (fnn,fid))
        mydb.commit()

        ###RPN Face Detection
        
        ##
        # Detect the faces
        image = cv2.imread("static/upload/"+fnn)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw the rectangle around each face
        j = 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rectface="R"+fnn
            cv2.imwrite("static/upload/"+rectface, mm)
            image = cv2.imread("static/upload/"+rectface)
            #cropped = image[y:y+h, x:x+w]
            #gg="C"+fnn
            #cv2.imwrite("static/upload/"+gg, cropped)
            #mm2 = PIL.Image.open("static/upload/"+gg)
            #rz = mm2.resize((100,100), PIL.Image.ANTIALIAS)
            #rz.save("static/upload/"+gg)
            j += 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #rectface="S"+fnn
            #cv2.imwrite("static/upload/"+rectface, mm)
            image = cv2.imread("static/upload/"+fnn)
            cropped = image[y:y+h, x:x+w]
            gg="C"+fnn
            cv2.imwrite("static/upload/"+gg, cropped)
            

        msg2="Uploaded Success"
        return redirect(url_for('process1',fid=fid))
        #except:
        #    print("dd")
    
        
    return render_template('upload.html')

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    uname=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    mycursor = mydb.cursor()
    

    mycursor.execute("SELECT * FROM sf_files where id=%s",(fid, ))
    data = mycursor.fetchone()
    fname="R"+data[2]

    if request.method=='POST':
        return redirect(url_for('process2',fid=fid))
        
    return render_template('process1.html',fname=fname)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    uname=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    mycursor = mydb.cursor()
   

    mycursor.execute("SELECT * FROM sf_files where id=%s",(fid, ))
    data = mycursor.fetchone()
    fname="C"+data[2]
    eimg="E"+fid+".png"
    print(fname)

    ####
    imageObj = cv2.imread('static/upload/'+fname)
    im_size=imageObj.shape
    w=im_size[1]
    h=im_size[0]
    blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
    green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])
     
    # Separate Histograms for each color
    '''plt.subplot(3, 1, 1)
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
    plt.ylim([0,h])'''

    # for clear view
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('static/graph/cover_graph.png')
    #plt.close()
    ####

    if request.method=='POST':
        message=request.form['message']
        binv=toBinary(message)
        lenb=len(binv)
        i=0
        bindata=''
        while i<lenb:
            bindata+=str(binv[i])
            i+=1
        mycursor.execute("update sf_files set message=%s,bindata=%s where id=%s", (message,bindata,fid))
        mydb.commit()
        ############
        # Enter the data to be transmitted
        data = bindata #'1011001'
         
        # Calculate the no of Redundant Bits Required
        m = len(data)
        r = calcRedundantBits(m)
         
        # Determine the positions of Redundant Bits
        arr = posRedundantBits(data, r)
         
        # Determine the parity bits
        arr = calcParityBits(arr, r)
         
        # Data to be transferred
        print("Data transferred is " + arr) 
         
        # Stimulate error in transmission by changing
        # a bit value.
        # 10101001110 -> 11101001110, error in 10th position.
         
        #arr = '11101001110'
        print("Error Data is " + arr)
        correction = detectError(arr, r)
        print("The position of error is " + str(correction))

        ######################################
        im = Image.open("static/upload/"+fname)
        if "exif" in im.info:
            exif_dict = piexif.load(im.info["exif"])
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = bindata
            exif_bytes = piexif.dump(exif_dict)
        else:
            exif_bytes = piexif.dump({"0th":{piexif.ImageIFD.ImageDescription:bindata}})

        im.save("static/upload/"+eimg, exif=exif_bytes)
        
        #####
        return redirect(url_for('binary_process',fid=fid))
        
    return render_template('process2.html',fid=fid,fname=fname)
#####################
def DeepCon_auto_encode():
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
    # summarize the dataset
    #print(X.shape, y.shape)

    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)

    
    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    n_bottleneck = n_inputs
    bottleneck = Dense(n_bottleneck)(e)

    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

def DeepCon_auto_decode():
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
    # summarize the dataset
    #print(X.shape, y.shape)
    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)

def calcRedundantBits(m):
 
    # Use the formula 2 ^ r >= m + r + 1
    # to calculate the no of redundant bits.
    # Iterate over 0 .. m and return the value
    # that satisfies the equation
 
    for i in range(m):
        if(2**i >= m + i + 1):
            return i
 
 
def posRedundantBits(data, r):
 
    # Redundancy bits are placed at the positions
    # which correspond to the power of 2.
    j = 0
    k = 1
    m = len(data)
    res = ''
 
    # If position is power of 2 then insert '0'
    # Else append the data
    for i in range(1, m + r+1):
        if(i == 2**j):
            res = res + '0'
            j += 1
        else:
            res = res + data[-1 * k]
            k += 1
 
    # The result is reversed since positions are
    # counted backwards. (m + r+1 ... 1)
    return res[::-1]
 
 
def calcParityBits(arr, r):
    n = len(arr)
 
    # For finding rth parity bit, iterate over
    # 0 to r - 1
    for i in range(r):
        val = 0
        for j in range(1, n + 1):
 
            # If position has 1 in ith significant
            # position then Bitwise OR the array value
            # to find parity bit value.
            if(j & (2**i) == (2**i)):
                val = val ^ int(arr[-1 * j])
                # -1 * j is given since array is reversed
 
        # String Concatenation
        # (0 to n - 2^r) + parity bit + (n - 2^r + 1 to n)
        arr = arr[:n-(2**i)] + str(val) + arr[n-(2**i)+1:]
    return arr
def detectError(arr, nr):
    n = len(arr)
    res = 0
 
    # Calculate parity bits again
    for i in range(nr):
        val = 0
        for j in range(1, n + 1):
            if(j & (2**i) == (2**i)):
                val = val ^ int(arr[-1 * j])
 
        # Create a binary no by appending
        # parity bits together.
 
        res = res + val*(10**i)
 
    # Convert binary to decimal
    return int(str(res), 2)
#Convert String to Binary   
def toBinary(a):
  l,m=[],[]
  for i in a:
    l.append(ord(i))
  for i in l:
    m.append(int(bin(i)[2:]))
  return m
#Convert Binary to String
def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
####################
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  


@app.route('/binary_process', methods=['GET', 'POST'])
def binary_process():
    uname=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    mycursor = mydb.cursor()
   
    mycursor.execute("SELECT * FROM sf_files where id=%s",(fid, ))
    data = mycursor.fetchone()
    fname="C"+data[2]
    #fname="aadr2.jpg"
    eimg="E"+fid+".png"
    bindata=data[4]
    print(fname)

    message=str(bindata)

    #####
    ####
    imageObj = cv2.imread('static/upload/'+eimg)
    im_size=imageObj.shape
    w=im_size[1]
    h=im_size[0]
    blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
    green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])
     
    # Separate Histograms for each color
    '''plt.subplot(3, 1, 1)
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
    plt.ylim([0,h])'''

    # for clear view
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('static/graph/stego_graph.png')
    #plt.close()
    ####
    
    if request.method=='POST':
        
        return redirect(url_for('embed_img',fid=fid))
    

    return render_template('binary_process.html',fname=fname,bindata=bindata)

@app.route('/embed_img', methods=['GET', 'POST'])
def embed_img():
    uname=""
    act=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    
    
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT * FROM sf_files where id=%s",(fid, ))
    data = mycursor.fetchone()
    name=data[1]
    fnn=data[2]
    fnn1="C"+fnn
    fname2="R"+data[2]
    fname="E"+fid+".png"
    em="EM"+fid+".png"
    sm="SM"+fid+".png"
    bindata=data[4]
    
    #####
    #PSNR Calc
    original = cv2.imread("static/upload/"+fnn1)
    compressed = cv2.imread("static/upload/"+fname, 1)
    value = PSNR(original, compressed)
    psnr="PSNR value is "+str(value)+" dB"
    print(psnr)
    #print(f"PSNR value is {value} dB")

    ######

    if request.method=='POST':
        ##
        im1 = Image.open('static/upload/'+fnn)
        im2 = Image.open('static/upload/'+fname)

        # Detect the faces
        image = cv2.imread("static/upload/"+fnn)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        x=0
        y=0
        # Draw the rectangle around each face
        j = 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        back_im = im1.copy()
        back_im.paste(im2, (x, y))
        back_im.save('static/upload/'+em, quality=95)
            
        ####
        im = Image.open("static/upload/"+em)
        if "exif" in im.info:
            exif_dict = piexif.load(im.info["exif"])
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = name
            exif_bytes = piexif.dump(exif_dict)
        else:
            exif_bytes = piexif.dump({"0th":{piexif.ImageIFD.ImageDescription:name}})

        im.save("static/upload/"+sm, exif=exif_bytes)
        ####
        
        act="1"
        return redirect(url_for('send',fid=fid,act=act))
    

    return render_template('embed_img.html',fname=fname,fname2=fname2,act=act,psnr=psnr)

@app.route('/send', methods=['GET', 'POST'])
def send():
    uname=""
    act=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    
    
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT * FROM sf_files where id=%s",(fid, ))
    data = mycursor.fetchone()
    name=data[1]
    email1=data[6]
    sm="SM"+fid+".png"

    if request.method=='POST':
        email=request.form['email']
        print(email)
        ##send mail
        mess="Dear "+name+", Your Encoded Stego ID"
        with app.app_context():
            msg = Message(subject="Stego Face", sender=app.config.get("MAIL_USERNAME"),recipients=[email], body=mess)
            with app.open_resource("static/upload/"+sm) as fp:  
                msg.attach("static/upload/"+sm, "images/png", fp.read())
            mail.send(msg)
        act="1"
    
    return render_template('send.html',fname=sm,fid=fid,act=act,email1=email1)

@app.route('/home', methods=['GET', 'POST'])
def home():
    uname=""
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    fid=""
    
    mycursor = mydb.cursor()
    

    #mycursor.execute("SELECT max(id)+1 FROM sf_files")
    #maxid = mycursor.fetchone()[0]
    #if maxid is None:
    #    maxid=1
                        
    if request.method=='POST':
        
        file = request.files['file']
        print(file.filename)
        #try:
        #    if file.filename == '':
        #        flash('No selected file')
        #        return redirect(request.url)
        #if file:
        fn=file.filename
        fnn="m1.png"
        #fnn = secure_filename(fn1)
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fnn))
        print("upload")
        filename2 = 'static/upload/'+fnn
        
        
        # Detect the faces
        image = cv2.imread("static/upload/"+fnn)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw the rectangle around each face
        j = 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rectface="r1.png"
            cv2.imwrite("static/upload/"+rectface, mm)
            image = cv2.imread("static/upload/"+rectface)
            #cropped = image[y:y+h, x:x+w]
            #gg="C"+fnn
            #cv2.imwrite("static/upload/"+gg, cropped)
            #mm2 = PIL.Image.open("static/upload/"+gg)
            #rz = mm2.resize((100,100), PIL.Image.ANTIALIAS)
            #rz.save("static/upload/"+gg)
            j += 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #rectface="S"+fnn
            #cv2.imwrite("static/upload/"+rectface, mm)
            image = cv2.imread("static/upload/"+fnn)
            cropped = image[y:y+h, x:x+w]
            gg="f1.png"
            cv2.imwrite("static/upload/"+gg, cropped)
            

        msg2="Uploaded Success"
        return redirect(url_for('decode1'))
        #except:
        #    print("dd")
    
        
    return render_template('home.html')


###Deep Convolutional





@app.route('/decode1', methods=['GET', 'POST'])
def decode1():
    uname=""
    
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    mycursor = mydb.cursor()
    
    fname="r1.png"
    

    if request.method=='POST':
        return redirect(url_for('decode2'))
        
    return render_template('decode1.html',fname=fname)

@app.route('/decode2', methods=['GET', 'POST'])
def decode2():
    uname=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    name=""
    st="2"
    print(uname)
    mycursor = mydb.cursor()
   

    fname="f1.png"
    
    
    
    if request.method=='POST':
        
    
        return redirect(url_for('decode3'))
        
    return render_template('decode2.html',fname=fname,st=st)

@app.route('/decodeBM', methods=['GET', 'POST'])
def decodeBM():
    ##Decode Binary Message 
    vv=""
    im = Image.open("static/upload/m1.png")
    vv=piexif.load(im.info["exif"])["0th"]\
        [piexif.ImageIFD.ImageDescription].decode("utf-8")
    
    ff=open("det.txt","w")
    ff.write(vv)
    ff.close()
    return render_template('decodeBM.html',fname=fname,st=st)

@app.route('/decode3', methods=['GET', 'POST'])
def decode3():
    uname=""
    act=""
    if 'username' in session:
        uname = session['username']
    name=""
    bindata=""
    rid=0
    print(uname)
    mycursor = mydb.cursor()
   

    fname="f1.png"
    ff=open("det.txt","r")
    vv=ff.read()
    ff.close()
    cutoff=3

    mycursor.execute("SELECT * FROM sf_files")
    dt = mycursor.fetchall()
    for rr in dt:
        ff="E"+str(rr[0])+".png"
        hash0 = imagehash.average_hash(Image.open("static/upload/"+ff)) 
        hash1 = imagehash.average_hash(Image.open("static/upload/f1.png"))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<=cutoff:
            print("yes")
            st="1"
            rid=rr[0]
            break
        else:
            st="2"
            print("no")
    if st=="1":
        mycursor.execute("SELECT * FROM sf_files where id=%s",(rid, ))
        dtt = mycursor.fetchone()
        if vv==dtt[1]:
            bindata=dtt[4]
            act="1"
    else:
        act=""
            
    if request.method=='POST':
        
    
        return redirect(url_for('decode4'))
        
    return render_template('decode3.html',fname=fname,bindata=bindata,act=act)



@app.route('/decode4', methods=['GET', 'POST'])
def decode4():
    uname=""
    
    act=""
    if 'username' in session:
        uname = session['username']
    name=""
    bindata=""
    message=""
    rid=0
    print(uname)
    mycursor = mydb.cursor()
   

    fname="f1.png"
    ff=open("det.txt","r")
    vv=ff.read()
    ff.close()
    
    cutoff=3

    mycursor.execute("SELECT * FROM sf_files")
    dt = mycursor.fetchall()
    for rr in dt:
        ff="E"+str(rr[0])+".png"
        hash0 = imagehash.average_hash(Image.open("static/upload/"+ff)) 
        hash1 = imagehash.average_hash(Image.open("static/upload/f1.png"))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<=cutoff:
            print("yes")
            st="1"
            rid=rr[0]
            break
        else:
            st="2"
            print("no")
    if st=="1":
        mycursor.execute("SELECT * FROM sf_files where id=%s",(rid, ))
        dtt = mycursor.fetchone()
        if vv==dtt[1]:
            bindata=dtt[4]
            message=dtt[3]
            act="1"
    else:
        act=""
            
    
    return render_template('decode4.html',fname=fname,message=message,act=act)

@app.route('/guest', methods=['GET', 'POST'])
def guest():
    uname=""
    
    fid=""
    
    mycursor = mydb.cursor()
    

    if request.method=='POST':
        
        file = request.files['file']
        
        fn=file.filename
        fnn="m1.png"
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fnn))
        print("upload")
        filename2 = 'static/upload/'+fnn
        
        
        # Detect the faces
        image = cv2.imread("static/upload/"+fnn)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw the rectangle around each face
        j = 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rectface="r1.png"
            cv2.imwrite("static/upload/"+rectface, mm)
            image = cv2.imread("static/upload/"+rectface)
            j += 1
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            image = cv2.imread("static/upload/"+fnn)
            cropped = image[y:y+h, x:x+w]
            gg="f1.png"
            cv2.imwrite("static/upload/"+gg, cropped)
            

        msg2="Uploaded Success"
        return redirect(url_for('page1'))
        #except:
        #    print("dd")
    
        
    return render_template('guest.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    uname=""
    
    if 'username' in session:
        uname = session['username']
    name=""
    print(uname)
    mycursor = mydb.cursor()
    
    fname="r1.png"
    

    if request.method=='POST':
        return redirect(url_for('page2'))
        
    return render_template('page2.html',fname=fname)

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    uname=""
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    name=""
    st="2"
    print(uname)
    mycursor = mydb.cursor()
   

    fname="f1.png"
    
    
    
    if request.method=='POST':
        
    
        return redirect(url_for('page3'))
        
    return render_template('page2.html',fname=fname,st=st)


@app.route('/page3', methods=['GET', 'POST'])
def page3():
    uname=""
    act=""
    if 'username' in session:
        uname = session['username']
    name=""
    bindata=""
    rid=0
    print(uname)
    mycursor = mydb.cursor()
   

    fname="f1.png"
    ff=open("det.txt","r")
    vv=ff.read()
    ff.close()
    cutoff=3

    mycursor.execute("SELECT * FROM sf_files")
    dt = mycursor.fetchall()
    for rr in dt:
        ff="E"+str(rr[0])+".png"
        hash0 = imagehash.average_hash(Image.open("static/upload/"+ff)) 
        hash1 = imagehash.average_hash(Image.open("static/upload/f1.png"))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<=cutoff:
            print("yes")
            st="1"
            rid=rr[0]
            break
        else:
            st="2"
            print("no")
    if st=="1":
        mycursor.execute("SELECT * FROM sf_files where id=%s",(rid, ))
        dtt = mycursor.fetchone()
        if vv==dtt[1]:
            bindata=dtt[4]
            act="1"
    else:
        act=""
            
    if request.method=='POST':
        
    
        return redirect(url_for('page4'))
        
    return render_template('page3.html',fname=fname,bindata=bindata,act=act)



@app.route('/page4', methods=['GET', 'POST'])
def page4():
    uname=""
    
    act=""
    if 'username' in session:
        uname = session['username']
    name=""
    bindata=""
    message=""
    rid=0
    print(uname)
    mycursor = mydb.cursor()
   

    fname="f1.png"
    ff=open("det.txt","r")
    vv=ff.read()
    ff.close()
    
    cutoff=3

    mycursor.execute("SELECT * FROM sf_files")
    dt = mycursor.fetchall()
    for rr in dt:
        ff="E"+str(rr[0])+".png"
        hash0 = imagehash.average_hash(Image.open("static/upload/"+ff)) 
        hash1 = imagehash.average_hash(Image.open("static/upload/f1.png"))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<=cutoff:
            print("yes")
            st="1"
            rid=rr[0]
            break
        else:
            st="2"
            print("no")
    if st=="1":
        mycursor.execute("SELECT * FROM sf_files where id=%s",(rid, ))
        dtt = mycursor.fetchone()
        if vv==dtt[1]:
            bindata=dtt[4]
            message=dtt[3]
            act="1"
    else:
        act=""
            
    
    return render_template('page4.html',fname=fname,message=message,act=act)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
