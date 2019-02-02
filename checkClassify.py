import cv2
import sys
import os
import imutils
import numpy as np
import subprocess
from imutils import contours
from imutils.perspective import four_point_transform

strB1 = "ก่อนอาหาร"
strA1 = "หลังอาหาร"
strA2 = "หลังอาหาธ"
str2 = "เช้า"
str3 = "กลางวัน"
str4 = "เย็น"

def text_from_image_file(image_name,lang):
    output_name = "OutputImg"
    return_code = subprocess.call(['tesseract',image_name,output_name,'-l',lang,'-c','preserve_interword_spaces=1 --tessdata-dir ./tessdata_best/'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = open(output_name+'.txt','r',encoding='utf-8')
    return d.read()

def More_Gray(gamma,image) : #make picture more clearly
    gamma1 = gamma
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma1) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    return res

def Spell_checker(name):
    f = open(name + ".txt")
    
    line = f.readline()
    while line:
        if(line.find(strB1) > 0):
            print ('ก่อนอาหาร')
            if(line.find(str2) >0):
                print('เช้า')
            if(line.find(str3) >0):
                print('กลางวัน')
            if(line.find(str4) >0):
                print('เย็น')
        if(line.find(strA1) > 0 or line.find(strA2) > 0):
            print ('หลังอาหาร')
            if(line.find(str2) >0):
                print('เช้า')
            if(line.find(str3) >0):
                print('กลางวัน')
            if(line.find(str4) >0):
                print('เย็น')
        line = f.readline()

def cvt_to_JSON(_isPeriod, _isEatBefore,_isEatBreakfast, _isEatLunch, _isEatDinner, _isEatBedTime, _isRoutine, _periodHour) :
    output = {}
    output["isPeriod"] = _isPeriod
    data = {}
    data["isEatBefore"] = _isEatBefore
    data["isEatBreakfast"] = _isEatBreakfast
    data["isEatLunch"] = _isEatLunch
    data["isEatDinner"] = _isEatDinner
    data["isEatBedTime"] = _isEatBedTime
    output["data"] = data
    conv_json = json.dumps(output, ensure_ascii = False)
    print(conv_json)

def main(argv) :
    image = cv2.imread(argv[0]) 
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = More_Gray(3,gray) #make picture more clear
    blurred = cv2.GaussianBlur(gray, (5 , 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    kernel = np.ones((6,6),np.uint8)
    dilation = cv2.dilate(edged,kernel,iterations = 1)
    # cv2.cimshow('dilation' , dilation)
    contourmask,contours,hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    fname = argv[0].split(".")[0]
    temp = image
    
    with open(fname+".txt","w") as f:
        for cnt in contours[1:] :
            x, y, w, h = cv2.boundingRect(cnt)
            # if (h / w < 0.7 and h * w > 500) :
            if h > 30 and h < 50 :
            # if h * w > 100000 :
                cv2.rectangle(temp,(x,y),(x+w,y+h),(0,0,255),2)
                roi = image[y:y+h, x:x+w]
                cv2.imwrite( str(w*h) + ".png" , roi)
                f.write(text_from_image_file( str(w*h) + ".png",'tha'))
        # temp = text_from_image_file( str(w*h) + ".png",'tha')
        # if(temp.find(strA1) > 0 or temp.find(strA2) > 0):
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                os.remove( str(w*h) + ".png")
    cv2.imshow('img' , image) 
    cv2.waitKey(0)
    Spell_checker(fname)
    
main(sys.argv[1:])