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
    image = imutils.resize(image, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = More_Gray(3,gray) #make picture more clear
    blurred = cv2.GaussianBlur(gray, (5 , 5), 0)
    # cv2.imshow('blurred' , blurred) 
    edged = cv2.Canny(blurred, 50, 200, 255)
    # cv2.imshow('edged' , edged) 
    kernel2 = np.ones((14,3),np.uint8)
    erode = cv2.dilate(edged,kernel2,iterations = 1)
    cv2.imshow('erode' , erode)
    kernel = np.ones((3,5),np.uint8)
    dilation = cv2.dilate(erode,kernel,iterations = 1)
    cv2.imshow('dilation' , dilation) 

    kernel = np.ones((3,12),np.uint8)
    erode = cv2.erode(dilation,kernel)
    # cv2.imshow('erode' , erode) 

    kernel = np.ones((12,3),np.uint8)
    erode = cv2.erode(dilation,kernel)
    # cv2.imshow('erode' , erode) 

    kernel = np.ones((14,3),np.uint8)
    erode = cv2.erode(dilation,kernel)
    cv2.imshow('erode' , erode) 
    
    contourmask,contours,hierarchy = cv2.findContours(erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    fname = argv[0].split(".")[0]
    temp = image
    
    # with open(fname+".txt","w") as f:
    for idx,cnt in enumerate(contours[1:]) :
        if idx > len(contours[1:]) :
            break
        x, y, w, h = cv2.boundingRect(cnt)
        # if (h / w < 0.7 and h * w > 500) :
        # if h > 30 and h < 50 :
        if h * w > 900 and h * w < 15000:
            # cv2.rectangle(temp,(x,y),(x+w,y+h),(0,0,255),2)
            roi = image[y:y+h, x:x+w]
            cv2.imwrite( ".//data//" +str(w*h) + ".png" , roi)
                # f.write(text_from_image_file( str(w*h) + ".png",'tha'))
        # temp = text_from_image_file( str(w*h) + ".png",'tha')
        # if(temp.find(strA1) > 0 or temp.find(strA2) > 0):
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                # os.remove( str(w*h) + ".png")
    cv2.imshow('img' , image) 
    cv2.waitKey(0)
    # Spell_checker(fname)
    
main(sys.argv[1:])