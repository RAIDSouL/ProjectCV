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
    
    isEatBefore = False
    isPeriod = bool(False)
    isEatBreakfast = False
    isEatLunch = False
    isEatDinner = False
    isEatBedTime = False
    isRoutine = False
    periodHour = 0

    line = f.readline()
    while line:
        if(line.find(strB1) > 0):
            # print ('ก่อนอาหาร')
            isEatBefore = True
            if(line.find(str2) >0):
                isEatBreakfast = True
                # print('เช้า')
            if(line.find(str3) >0):
                # print('กลางวัน')
                isEatLunch = True
            if(line.find(str4) >0):
                # print('เย็น')
                isEatDinner = True
        if(line.find(strA1) > 0 or line.find(strA2) > 0):
            # print ('หลังอาหาร')
            isEatBefore = False
            if(line.find(str2) >0):
                # print('เช้า')
                isEatBreakfast = True
            if(line.find(str3) >0):
                # print('กลางวัน')
                isEatLunch = True
            if(line.find(str4) >0):
                # print('เย็น')
                isEatDinner = True
        line = f.readline()
        JSON_Creator(isPeriod, isEatBefore, isEatBreakfast, isEatLunch, isEatDinner, isEatBedTime, isRoutine, periodHour)

def JSON_Creator(_isPeriod, _isEatBefore,_isEatBreakfast, _isEatLunch, _isEatDinner, _isEatBedTime, _isRoutine, _periodHour) :
    temp = open("temp.txt", "w")
    temp.write("{")
    if _isPeriod == False :
        temp.write ("\"" + "isPeriod" + "\"" + " : " + "false ,")

        # "Data" : {
        temp.write ("\"" + "Data" + "\"" + " : " + "{")

        if _isEatBefore :
            temp.write ("\"" + "isEatBefore" + "\""+ " : " + "true ,")
        else : temp.write ("\"" + "isEatBefore" + "\""+ " : " + "false ,")
        if _isEatBreakfast :
            temp.write ("\"" + "isEatBreakfast" + "\""+ " : " + "true ,")
        else : temp.write ("\"" + "isEatBreakfast" + "\""+ " : " + "false ,")
        if _isEatLunch :
            temp.write ("\"" + "isEatLunch" + "\""+ " : " + "true ,")
        else : temp.write ("\"" + "isEatLunch" + "\""+ " : " + "fasle ,")
        if _isEatDinner :
            temp.write ("\"" + "isEatDinner" + "\""+ " : " + "true ,")
        else : temp.write ("\"" + "isEatDinner" + "\""+ " : " + "fasle ,")
        if _isEatBedTime :
            temp.write ("\"" + "isEatBedTIme" + "\""+ " : " + "true")
        else : temp.write ("\"" + "isEatBedTIme" + "\""+ " : " + "false")

        temp.write("}")
        
    if _isPeriod == True : 
        temp.write("\"" + "isPeriod" + "\"" + " : " + "ture ,")
        
        # "Data" : {
        temp.write ("\"" + "Data" + "\"" + " : " + "{")
        if isRoutine :
            temp.write("\"" + "isRoutine" + "\"" + " : " + "true,")
        else : temp.write("\"" + "isRoutine" + "\"" + " : " + "false,")
        temp.write("\"" + "PeriodHour" + "\"" + " : " + str(periodHour))
        temp.write("}")

    temp.write("}")

def main(argv) :
    image = cv2.imread(argv[0]) 
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = More_Gray(3,gray) #make picture more clear
    blurred = cv2.GaussianBlur(gray, (5 , 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    kernel = np.ones((2,8),np.uint8)
    dilation = cv2.dilate(edged,kernel,iterations = 1)
    contourmask,contours,hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    fname = argv[0].split(".")[0]
     
    with open(fname+".txt","w") as f:
        for cnt in contours[1:] :
            x, y, w, h = cv2.boundingRect(cnt)
            if (h / w < 0.7 ) :
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                roi = image[y:y+h, x:x+w]
                cv2.imwrite( str(w*h) + ".png" , roi)
                f.write(text_from_image_file( str(w*h) + ".png",'tha'))
                os.remove( str(w*h) + ".png")
    Spell_checker(fname)
    F_temp = open("temp.txt", "r")
    print(F_temp.readline())
    # os.remove("OutputImg.txt")
    # os.remove("temp.txt")

main(sys.argv[1:])