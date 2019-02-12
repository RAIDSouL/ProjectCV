import cv2
import sys
import re
import os
import json
import imutils
import numpy as np
import subprocess
from imutils import contours
from imutils.perspective import four_point_transform


 
strTime = ["เช้า","กลางวัน","เย็น","ก่อนอาหาร","หลังอาหาร","หลังอาหารเช้าทันที","หลังอาหารเช้า"]

datalists = []

pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")

def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
        
        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs
    
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost) # substitution
    # for r in range(rows):
    #     print(dist[r])
    
    return dist[row][col]


def tsplit(string, delimiters):
    """Behaves str.split but supports multiple delimiters."""
    
    delimiters = tuple(delimiters)
    stack = [string,]
    
    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i+j, _substring)
            
    return stack

def text_from_image_file(image_name,lang):
    output_name = "OutputImg"
    return_code = subprocess.call(['tesseract',image_name,output_name,'-l',lang,'-c','preserve_interword_spaces=1 --tessdata-dir ./tessdata_best/'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = open(output_name+'.txt','r',encoding='utf-8')
    str_read = d.read()
    # char_to_remove = temp.split()
    # char_to_remove = re.findall(pattern, temp)
    
    temp = tsplit(str_read,(',', '/', '-', '=',' '))
    ouput = []
    for idx in temp :
        char_to_remove = re.findall(pattern, idx)

        list_with_char_removed = [char for char in idx if not char in char_to_remove]

        
        if len(''.join(list_with_char_removed)) != 0 :
           ouput = ouput + [''.join(list_with_char_removed)]
    return ouput

def More_Gray(gamma,image) : #make picture more clearly
    gamma1 = gamma
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma1) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    return res

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
    image = imutils.resize(image, height=700)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = More_Gray(3,gray) #make picture more clear
    blurred = cv2.GaussianBlur(gray, (7 , 7), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    kernel = np.ones((3,8),np.uint8)
    dilation = cv2.dilate(edged,kernel,iterations = 1)
    contourmask,contours,hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    fname = argv[0].split(".")[0]
    datalists = []
    #filter
    lower_blue = np.array([115,50,50])
    upper_blue = np.array([130,255,255])
    with open(fname+".txt","w") as f:
        for cnt in contours[1:] :
            x, y, w, h = cv2.boundingRect(cnt)
            if(w * h > 500) :
                roi = image[y:y+h, x:x+w]
                cv2.imwrite( "temp//" + str(w*h) +".png" , roi)
                # filter algo
                filter_img = cv2.imread(str(w*h) + ".png")
                filter_img = imutils.resize(filter_img, height=300)
                hsv = cv2.cvtColor(filter_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                filter_img = imutils(mask, height=10)
                _,contours,_ = cv2.findContours(filter_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours) == 1) :
                    datalists.append(text_from_image_file( str(w*h) + ".png",'tha'))
                os.remove( str(w*h) + ".png")
    # cv2.show("image",image)

    isEatingBefore = False
    _isEatBreakfast = False
    _isEatLunch = False
    _isEatDinner = False
    _isEatBedTime =False 
    # print(datalists)
    for idx,data in enumerate(strTime) :
        for txt in datalists :
            if iterative_levenshtein(data,txt) <= 2 and idx == 0 :
                _isEatBreakfast = True
            if iterative_levenshtein(data,txt) <= 2 and idx == 1 :
                _isEatLunch = True
            if iterative_levenshtein(data,txt) <= 2 and idx == 2 :
                _isEatDinner = True
            if iterative_levenshtein(data,txt) <= 2 and idx == 3 :
                _isEatBedTime = True
            if iterative_levenshtein(data,txt) <= 2 and idx == 4 :
                isEatingBefore = True
            if iterative_levenshtein(data,txt) <= 2 and idx == 5 :
                isEatingBefore = False
            if iterative_levenshtein(data,txt) <= 2 and idx == 5 :
                isEatingBefore = False
                _isEatBreakfast = True
            if iterative_levenshtein(data,txt) <= 2 and idx == 6 :
                isEatingBefore = False
                _isEatBreakfast = True

    cvt_to_JSON(False, isEatingBefore,_isEatBreakfast, _isEatLunch, _isEatDinner, _isEatBedTime, False, "_periodHour")

main(sys.argv[1:])