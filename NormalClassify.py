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

datalists = []

pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")

def text_from_image_file(image_name,lang):
    output_name = "OutputImg"
    return_code = subprocess.call(['tesseract',image_name,output_name,'-l',lang,'-c','preserve_interword_spaces=1 --tessdata-dir ./tessdata_best/'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = open(output_name+'.txt','r',encoding='utf-8')
    temp = d.read()
    char_to_remove = re.findall(pattern, temp)
    list_with_char_removed = [char for char in temp if not char in char_to_remove]
    result_string = ''.join(list_with_char_removed)
    sad = result_string.replace(" ", "")
    return sad
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
    for r in range(rows):
        print(dist[r])
    
 
    return dist[row][col]

def remove_whitespace(text):
    text = text.replace(" ","")
    return text

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
    cvt_to_JSON(isPeriod, isEatBefore, isEatBreakfast, isEatLunch, isEatDinner, isEatBedTime, isRoutine, periodHour)

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
    with open(fname+".txt","w") as f:
        for cnt in contours[1:] :
            x, y, w, h = cv2.boundingRect(cnt)
            if (h * w > 500) :
                # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                roi = image[y:y+h, x:x+w]
                cv2.imwrite( str(w*h) + ".png" , roi)
                # f.write(text_from_image_file( str(w*h) + ".png",'tha'))
                datalists = datalists + text_from_image_file( str(w*h) + ".png",'tha')
                # line = f.readline()
                # if(line.find(strA1) > 0 or line.find(strA2) > 0 ) :
                    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                os.remove( str(w*h) + ".png")
    # Spell_checker(fname)
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


    # os.remove("OutputImg.txt")
    # os.remove("temp.txt")

main(sys.argv[1:])