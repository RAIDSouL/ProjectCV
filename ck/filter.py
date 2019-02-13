import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import imutils
import subprocess
import os
from random import randint


# lower_blue = np.array([115,30,30])
# upper_blue = np.array([140,255,255])
# # for idx in range(1,22):
# image = cv2.imread("11"+".jpg")
# image = imutils.resize(im, height=500)
# hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

lower_blue = np.array([115,30,30])
upper_blue = np.array([140,255,255])

# image = cv2.imread("3"+".png")

image = cv2.imread("20190213_052347"+".jpg")
image = imutils.resize(image, height=500)
recim = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7 , 7),0)
kernel = np.ones((3,10),np.uint8)
edged = cv2.Canny(blurred, 50, 200, 255)
cv2.imshow("edge" , edged)
dilation = cv2.dilate(edged,kernel,iterations = 1)
cv2.imshow("dilation", dilation)
contourmask,contours,hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
datalists = []
a=0
b=20
for cnt in contours[0:] :
    x, y, w, h = cv2.boundingRect(cnt)
    if(w * h > 500) :
        roi = image[y:y+h, x:x+w]
        roi = imutils.resize(roi, height=100)
        cv2.rectangle(recim,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite( str(w*h) +".png" , roi)

        filterImg = cv2.imread(str(w * h) + ".png")
        cv2.imshow("test" , filterImg)
        cv2.waitKey(0)

        hsv = cv2.cvtColor(filterImg, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        if(np.sum(mask) > 1000) :
            CropImg = filterImg[0:100,85:500]
            cv2.imshow("conImg" , CropImg)
            cv2.imwrite( str(w*h) +".png" , CropImg)
            data = text_from_image_file( str( w * h ) + ".png",'tha')
            print(data)


cv2.imshow("sad",recim)
cv2.waitKey(0)

#         cropImg = filterImg.copy()
        

#         hsv = cv2.cvtColor(filterImg, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, lower_blue, upper_blue)
#         if(np.sum(mask) != 0) :
#             print(np.sum(mask))
#         if(np.sum(mask) > 1000) :
#             _,contours2,_ = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#             for cnt1 in contours2[1:] :
#                 x2, y2, w2, h2 = cv2.boundingRect(cnt1)
#                 if( w / h > 0.9 and w / h < 1.1) : 
#                     realImg = filterImg[0:100,65:65+100]
#             # realImg = filterImg[a:a+100,b:b+500]
#             # cv2.imwrite( str(w*h) +".png" , realImg)
#             cv2.imshow("er1",realImg)
#             data = text_from_image_file( str( w * h ) + ".png",'tha')
#             print(data)
#             cv2.waitKey(0)
#             datalists = datalists + data
#         # os.remove( str(w * h) + ".png")
        

# # cv2.imshow("kodsad" , mask)
# # print(datalists)        


# print(np.sum(mask))
# cv2.imshow("im",mask)
# cv2.waitKey(0)

# # upper mask (175-180)
# lower_red = np.array([175,110,110])
# upper_red = np.array([180,255,255])
# mask = cv2.inRange(img_hsv, lower_red, upper_red)
# # upper mask (170-180)
# lower_red_2 = np.array([0,120,120])
# upper_red_2 = np.array([5,255,255])
# mask2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)

# # join my masks
# mask = mask + mask2




# pic[np.where(mask==0)] = 0

