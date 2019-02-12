import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import imutils
import subprocess
import os
from random import randint

pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")

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


image = cv2.imread("1700"+".png")
image = imutils.resize(image, height=100)
cv2.imwrite( "1700" +".png" , image)
cv2.imshow("sad",image)
data = text_from_image_file("1700"+".png",'tha')
cv2.waitKey(0)
print(data)
