 roi = image[y:y+h, x:x+w]
                cv2.imwrite( str(w*h) +".png" , roi)
                filter_img = cv2.imread(str(w*h) + ".png")
                filter_img = imutils.resize(filter_img, height=300)
                hsv = cv2.cvtColor(filter_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                filter_img = imutils.resize(mask, height=10)
                _,contours,_ = cv2.findContours(filter_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours) == 1)
                    datalists.append(text_from_image_file( str(w*h) + ".png",'tha'))
                os.remove( str(w*h) + ".png")