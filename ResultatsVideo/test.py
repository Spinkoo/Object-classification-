

import cv2

i=23

while(True):
	try:
		cv2.imshow("",cv2.imread(str(i)+".png",1))
		cv2.waitKey(200)
		i+=1
	except:
		exit()