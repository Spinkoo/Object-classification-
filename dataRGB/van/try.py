import cv2

sift = cv2.xfeatures2d.SIFT_create(5)

img= cv2.imread("boulevard_1_28.png",1)
keypoints, descriptors = sift.detectAndCompute(img, None)
print(descriptors[0])