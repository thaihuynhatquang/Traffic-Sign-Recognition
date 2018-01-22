import cv2
import numpy as np

imgData = []
keyData = []
descriptorData = []
detector = cv2.xfeatures2d.SURF_create(500)

def readImgData(path):
    for i in range (0,10):
        temp = cv2.imread(str(path) + str(i+1) + '.jpg')
        imgData.append(temp)
        tempKey, tempDescriptor = detector.detectAndCompute(temp, None)
        keyData.append(tempKey)
        descriptorData.append(tempDescriptor)

readImgData("traffic_sign\\")

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

def compareImg(obj):
    keyObj, descriptorObj = detector.detectAndCompute(obj,None)
    maxFeature = 0
    idx = -1
    for i in range(0, 10):
        matches = flann.knnMatch(descriptorObj, descriptorData[i], k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        countMatches = 0
        for j,(m,n) in enumerate(matches):
            if m.distance < 0.6*n.distance:
                matchesMask[j]=[1,0]
                countMatches += 1
        if countMatches > maxFeature:
            maxFeature = countMatches
            idx = i + 1
    if idx > 7:
        return 8
    return idx

img = cv2.imread("test.png",1)
temp = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 2)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
maskRed = cv2.inRange(hsv, np.array([100, 170, 50]), np.array([140, 255, 255]))
maskBlue = cv2.inRange(hsv, np.array([143,36,50]), np.array([170,255,255]))
mask = cv2.bitwise_or(maskRed, maskBlue)
# mask = cv2.GaussianBlur(mask, (9, 9), 2, 2)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
mask = cv2.dilate(mask,element,iterations = 1)
_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
listSubImg = []
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.05*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    rect = cv2.boundingRect(contour)
    if (len(approx) > 8) & (len(approx) < 15) & (area > 30) & (rect[2] < 300) & (rect[3] < 300) & (rect[2] >= 20) or (rect[3] >= 20):
        x,y,w,h = rect

        subImg = temp[y:y+h,x:x+w,]
        id = compareImg(subImg)
        id = compareImg(subImg)
        if id == -1:
            continue
        cv2.imwrite(str(contours.index(contour))+".png",subImg)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # temp = [count_frame, x, y]
        # rectangles.append(temp)

        print (id)
# cv2.imshow("maskRed_lower", maskRed_lower)
# cv2.imshow("maskRed_higher", maskRed_higher)
# cv2.imshow("maskBlue", maskBlue)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
