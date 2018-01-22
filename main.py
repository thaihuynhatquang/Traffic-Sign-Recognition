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

vidcap = cv2.VideoCapture("test4.avi")
success, frame = vidcap.read()
if vidcap.isOpened() == False:
    print ("Error open video")

fps = int(vidcap.get(cv2.CAP_PROP_FPS))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

videoOut = cv2.VideoWriter("output4.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))
rectangles = []
count_frame = 0

while success:
    success, frame = vidcap.read()
    if success == False:
        break
    frame2 = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskRed = cv2.inRange(hsv, np.array([100, 170, 50]), np.array([140, 255, 255]))
    maskBlue = cv2.inRange(hsv, np.array([143,36,50]), np.array([170,255,255]))
    mask = cv2.bitwise_or(maskRed, maskBlue)
    # mask = cv2.GaussianBlur(mask, (5, 5), 2, 2)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,element,iterations = 1)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.05*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        if (len(approx) > 8) & (len(approx) < 15) & (area > 30) & (rect[2] < 300) & (rect[3] < 300) & (rect[2] >= 20) or (rect[3] >= 20):
            x,y,w,h = rect
            subImg = frame[y:y+h,x:x+w]
            id = compareImg(subImg)
            if id == -1:
                continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, str(id), (x+w+20, y+h+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255,255))
            temp = [id, count_frame, x, y, x+w, y+h]
            rectangles.append(temp)

    # cv2.imshow("frame", frame)
    videoOut.write(frame)
    # cv2.imshow("mask", mask)
    k = cv2.waitKey(1)
    if k in [27, ord('Q'), ord('q')]: # exit on ESC
        break
    count_frame = count_frame + 1
# print (np.array(rectangles))

with open('output4.txt', 'w') as file:
    file.write(str(len(rectangles)) + '\n')
    for temp in rectangles:
        strArr = [str(a) for a in temp]
        file.write(" ".join(strArr))
        file.write('\n')
vidcap.release()
videoOut.release()
cv2.destroyAllWindows()
