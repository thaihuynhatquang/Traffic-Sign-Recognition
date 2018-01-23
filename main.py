import cv2
import numpy as np

keyData = []
descriptorData = []

# Initiate SIFT/SURF detector
detector = cv2.xfeatures2d.SIFT_create()
SCALE = 1

# paths
TRAFFIC_SIGN_PATH = 'traffic_sign\\'
OUTPUT_PATH = 'Output\\'
INPUT_PATH = 'Input\\'
INPUT_VIDEO = 'test0.avi'

def readImgData(path):
    for i in range (0, 10):
        temp = cv2.imread(path + str(i+1) + '.jpg', SCALE)
        tempKey, tempDescriptor = detector.detectAndCompute(temp, None)
        keyData.append(tempKey)
        descriptorData.append(tempDescriptor)

def compareImg(obj):
    keyObj, descriptorObj = detector.detectAndCompute(obj,None)
    maxMatches = 0
    idx = -1 # idx = -1 means does not found matched image

    for i in range(0, 10):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptorObj, descriptorData[i], k = 2)

        # Apply ratio test
        good = []
        countMatches = 0
        for m,n  in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                countMatches += 1
        if countMatches > maxMatches:
            maxMatches = countMatches
            idx = i
    if idx < 7:
        return idx+1
    return 8

## main ##
readImgData(TRAFFIC_SIGN_PATH)
vidcap = cv2.VideoCapture(INPUT_PATH + INPUT_VIDEO)
success, frame = vidcap.read()
if vidcap.isOpened() == False:
    print ("Error open video")

# video writer for the result
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoOut = cv2.VideoWriter(OUTPUT_PATH + 'output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

rectangles = []
count_frame = 0

while success:
    count_frame = count_frame + 1
    success, frame = vidcap.read(SCALE)
    if success == False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskRed = cv2.inRange(hsv, np.array([100, 170, 50]), np.array([140, 255, 255]))
    maskBlue = cv2.inRange(hsv, np.array([143,36,50]), np.array([170,255,255]))
    mask = cv2.bitwise_or(maskRed, maskBlue)
    # mask = cv2.Canny(mask, 75,200)
    # mask = cv2.GaussianBlur(mask, (5, 5), 2, 2)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,element,iterations = 1)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    foundSign = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.05*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        if (len(approx) > 8) & (len(approx) < 15) & (area > 30) & (rect[2] < 300) & (rect[3] < 300) & (rect[2] >= 20) or (rect[3] >= 20):
            x,y,w,h = rect
            subImg = frame[y:y+h,x:x+w]
            id = compareImg(subImg)
            if id == 0:
                continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, str(id), (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255,255), 3, cv2.LINE_AA)
            temp = [count_frame, id, x, y, x+w, y+h]
            rectangles.append(temp)
            foundSign = 1
        if foundSign: # if found a traffic sign in a frame
            break

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    videoOut.write(frame)

    k = cv2.waitKey(1)
    if k in [27, ord('Q'), ord('q')]: # exit on ESC
        break
vidcap.release()
videoOut.release()
cv2.destroyAllWindows()

# write result to Output.txt
with open(OUTPUT_PATH + 'Output.txt', 'w') as file:
    file.write(str(len(rectangles)) + '\n')
    for temp in rectangles:
        strArr = [str(a) for a in temp]
        file.write(" ".join(strArr))
        file.write('\n')
