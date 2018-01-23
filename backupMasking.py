def getMaskRed(img_hsv):
    # lower mask (0-10)
    lower_red = np.array([0,70,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,70,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    return mask0 + mask1
    # lower_red = np.array([0,100,100])
    # upper_red = np.array([20,255,255])
    # return cv2.inRange(img_hsv, lower_red, upper_red)

def getMaskBlue(img_hsv):
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    return cv2.inRange(img_hsv, lower_blue, upper_blue)

# change constrast
# cv2.convertScaleAbs(frame, frame, 1.25, 0)
