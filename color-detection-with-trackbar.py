import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("U - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 0, 255, nothing)

while True:
    frame = cv2.imread("039569.jpg")
    ret, frame2 = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    #frame = cv2.resize(frame, (800,450))


    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_blue = np.array([l_h , l_s , l_v])
    upper_blue = np.array([u_h , u_s , u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv2, lower_blue, upper_blue)

    #lower_not_red = np.array([20, 0, 0])
    #upper_not_red = np.array([160, 255, 255])

    cv2.imshow("mask", mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    mask = cv2.bitwise_and(frame, mask)

    mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

    mask2 = cv2.bitwise_and(frame2, mask2)

    cv2.imshow("video", mask2)
    #cv2.imshow("color", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
