import cv2 as cv

cap = cv. VideoCapture(1)

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)

while True:
    _, frame = cap.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    height, width, _ = frame.shape

    cx = int(width/2)
    cy = int(height/2)

    pixel_center = frame[cy,cx]
    hue_value = pixel_center[0]

    color = "Undefined"

    if hue_value < 3:
        color = "RED"
    elif hue_value < 22:
        color = "ORANGE"
    elif hue_value < 33:
        color = "YELLOW"
    elif hue_value < 78:
        color = "GREEN"
    elif hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "RED"

    pixel_center_bgr = frame[cy, cx]
    cv.putText(frame, color, (10,50), 0, 1, (255,0,0), 2)

    print(pixel_center)
    cv.circle(frame, (cx,cy), 5, (255,0,0), 3)

    cv.imshow("Frame", frame)
    key = cv.waitKey(1)
    if key==27:
        break


cap.release()
cv.destroyAllWindows