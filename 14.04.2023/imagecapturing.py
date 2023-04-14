import cv2

cap = cv2.VideoCapture(1)

num = 0

while cap.isOpened():

    succes1, img = cap.read()
    
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        #cv2.imwrite('images/Data-14042023/Workimages' + str(num) + '.png', img)
        cv2.imwrite('chess'+str(num)+'.png', img)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)

cap.release()

cv2.destroyAllWindows()

'''
import cv2

# Define video capture device
cap = cv2.VideoCapture(1)

# Set capture device properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Loop over frames from video capture device
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for user input to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()
'''

