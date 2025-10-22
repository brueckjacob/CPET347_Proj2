import cv2
import numpy as np

def main(camera_index = 0):

    # Haar detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Failed to load Haar cascade")
        return
    
    # HOG detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Finds and opens webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame detected")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Gets data for Haar detection
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Creates two separate frames
            haar_frame = frame.copy()
            hog_frame = frame.copy()

            # Builds square based on haar data
            for(x, y, w, h) in faces:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(haar_frame, top_left, bottom_right, (255, 0, 0), thickness = 2)

            # Gets data for HOG detection
            rects, weights = hog.detectMultiScale(
                hog_frame,
                winStride=(8,8),
                padding=(8,8),
                scale=1.05
            )

            # Builds rectangle based on HOG data
            for (x, y, w, h) in rects:
                top_left2 = (x, y)
                bottom_right2 = (x + w, y + h)
                cv2.rectangle(hog_frame, top_left2, bottom_right2, (0, 255, 0), thickness = 2)

            # Displays both detections
            cv2.imshow("Viola-Jones Face Detection", haar_frame)
            cv2.imshow("HOG Detection", hog_frame)

            # Waits for 'q' input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()