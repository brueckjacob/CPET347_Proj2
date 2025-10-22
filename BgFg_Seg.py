import cv2
import numpy as np

def main():
    # Open web cam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    # Background subtractor using Gaussian Mixture Model
    backSub = cv2.createBackgroundSubtractorMOG2(
        history = 500,          # Number of frames
        varThreshold = 16,      # Threshold on square Mahalanobis distance
        detectShadows = True    # Detect and mark shadows
    )

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to find frame")
            return
        
        # Resize for perfomance
        frame = cv2.resize(frame, (640, 480))

        # Apply bg subtractor
        fg_mask = backSub.apply(frame)

        # Improve mask quality
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Extract foreground
        foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # Display results
        cv2.imshow('Original', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        #cv2.imshow('Foreground Extracted', foreground)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()