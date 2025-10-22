import cv2
import numpy as np

def main():
    max_corners = 70
    quality = 0.4
    min_distance = 7
    lk_params = dict(
        winSize = (15, 15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    feature_threshold = 0.5

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Failed to load Haar cascade")
        return

    cap = cv2.VideoCapture(0)

    tracking = False
    bbox = None
    p0 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame detected")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not tracking:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                x, y, w, h = faces[0]
                bbox = np.array([x, y, w, h], dtype = np.float32)
                roi_gray = gray[y:y + h, x:x + w]

                p0 = cv2.goodFeaturesToTrack(
                    roi_gray,
                    maxCorners = max_corners,
                    qualityLevel = quality,
                    minDistance = min_distance
                )

                if p0 is not None:
                    p0 += np.array([x, y], dtype = np.float32)
                    tracking = True
                    # Store previous gray conversion
                    prev_gray = gray.copy()

        else:
            # Optical Flow Tracking
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            
            # Keep only good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # Bounding box
            if len(good_new) > 0:
                # Average Translation
                dx = np.mean(good_new[:, 0] - good_old[:, 0])
                dy = np.mean(good_new[:, 1] - good_old[:, 1])
                bbox[:2] += np.array([dx, dy], dtype = np.float32)
            
                # Draw bbox in red
                x, y, w, h = bbox.astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness = 2)

                # Draw feature points
                for pt in good_new:
                    cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

                # Re-detection logic
                if len(good_new) < feature_threshold * max_corners:
                    tracking = False
                    p0 = None

            else:
                # Lost all points
                tracking = False
                p0 = None

            p0 = good_new.reshape(-1, 1, 2)

        cv2.imshow("Face Tracking (Red)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()