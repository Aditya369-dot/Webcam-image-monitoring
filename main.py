import cv2
import time

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video capture.")
    exit()

time.sleep(1)  # Allow the camera to warm up

first_frame = None
try:
    while True:
        check, frame = video.read()
        if not check:
            break  # If frame is not read correctly, exit the loop

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if first_frame is None:
            first_frame = gray_frame_gau
            continue

        delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
        thresh_frame = cv2.threshold(delta_frame, 35, 255, cv2.THRESH_BINARY)[1]
        dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cv2.imshow("Motion Detection", dil_frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break
finally:
    video.release()
    cv2.destroyAllWindows()



