import face_catcher
import recoder
import LLM
import cv2 as cv
import threading
import numpy as np
import os

os.makedirs("whoareyou_cache", exist_ok=True)

def main():
    unknown_counter = 0
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Fail to start camera")
        exit(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame")
            exit(0)

        face_in_db = face_catcher.detect_face(frame)

        if (not face_in_db) and unknown_counter < 100:
            unknown_counter += 1
        elif (not face_in_db) and unknown_counter >= 100:
            print("Unknown_counter = 0")
            t1 = threading.Thread(target = face_catcher.crop_and_import, args = (frame,))
            t1.daemon = True
            t1.start()
            unknown_counter = 0

        cv.imshow("test", frame)

        if cv.waitKey(1) == ord("q"):
            cap.release()
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
