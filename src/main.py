import face_catcher
import cv2 as cv
import os
import config
import numpy as np


def main() -> None:
    """
    The main function for the face recognition application.
    This function initializes the camera and processes video frames in a loop. 
    It operates in three modes: recognition, enrollment, and update. The mode 
    determines the behavior of the application based on the state of detected 
    faces. The application switches modes dynamically based on the detection 
    of unknown faces or other conditions.
    Modes:
        - recognition: Detects and identifies faces in the video feed. If an 
          unknown face is detected repeatedly, the mode switches to either 
          enrollment or update.
        - enrollment: Enrolls a new face into the system by cropping and 
          importing the detected face.
        - update: Updates the database with new information for an existing 
          face.
    The function also handles camera initialization, frame capturing, and 
    displays the video feed in a window. The application exits when the user 
    presses the 'q' or 'Q' key.
    Raises:
        SystemExit: If the camera fails to start or if frames cannot be 
        retrieved from the video feed.
    """

    # recognition mode / enrollment mode / update mode
    config.MODE = "recognition"

    # Initialize the face_catcher object
    unknown_counter = 0
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Fail to start camera")
        exit(0)

    # Create the info window
    cv.namedWindow("User Info")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame")
            exit(0)

        state, face_name, id, face_hobby= face_catcher.detect_face(frame)

        if config.MODE == "recognition":

            if (state == 2 or face_name == "unknown"):
                unknown_counter += 1
                if unknown_counter >= config.UNKNOWN_COUNTER_THRESHOLD:
                    if state == 2:
                        print("Detected unknown face repeatedly, switching to enrollment mode...")
                        config.MODE = "enrollment"
                    elif face_name == "unknown":
                        print("Detected unknown face, switching to enrollment mode...")
                        config.MODE = "update"
                    unknown_counter = 0
            else:
                unknown_counter = 0
        
        elif config.MODE == "enrollment" and not config.IS_ENROLLMENT:
            print("Enrollment mode: enrolling face...")
            config.IS_ENROLLMENT = True
            face_catcher.crop_and_import(frame)
            # mode = "recognition"
        
        elif config.MODE == "update" and not config.IS_UPDATE:
            print("Update mode: updating face...")
            config.IS_UPDATE = True
            face_catcher.updating_db(id)
            # mode = "recognition"

        # Create a blank image for the floating info window
        info_img = np.zeros((400, 800, 3), dtype=np.uint8)  # 200x400 pixels black image

        # Overlay the detected user's name and hobby
        cv.putText(info_img, f"Name: {face_name}", (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 375)
        cv.putText(info_img, f"Hobby: {face_hobby}", (10, 150),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 375)
        cv.putText(info_img, f"Mode: {config.MODE}", (10, 350),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 375)
        

        # Show the main video feed and the info window
        cv.imshow("test", frame)
        cv.imshow("User Info", info_img)


        if cv.waitKey(1) == ord("q") or cv.waitKey(1) == ord("Q"):
            cap.release()
            cv.destroyAllWindows()
            config.clean_up_audio()
            break


if __name__ == "__main__":
    main()
