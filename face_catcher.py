import cv2 as cv
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sqlite3
import recoder
import LLM
import threading

import json

from config import USERNAME

if not os.path.exists("face_output"):
    os.mkdir("face_output")

test_img_people = cv.imread("test_people.jpg")

# Load Pretrained Model
FACE_DETECT = cv.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

# Create a lock for the database connection
db_lock = threading.Lock()


def connect_db():
    """
    Conneect to DB, used by other threads
    """
    conn = sqlite3.connect("face.db", check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,                     
                hobby NOT NULL)''')
    conn.commit()
    return conn, c

def get_new_id(cursor):
    """
    Get new id for the face
    """
    cursor.execute("SELECT MAX(id) FROM faces")
    max_id = cursor.fetchone()[0]
    return max_id + 1 if max_id is not None else 0

def crop_face(img: np.ndarray):
    """
    Crop the face from the image and return the cropped image and gray image
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_y, img_x = gray.shape[:2]
    face = FACE_DETECT.detectMultiScale(gray, minSize=[100,100])
    if face is None or len(face) == 0:
        return None, None
    
    for x,y,w,h in face:
        end_x = img_x if x+w > img_x else x+w
        end_y = img_y if y+h > img_y else y+h

        crop_img = img[y:end_y, x:end_x]

    return crop_img, gray[y:end_y, x:end_x]

def detect_face(img: np.ndarray):

    conn, c = connect_db()
    recognizer = cv.face.LBPHFaceRecognizer.create()
    trainer_path = "trainer/trainer.yml"

    if not os.path.exists("trainer/trainer.yml"):
        crop_and_import(img)

    try:
        recognizer.read("trainer/trainer.yml")
    except cv.error as e:
        print("[ERROR] Unable to read model:", e)
        conn.close()
        return False

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = FACE_DETECT.detectMultiScale(
        gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=[100, 100]
        )

    recognized = False
    if faces is None or len(faces) == 0:
        conn.close()
        return False
    
    for x, y, w, h in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), color = (0, 0, 255), thickness = 2)

        try:
            ids, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        except Exception as e:
            print("[ERROR] Unable to predict face:", e)
            continue

        if confidence < 60:
            with db_lock:
                c.execute(f"SELECT name, hobby FROM faces WHERE id = {ids}")
                record = c.fetchone()
            face_name = record[0] if record else "inDB, no Record"

            if face_name == "unknown":
                with db_lock:
                    c.execute(f"UPDATE faces SET name = 'unknown' WHERE id = {ids}")
                    conn.commit()

            cv.putText(img, face_name, (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
            cv.putText(img, str(confidence), (x + 10, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
            recognized = True
        else:
            cv.putText
        
    conn.close()
    return recognized


    #         face_name = c.execute(f"SELECT name, hobby FROM faces WHERE id = {ids}").fetchone()
    #         face_name = "inDB, no Record" if not face_name else face_name[0]

    #         if face_name == "unknown":
    #             # Update name in DB
    #             c.execute(f"UPDATE faces SET name = 'unknown' WHERE id = {ids}")
    #             conn.commit()
    #         cv.putText(img, face_name, (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
    #         cv.putText(img, str(confidence), (x + 10, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
    #     else:
    #         cv.putText(img, "Unknown", (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
    #         return False
    
    # return True

def import_face(gray_img: np.ndarray, id: int):
    """
    take gray image as the parameter
    And and int id for id
    """
    os.makedirs("trainer", exist_ok = True)
    recognier = cv.face.LBPHFaceRecognizer.create()

    if not os.path.isfile(r"trainer/trainer.yml"):
        recognier.train([gray_img], np.array([id]))
    else:
        try:
            print("yml exist")
            recognier.read(r"trainer/trainer.yml")
            recognier.update([gray_img], np.array([id]))
        except Exception as e:
            print(f"[ERROR] Failed to update model: {e}")
            return
    
    try:
        recognier.write(r"trainer/trainer.yml")
    except Exception as e:
        print(f"[ERROR] Failed to update model: {e}")


def crop_and_import(img: np.ndarray):
    face, gray_face = crop_face(img)
    print("     Face Cropped")
    
    if face is None or gray_face is None or face.size == 0 or gray_face.size == 0:
        print("         No Output")
        return
    
    c, conn = connect_db()

    id = c.execute("SELECT MAX(id) FROM faces").fetchone()[0] + 1 if c.execute("SELECT MAX(id) FROM faces").fetchone()[0] is not None else 0
    import_face(gray_img = gray_face, id = id)
    print("     Face Imported")

    # Saving face to folder
    output_path = os.path.join("face_output", str(id))
    os.makedirs(output_path, exist_ok=True)
    cv.imwrite(f"{output_path}/{datetime.now().strftime(r'%d%m%Y_%H%M%S')}.jpg", face)

    # Saving id to DB
    c.execute(f"INSERT INTO faces (name, hobby) VALUES ('unknown', 'unknown')")
    conn.commit()

    recoder_thread = threading.Thread(target = db_update_name, args = (id,))
    recoder_thread.daemon = True
    recoder_thread.start()

    print("face_added!")
    return

def db_update_name(id: int):
    recoder.record(f"whoareyou_cache/{id}.wav")
    for counter in range(5):
        try:
            LLM_reply = LLM.get_name_hobby(USERNAME, f"whoareyou_cache/{id}.wav")
            LLM_dict = json.loads(LLM_reply)
            break
        except Exception as e:
            print("Parse Error, retrying...")
            counter += 1
            continue

    # DB for other thread
    c, conn = connect_db()
    c.execute(f"UPDATE faces SET name = '{LLM_dict['speaker']}', hobby = '{LLM_dict['hobby']}' WHERE id = {id}")
    conn.commit()

    face_name = c.execute(f"SELECT name, hobby FROM faces WHERE id = {id}").fetchone()
    print(str(face_name))
    conn.close()


if __name__ == "__main__":
    test_person = cv.imread(r"face_output\0\12042025_115719.jpg")
    crop_and_import(test_person)
    detect_face(test_img_people)
