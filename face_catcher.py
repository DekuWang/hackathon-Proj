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
FACE_DETECT = cv.CascadeClassifier("pretrained_model\haarcascade_frontalface_default.xml")

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

def crop_face(img: np.ndarray):
    """
    Crop the face from the image and return the cropped image and gray image
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = FACE_DETECT.detectMultiScale(
        gray, 
        scaleFactor = 1.1, 
        minNeighbors = 5, 
        minSize=[100, 100]
        )
    
    if faces is None or len(faces) == 0:
        return None, None
    
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    img_y, img_x = img.shape[:2]
    end_x = img_x if x+w > img_x else x+w
    end_y = img_y if y+h > img_y else y+h
    crop_img = img[y:end_y, x:end_x]
    gray_face = gray[y:end_y, x:end_x]
    return crop_img, gray_face

def detect_face(img: np.ndarray):
    """
    return 0 = no face detected,
    return 1 = face detected, 
    return 2 = face detected, but not in DB
    """
    conn, c = connect_db()
    recognizer = cv.face.LBPHFaceRecognizer.create()
    trainer_path = "trainer/trainer.yml"

    if not os.path.exists(trainer_path):
        crop_and_import(img)

    try:
        recognizer.read("trainer/trainer.yml")
    except cv.error as e:
        print("[ERROR] Unable to read model:", e)
        conn.close()
        return 0

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = FACE_DETECT.detectMultiScale(
        gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=[100, 100]
        )

    recognized = 0
    if faces is None or len(faces) == 0:
        conn.close()
        return 0
    
    # get largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    cv.rectangle(img, (x,y), (x+w, y+h), color = (0, 0, 255), thickness = 2)

    try:
        ids, confidence = recognizer.predict(gray[y:y+h, x:x+w])
    except Exception as e:
        print("[ERROR] Unable to predict face:", e)
        conn.close()
        return 0

    if confidence < 60:
        with db_lock:
            c.execute(f"SELECT name, hobby FROM faces WHERE id = {ids}")
            record = c.fetchone()
        face_name = record[0] if record else "inDB, no Record"

        cv.putText(img, face_name, (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
        cv.putText(img, str(confidence), (x + 10, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
        recognized = 1
    else:
        cv.putText(img, "Unknown", (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
        recognized = 2

    conn.close()
    return recognized


def import_face(gray_img: np.ndarray, id: int):
    """
    take gray image as the parameter
    And and int id for id
    """
    os.makedirs("trainer", exist_ok = True)
    trainer_path = "trainer/trainer.yml"
    recognier = cv.face.LBPHFaceRecognizer.create()

    if not os.path.exists(trainer_path):
        recognier.train([gray_img], np.array([id]))
    else:
        try:
            print("yml exist")
            recognier.read(trainer_path)
            recognier.update([gray_img], np.array([id]))
        except Exception as e:
            print(f"[ERROR] Failed to update model: {e}")
            return
    
    try:
        recognier.write(trainer_path)
    except Exception as e:
        print(f"[ERROR] Failed to update model: {e}")
        return

def crop_and_import(img: np.ndarray):
    face, gray_face = crop_face(img)
    print("     Face Cropped")
    
    if face is None or gray_face is None or face.size == 0 or gray_face.size == 0:
        print("         No Output")
        return
    
    conn, c = connect_db()
    with db_lock:
        c.execute(f"INSERT INTO faces (name, hobby) VALUES ('unknown', 'unknown')")
        conn.commit()
        new_id = c.lastrowid
    
    import_face(gray_img = gray_face, id = new_id)
    print("     Face Imported")

    conn.close()

    recoder_thread = threading.Thread(target = db_update_name, args = (new_id,))
    recoder_thread.daemon = True
    recoder_thread.start()

    print("     Face Added")
    return

def db_update_name(id: int):
    audio_file = os.path.join("whoareyou_cache", f"{id}.wav")
    recoder.record(audio_file)
    LLM_dict = None
    for attempt in range(5):
        try:
            LLM_reply = LLM.get_name_hobby(USERNAME, f"whoareyou_cache/{id}.wav")
            LLM_dict = json.loads(LLM_reply)
            break
        except Exception as e:
            print("Parse Error, retrying...")
            counter += 1
            continue
    
    if LLM_dict is None:
        print("Failed to parse LLM response")
        return

    # DB for other thread
    conn, c = connect_db()

    with db_lock:
        c.execute(f"UPDATE faces SET name = '{LLM_dict['speaker']}', hobby = '{LLM_dict['hobby']}' WHERE id = {id}")
        conn.commit()

        c.execute(f"SELECT name, hobby FROM faces WHERE id = {id}")
        face_record = c.fetchone()
    conn.close()

    if face_record is None:
        print("No record found")
        return
    else:
        print(f"     Face Name: {face_record[0]}")

if __name__ == "__main__":
    test_person_path = os.path.join("face_output", "0", "12042025_115719.jpg")
    if os.path.exists(test_person_path):
        test_person = cv.imread(test_person_path)
        crop_and_import(test_person)
    detect_face(test_img_people)
