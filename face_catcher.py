# Project Modules
import config
import LLM
import recoder

# Third-party Libraries
import cv2 as cv
import numpy as np
import json
import types

# Built-in Libraries
import os
import sqlite3
import threading


# Load Pretrained Model
FACE_DETECT = cv.CascadeClassifier(config.LBPH_MODEL_PATH)

# Create a lock for the database connection
db_lock = threading.Lock()

def connect_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Establishes a connection to the SQLite database and creates a table for storing face data if it does not exist.
    Returns:
        tuple[sqlite3.Connection, sqlite3.Cursor]: A tuple containing the database connection object and the cursor object.
    The table created has the following structure:
        - id: An auto-incrementing integer serving as the primary key.
        - name: A text field to store the name of the individual.
        - hobby: A field to store the hobby of the individual (data type not explicitly specified).
    """
    conn = sqlite3.connect(config.DB, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,                     
                hobby NOT NULL)''')
    conn.commit()
    return conn, c

def crop_face(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop the face from the given image.
    This function detects faces in the input image using a pre-trained face detection model 
    and crops the largest detected face. It also returns the corresponding grayscale version 
    of the cropped face.
    Args:
        img (np.ndarray): The input image in BGR format as a NumPy array.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The cropped image of the largest detected face in BGR format.
            - The cropped grayscale image of the largest detected face.
          If no face is detected, both elements of the tuple will be None.    
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

def detect_face(img: np.ndarray) -> tuple[int, str, int, str]:
    """
    Detects a face in the given image, identifies it using a pre-trained model, 
    and retrieves associated information from the database.
    Args:
        img (np.ndarray): The input image in which a face is to be detected.
    Returns:
        tuple[int, str, int, str]: A tuple containing:
            - An integer indicating the recognition status:
                0 - No face detected.
                1 - Face recognized.
                2 - Face detected but not recognized.
            - A string representing the name of the recognized person (or None if not recognized).
            - An integer representing the ID of the recognized person (or None if not recognized).
            - A string representing the hobby of the recognized person (or "Unknown" if not recognized).
    Notes:
        - The function uses a pre-trained LBPH face recognizer model stored in "trainer/trainer.yml".
        - If the model file does not exist, it initializes the model using `model_initialization()`.
        - The function retrieves the largest detected face in the image for recognition.
        - If the confidence score of the prediction is below 70, the face is considered recognized.
        - The function interacts with a database to fetch the name and hobby of the recognized person.
        - If no face is detected or the face is not recognized, appropriate default values are returned.
    """
    conn, c = connect_db()
    recognizer = cv.face.LBPHFaceRecognizer.create(radius = 1, neighbors = 8, grid_x = 8, grid_y = 8)

    if not os.path.exists(config.TRAINER):
        model_initialization()

    try:
        recognizer.read(config.TRAINER)
    except cv.error as e:
        print("[ERROR] Unable to read model:", e)
        conn.close()
        return 0, None, None, "Unknown"

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
        return 0, None, None, "Unknown"
    
    # get largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    cv.rectangle(img, (x,y), (x+w, y+h), color = (0, 0, 255), thickness = 2)

    try:
        ids, confidence = recognizer.predict(gray[y:y+h, x:x+w])
    except Exception as e:
        print("[ERROR] Unable to predict face:", e)
        conn.close()
        return 0, None, None, "Unknown"

    if confidence < 70:
        with db_lock:
            c.execute(f"SELECT name, hobby FROM faces WHERE id = {ids}")
            record = c.fetchone()
        face_name = record[0] if record else "inDB, no Record"
        hobby = record[1] if record else "inDB, no Record"

        cv.putText(img, face_name, (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
        cv.putText(img, str(confidence), (x + 10, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
        recognized = 1, face_name, ids, hobby
    else:
        cv.putText(img, "Unknown", (x + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 1)
        recognized = 2, None, None, "Unknown"

    conn.close()
    return recognized


def import_face(gray_img: np.ndarray, id: int) -> None:
    """
    Imports a grayscale face image into the face recognition trainer.
    This function creates or updates a face recognition model using the 
    Local Binary Patterns Histograms (LBPH) algorithm. It saves the trained 
    model to a file named 'trainer.yml' in the 'trainer' directory.
    Args:
        gray_img (np.ndarray): A grayscale image of the face to be added to the trainer.
        id (int): The unique identifier associated with the face.
    Returns:
        None
    Notes:
        - If the 'trainer.yml' file does not exist, a new model is created and trained.
        - If the 'trainer.yml' file exists, the model is updated with the new face data.
        - Any errors during the training or saving process are logged to the console.
    """
    recognier = cv.face.LBPHFaceRecognizer.create(radius = 1, neighbors = 8, grid_x = 8, grid_y = 8)

    if not os.path.exists(config.TRAINER):
        recognier.train([gray_img], np.array([id]))
    else:
        try:
            print("yml exist")
            recognier.read(config.TRAINER)
            recognier.update([gray_img], np.array([id]))
        except Exception as e:
            print(f"[ERROR] Failed to update model: {e}")
            return
    
    try:
        recognier.write(config.TRAINER)
    except Exception as e:
        print(f"[ERROR] Failed to update model: {e}")
        return

def crop_and_import(img: np.ndarray) -> None:
    """
    Processes an image to detect and crop a face, imports the face data into a database, 
    and starts a background thread to update the name associated with the face.
    Args:
        img (np.ndarray): The input image as a NumPy array.
    Returns:
        None
    Workflow:
        1. Detects and crops the face from the input image.
        2. If no face is detected, the function exits early.
        3. Connects to the database and inserts a new record with default values ('unknown').
        4. Imports the cropped grayscale face into the database associated with the new record.
        5. Starts a background thread to update the name associated with the face.
        6. Closes the database connection and logs the process steps.
    Notes:
        - The function assumes the existence of helper functions `crop_face`, `connect_db`, 
          `import_face`, and `db_update_name`.
        - Threading is used to handle the name update process asynchronously.
        - Proper database locking is implemented to ensure thread safety.
    """
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

def db_update_name(id: int) -> None:
    """
    Updates the name and hobby of a person in the database based on their ID.
    This function records an audio file for the given ID, sends it to an LLM 
    (Large Language Model) for processing to extract the speaker's name and hobby, 
    and updates the database with the retrieved information. If the LLM response 
    cannot be parsed after multiple attempts, the function exits without updating 
    the database.
    Args:
        id (int): The unique identifier of the person whose information is to be updated.
    Raises:
        Exception: If there are issues with parsing the LLM response or database operations.
    Side Effects:
        - Records an audio file and saves it in the "data/audio" directory.
        - Updates the database with the name and hobby of the person.
        - Prints messages to the console for debugging and status updates.
        - Modifies global configuration flags (`config.MODE`, `config.IS_ENROLLMENT`, 
          and `config.IS_UPDATE`).
    Notes:
        - The function retries up to 5 times to parse the LLM response.
        - A database lock (`db_lock`) is used to ensure thread-safe operations.
        - If the database update is successful, the updated record is fetched and printed.
    """
    audio_file = os.path.join(config.AUDIO_PATH, f"{id}.wav")
    recoder.record(audio_file)
    LLM_dict = None
    for attempt in range(5):
        try:
            LLM_reply = LLM.get_name_hobby(config.USERNAME, audio_file)
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
        if LLM_dict['hobby'] == "unknown":
            c.execute("UPDATE faces SET name = ? WHERE id = ?",
                      (LLM_dict['speaker'], id))
        else:
            c.execute("UPDATE faces SET name = ?, hobby = ? WHERE id = ?", 
            (LLM_dict['speaker'], LLM_dict['hobby'], id))
        conn.commit()

        c.execute(f"SELECT name, hobby FROM faces WHERE id = {id}")
        face_record = c.fetchone()
    conn.close()

    if face_record is None:
        print("No record found")
        return
    else:
        print(f"     Face Name: {face_record[0]}")

    config.MODE = "recognition"

    if config.IS_ENROLLMENT:
        config.IS_ENROLLMENT = False
    elif config.IS_UPDATE:
        config.IS_UPDATE = False
    
def updating_db(id) -> None:
    """
    Starts a daemon thread to update the database with the given ID.
    Args:
        id (int): The identifier to be used for updating the database.
    Returns:
        None
    """
    t1 = threading.Thread(target = db_update_name, args = (id,))
    t1.daemon = True
    t1.start()
    
def model_initialization() -> None:
    """
    Initializes the face recognition model by processing a default image and 
    storing the extracted face data into a database.
    Steps:
    1. Reads a default image file named "default.jpg".
    2. Extracts the face and its grayscale version using the `crop_face` function.
    3. If no face is detected or the extracted data is invalid, prints a message and exits.
    4. Connects to the database and inserts a new record with a predefined name and hobby.
    5. Imports the grayscale face data into the database associated with the new record.
    6. Closes the database connection.
    Note:
    - The function assumes the existence of the "default.jpg" file in the working directory.
    - The database schema must include a `faces` table with `name` and `hobby` fields.
    - Thread safety is ensured using a `db_lock` during database operations.
    Raises:
    - Any exceptions related to file I/O, database connection, or SQL execution are not handled explicitly.
    """
    print("Model Initialization")
    default_img = cv.imread(config.DEFAULT_IMG)
    face, gray_face = crop_face(default_img)
    print("     Face Cropped")
    
    if face is None or gray_face is None or face.size == 0 or gray_face.size == 0:
        print("         No Output")
        return
    
    conn, c = connect_db()
    with db_lock:
        c.execute(f"INSERT INTO faces (name, hobby) VALUES ('Kobe Bryant', 'Basketball')")
        conn.commit()
        new_id = c.lastrowid
    
    import_face(gray_img = gray_face, id = new_id)
    conn.close()

# Test Cases
# if __name__ == "__main__":
#     test_person_path = os.path.join("face_output", "0", "12042025_115719.jpg")
#     if os.path.exists(test_person_path):
#         test_person = cv.imread(test_person_path)
#         crop_and_import(test_person)
#     default_img = cv.imread("default.jpg")
#     detect_face(default_img)
