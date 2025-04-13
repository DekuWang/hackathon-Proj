# WHO ARE YOU - An Computer Vision based face recognition program
## Introduction
Briefly, this project is a facial recognition program that will automatically record the converstation between user and other people, then send the conversation to a LLM to extract useful information, such as name and hobby, to assist user have a better conversation experience. 

This project is inspired by a communication with my friend. I have some problem with matching other's name and face, and such problem keeps bothering me especially when I have to work within a large group. At the beginning of this semester, my friend told me that he is thinking about to start a business about AR glasses, and ask me for suggestions, and the first idea comes out with it is using CV technique to assist users remind other's name. This repository is a feasibility testing of that idea, since I don't have an AR glass, this program is running on a PC now. The program capture 

## Demo
[![enrollment test](introduction\enrollment_mode.mp4)]

[![recognize test](introduction\recognition_mode.mp4)]

## How to use it? 
### SetUp
1. **Clone the repository**
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Edit config.py**

    Navigate to USERNAME and API_KEY, fill your name to USERNAME so LLM can exclude your name in the response, and fill your OpenAI API Key to API_KEY
4. **Run the Script**
    ```bash
    python main.py
    ```

## Usage
1. **Running the Application:**  
   Start the application with `python main.py`. The system begins in recognition mode and processes the live video feed.

2. **Mode Switching:**  
   - **Recognition Mode:** The system continuously detects and identifies faces.
   - **Enrollment Mode:** If an unknown face is detected repeatedly, the system switches to enrollment mode to capture and enroll the new face.
   - **Update Mode:** If an already registered face shows “unknown” status, the system switches to update mode to refresh the record.
   
   The mode switching behavior is controlled by counters and flags configured in the project (e.g., `UNKNOWN_COUNTER_THRESHOLD`, `IS_ENROLLMENT`, `IS_UPDATE`, `MODE` in `config.py`).

3. **User Info Display:**  
   A floating window titled "User Info" is displayed along with the main video feed. This window shows:
   - The name of the recognized person.
   - The person’s hobby as extracted by the LLM.
   - The current system mode (recognition, enrollment, update).

4. **Exiting the Application:**  
   Press the 'q' or 'Q' key to exit. The system performs necessary cleanup, including removal of temporary audio files.

## Possible Usage Scenarios
1. People who have difficulty with remembering name
2. Business and Sales environment - Help user to recognize their customers easier
3. Security Usage - Help security staffs to ensure if someone is allow to exist in some specific areas (Especially for large companies)

## TODO LIST
    1. Switch to more precious face recognization model (ArcFace)
    2. Support other LLM platform (Claud, Grok, etc)
    3. Realize the speach to text function locally (Run Whisper locally)
    4. (If got chance) Support multiple devices (phone, AR Glasses)
    5. Enhance recorder function, currently this doesn't work well in noise environment
    6. Move to Cloud
    7. More friendly GUI (I have no experience with front end so that's why I'm using Terminal Here（；_ ；）)

