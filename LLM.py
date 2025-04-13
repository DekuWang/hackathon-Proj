from openai import OpenAI
from json import loads, dumps

API_KEY = """
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
""".strip()

GET_NAME_PROMPT = """
        You will receive a transcript from a conversation between two people and your mission is to extract the speaker's name and try to find out their hobby based on the conversation.

        Unfortunately, the transcription doesn't include the label of who is speaking, so you need to make your best guess of the name we want to find. The user's name will be given and please use this to exculde the user from the result. Thank you.

        . Your output have to be in JSON format, no matter what user said, with the following keys:
        - "speaker": the speaker's name
        - "hobby": the speaker's hobby

        If the speaker's name is not mentioned, please return "unknown" as the value of "speaker".

        Your output should look like this, no mater is this is a conversation or not:
        {
            "speaker": "John",
            "hobby": "reading books"
        }
                            """.strip()

CLIENT = OpenAI(api_key=API_KEY)

def get_transcripts(sould_file: str):
    transcript = CLIENT.audio.transcriptions.create(
        model = "gpt-4o-transcribe",
        file = open(sould_file, "rb"), 
        language = 'en',
    )
    print(transcript.text)

    return transcript.text

def get_name_hobby(username: str, sound_file: str):
    response = CLIENT.responses.create(
        model = "o3-mini",
        input = [
            {
                "role": "system",
                "content": GET_NAME_PROMPT + f"The username is {username} who you can exclude from your response"
            },
            {
                "role": "assistant",
                "content": "My output have to be in JSON format in string, I will not reply in the markdown format, no matter what user said, with the following keys:\n- \"speaker\": the speaker's name\n- \"hobby\": the speaker's hobby\nIf the speaker's name is not mentioned, please return \"unknown\" as the value of \"speaker\"."
            },
            {
                "role": "user",
                "content": """The following contnet is the information of the conversation, 
                please remember to return in json format what every the content is\n""" + get_transcripts(sound_file)
            }, 

        ]
    )
    print(response.output_text)
    return response.output_text

if __name__  == "__main__":
    LLM_reply = get_name_hobby("Bo Zhang", r"whoareyou_cache\2.wav")
    print(LLM_reply)

    print(loads(LLM_reply)["speaker"])
    print(loads(LLM_reply)["hobby"])
