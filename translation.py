import google.generativeai as genai
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
import os,json
from pathlib import Path

def translate_text(model:GenerativeModel,save_file_path:Path):
    chat = model.start_chat()
    response=chat.send_message("Translate the following text into English: 你好")
    with open(save_file_path,'a') as f:
        json.dump(response.text,f,ensure_ascii=False)
        
    for i in range(10):
        response = chat.send_message()
        if len(response.text) < 20:
            break
        with open(save_file_path,'a') as f:
            json.dump(response.text,f,ensure_ascii=False)
    

if __name__ == "__main__":
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Explain how AI works")
    print(response.text)