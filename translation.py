import google.generativeai as genai
from google.generativeai import GenerativeModel,ChatSession
from dotenv import load_dotenv
import os,json,time
from pathlib import Path
from typing import Literal

def do_first_chat(chat:ChatSession,sf_path:Path,text:str)->tuple[float,int,int]:
    """最初のチャットを行い、翻訳者の能力を確認する。

    Args:
        chat (ChatSession): _description_
        sf_path (Path): _description_
        text (str): 翻訳対象のテキスト

    Returns:
        tuple[float,int,int]: 
        - take_time: チャットにかかった時間
        - prompt_token_count: プロンプトのトークン数
        - candidates_token_count: 返信のトークン数
    """
    st=time.time()
    response=chat.send_message(f"あなたは優れた翻訳者です。これから英語の長文を送るので、できるだけ長く日本語に翻訳してください。\n一度に翻訳できなくても、何回かに分けて完全な翻訳を作成する予定なので、あなたは端折らずに翻訳してください。\nまた、元のmdと同じ構成を保って出力することを心がけてください。\n\n<翻訳対象>\n{text}")
    with open(sf_path,'x') as f:
        f.write(response.text+'\n')
    take_time=time.time()-st
    usage = response.usage_metadata
    return take_time,usage.prompt_token_count,usage.candidates_token_count 

def do_repeat_chat(chat:ChatSession,sf_path:Path)->tuple[float,int,int]:
    st=time.time()
    response = chat.send_message(f"あなたは優れた翻訳者です。以前の人が翻訳してくれた部分を見て、続きを翻訳してください。\nただし、もし前の人で全文の翻訳が完了している場合は、「completed」とだけ返信してください。")
    if len(response.text) < 20:
        print("翻訳が完了しました。")
        return None
    with open(sf_path,'a') as f:
        f.write(response.text+'\n')
    usage = response.usage_metadata      
    return time.time()-st,usage.prompt_token_count,usage.candidates_token_count

def setup_genai(use_model:Literal["1.5_flash","1.5_pro","2.0_flash"]):
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    if use_model=="1.5_flash":
        model = genai.GenerativeModel("gemini-1.5-flash")
    elif use_model=="1.5_pro":
        model = genai.GenerativeModel("gemini-1.5-pro")
    elif use_model=="2.0_flash":
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
    return model

def translate_text(model:GenerativeModel,save_folder:Path,input_md_path:Path):
    """input_md_pathのテキストを翻訳し、save_folder以下に保存する。

    Args:
        model (GenerativeModel): _description_
        save_folder (Path): _description_
        input_md_path (Path): _description_
    """
    print(f"{input_md_path.name}の翻訳を開始します。")
    chat = model.start_chat()
    with open(input_md_path,'r') as f:
        text = f.read()
    sf_path=save_folder / input_md_path.name
    take_time=do_first_chat(chat,sf_path,text)
    print(f"最初のチャットにかかった時間: {take_time[0]:.2f}秒")
    for i in range(10):
        res=do_repeat_chat(chat,sf_path)
        print(f"{i+2}回目のチャットにかかった時間: {res[0]:.2f}秒")
        if res is None:
            break
    print(f"{input_md_path.name}の翻訳が完了しました。")
    
if __name__ == "__main__":
    model=setup_genai(use_model="1.5_pro")
    data_folder=Path(__file__).parent / "data"
    input_md_path = data_folder / "output_md" / "part4_thought/ch21_Problem Solving.md"
    save_folder=data_folder / "output" / "part4_thought/ch21_Problem Solving.md"
    save_folder.mkdir(parents=True,exist_ok=True)
    translate_text(model,save_folder,input_md_path)