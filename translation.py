import google.generativeai as genai
from google.generativeai import GenerativeModel,ChatSession
from vertexai.preview.tokenization import get_tokenizer_for_model
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
    if len(response.text) < 1000:
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

def token_count(use_model_name:str,input_path:Path)->int:
    """input_pathのテキストのトークン数をカウントする。

    Args:
        use_model_name (str): apiで使用されているモデル名
        input_path (Path): token数をカウントするテキストのパス

    Returns:
        int: トークン数
    """
    if use_model_name=="models/gemini-1.5-pro":
        use_model_name="gemini-1.5-pro-002"
    elif use_model_name=="models/gemini-1.5-flash" or "models/gemini-2.0-flash-exp":
        use_model_name="gemini-1.5-flash-002"
    tokenizer=get_tokenizer_for_model(use_model_name)
    with open(input_path,'r') as f:
        text=f.read()
    response=tokenizer.count_tokens(text)
    return response.total_tokens

#TODO: 2.0はlong contextが弱いので、いい感じの場所で文章を区切れるようなしくみを導入すること
def translate_text(model:GenerativeModel,save_folder:Path,input_md_path:Path,is_free:bool):
    """input_md_pathのテキストを翻訳し、save_folder以下に保存する。

    Args:
        model (GenerativeModel): _description_
        save_folder (Path): _description_
        input_md_path (Path): _description_
        is_free (bool): apiは無料枠を使っているかどうか
    """
    print(f"{input_md_path.name}の翻訳を開始します。")
    chat = model.start_chat()
    with open(input_md_path,'r') as f:
        text = f.read()
    sf_path=save_folder / input_md_path.name
    response_info=do_first_chat(chat,sf_path,text)
    print(f"最初のチャットにかかった時間: {response_info[0]:.2f}秒\n 出力トークン数: {response_info[2]}")
    if model._model_name=="models/gemini-1.5-pro" and is_free:
        print("使用料制限により60秒待機します。")
        time.sleep(60)
    for i in range(10):
        print(f"{i+2}回目のチャットを開始します。")
        res=do_repeat_chat(chat,sf_path)
        if res is None:
            break
        print(f"{i+2}回目のチャットにかかった時間: {res[0]:.2f}秒\n 出力トークン数: {res[2]}")
        if model._model_name=="models/gemini-1.5-pro" and is_free:
            print("使用料制限により60秒待機します。")
            time.sleep(60)
    print(f"{input_md_path.name}の翻訳が完了しました。")
    
def translate_folder(use_model:str,save_folder:Path,input_folder:Path,is_free=True):
    """input_folder以下のテキストを翻訳し、save_folder以下に保存する。
    
    ある程度のロングコンテキストに強いgemini-1.5-proをおすすめする。

    Args:
        use_model (str): _description_
        save_folder (Path): _description_
        input_folder (Path): _description_
        is_free (bool): apiは無料枠を使っているかどうか
    """
    for input_md_path in input_folder.glob("*.md"):
        output_md_path=save_folder / input_md_path.name
        if output_md_path.exists():
            print(f"{output_md_path.name}はすでに存在します。")
            continue
        model=setup_genai(use_model)
        input_tokens=token_count(model._model_name,input_md_path)
        print(f"input_tokens: {input_tokens}")
        if input_tokens>10000:
            print(f"{input_md_path.name}のトークン数が10000を超えていますが大丈夫ですか？ y or n")
            ans=input()
            if ans=="n":
                continue
        translate_text(model,save_folder,input_md_path,is_free)
    print(f"{input_folder}内の全ての翻訳が完了しました。")

#TODO: 現在のmodelを2.0-flashに固定して最適化すること
if __name__ == "__main__":
    model=setup_genai(use_model="2.0_flash")
    use_model="1.5_pro"
    data_folder=Path(__file__).parent / "data"
    input_folder = data_folder / "input/deep_utopia/part5"
    save_folder=data_folder / "output/deep_utopia/part5"
    translate_folder(use_model,save_folder,input_folder)