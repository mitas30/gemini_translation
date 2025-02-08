import google.generativeai as genai
from google.generativeai import GenerativeModel,ChatSession
from vertexai.preview.tokenization import get_tokenizer_for_model
from dotenv import load_dotenv
import os,json,time
from pathlib import Path
from typing import Literal

def do_first_chat(chat:ChatSession,text:str)->tuple[str,float,int,int]:
    """最初のチャットを行い、翻訳者の能力を確認する。

    Args:
        chat (ChatSession): _description_
        text (str): 翻訳対象のテキスト

    Returns:
        tuple[str,float,int,int]: 
        - translated_md: 翻訳されたテキスト
        - take_time: チャットにかかった時間
        - prompt_token_count: プロンプトのトークン数
        - candidates_token_count: 返信のトークン数
    """
    st=time.time()
    response=chat.send_message(f"あなたは優れた翻訳者です。これから英語の長文を送るので、できるだけ長く日本語に翻訳してください。\n一度に翻訳できなくても、何回かに分けて完全な翻訳を作成する予定なので、あなたは端折らずに翻訳してください。\nまた、元のmdと同じ構成を保って出力することを心がけてください。\n\n<翻訳対象>\n{text}")
    translated_md=response.text
    take_time=time.time()-st
    usage = response.usage_metadata
    return translated_md,take_time,usage.prompt_token_count,usage.candidates_token_count 

def do_repeat_chat(chat:ChatSession)->tuple[str,float,int,int,bool]:
    """2回目以降のチャットで続きの翻訳を行う。

    Args:
        chat (ChatSession): _description_

    Returns:
        tuple[str,float,int,int,bool]:
        - translated_md: 翻訳されたテキスト
        - take_time: チャットにかかった時間
        - prompt_token_count: プロンプトのトークン数
        - candidates_token_count: 返信のトークン数
        - is_completed: 翻訳が完了したかどうか
    """
    st=time.time()
    response = chat.send_message(f"あなたは優れた翻訳者です。以前の人が翻訳してくれた部分を見て、続きを翻訳してください。\nただし、もし前の人で全文の翻訳が完了している場合は、「completed」とだけ返信してください。")
    usage = response.usage_metadata
    translated_text=response.text
    is_completed=False
    if usage.candidates_token_count < 100:
        print("翻訳が完了しました。")
        is_completed=True
    return translated_text,time.time()-st,usage.prompt_token_count,usage.candidates_token_count,is_completed

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
def translate_text(model:GenerativeModel,input_md:str)->str:
    """input_md_pathのテキストを翻訳し、save_folder以下に保存する。

    Args:
        model (GenerativeModel): _description_
        save_folder (Path): _description_
        input_md_path (Path): _description_
    """
    transltaed_md=""
    print(f"1回目のチャットを開始します。")
    chat = model.start_chat()
    response_info=do_first_chat(chat,input_md)
    transltaed_md+=response_info[0]
    print(f"最初のチャットにかかった時間: {response_info[1]:.2f}秒\n 出力トークン数: {response_info[3]}")
    for i in range(10):
        print(f"{i+2}回目のチャットを開始します。")
        res=do_repeat_chat(chat)
        transltaed_md+=res[0]
        print(f"{i+2}回目のチャットにかかった時間: {res[1]:.2f}秒\n 出力トークン数: {res[3]}")
        if res[4]:
            break
    return transltaed_md

# ─────────────────────────────
# ヘルパー関数: Markdown のテキストを約 max_words 単語ごとに分割する
def split_md_text(md_text: str, max_words: int) -> list[str]:
    """
    Markdown の文字列を、段落ごとに区切りながら
    約 max_words 単語になるように分割し、セグメントのリストを返す。

    Args:
        md_text (str): 分割対象の Markdown 文字列
        max_words (int): 1セグメントあたりの最大単語数の目安

    Returns:
        list[str]: 分割されたテキストのリスト
    """
    # 改行2つで段落ごとに分割
    paragraphs = md_text.split("\n\n")
    segments = []
    current_segment = ""
    current_word_count = 0

    for para in paragraphs:
        # 段落内の単語数を数える
        word_count = len(para.split())
        # もし現在のセグメントにこの段落を加えると max_words を超えてしまい、かつすでに何らかのテキストがあるなら…
        if current_segment and (current_word_count + word_count > max_words):
            segments.append(current_segment)
            # 新たなセグメントの開始
            current_segment = para
            current_word_count = word_count
        else:
            # まだセグメントに追加できる場合
            if current_segment:
                current_segment += "\n\n" + para
            else:
                current_segment = para
            current_word_count += word_count

    # 最後のセグメントを追加
    if current_segment:
        segments.append(current_segment)

    return segments

# ─────────────────────────────
# 必要に応じて、Markdown ファイルを読み込んで分割する関数
def split_markdown_file(md_path: Path, max_words: int=4000) -> list[str]:
    """
    指定された Markdown ファイルを読み込み、内容を分割してリストとして返す。

    Args:
        md_path (Path): 対象の Markdown ファイルパス
        max_words (int): 1セグメントあたりの最大単語数の目安

    Returns:
        list[str]: 分割されたテキストのリスト
    """
    with md_path.open("r", encoding="utf-8") as f:
        md_text = f.read()
    return split_md_text(md_text, max_words=max_words)

def translate_folder(use_model: str, save_folder: Path, input_folder: Path):
    """
    input_folder 以下の Markdown ファイルを翻訳し、save_folder 以下に保存する。
    ただし、トークン数が多い（10000 を超える）場合はファイル内容を分割して、
    分割された各セグメントごとに翻訳を実施します。

    Args:
        use_model (str): 利用するモデル名など
        save_folder (Path): 翻訳結果の保存先フォルダ
        input_folder (Path): 入力 Markdown ファイルが存在するフォルダ
    """
    for input_md_path in input_folder.glob("*.md"):
        output_md_path = save_folder / input_md_path.name
        if output_md_path.exists():
            print(f"{output_md_path.name}はすでに存在します。")
            continue

        model = setup_genai(use_model)
        # ※ token_count はモデルとファイルパスからトークン数を計測する既存関数と仮定
        input_tokens = token_count(model._model_name, input_md_path)
        print(f"{input_md_path.name} の input_tokens: {input_tokens}")

        # トークン数が多い場合は自動的に内容を分割する
        if input_tokens > 5000: 
            print(f"{input_md_path.name} のトークン数が 5000 を超えているため、内容を分割します。")
            segments = split_markdown_file(input_md_path)
            print(f"分割結果: {len(segments)} 個のセグメントに分割されました。")
            # 各セグメントごとに翻訳を実施（必要に応じて適宜実装してください）
            translated_segments = []
            for i, segment in enumerate(segments):
                print(f"セグメント {i+1} を翻訳中...")
                translated_segment = translate_text(model, segment)
                translated_segments.append(translated_segment)
            # セグメント間は「---」で区切って結合
            final_translated_text = "\n\n---\n\n".join(translated_segments)
            # 翻訳結果を保存
            with output_md_path.open("w", encoding="utf-8") as f:
                f.write(final_translated_text)
        else:
            # トークン数が少ない場合は従来通りに翻訳
            # ※ translate_text の実装が、ファイルパスから読み込んで翻訳する場合はそのままでOKです。
            with input_md_path.open("r", encoding="utf-8") as f:
                text = f.read()
            translated_text = translate_text(model, text)
            with output_md_path.open("w", encoding="utf-8") as f:
                f.write(translated_text)
    print(f"{input_folder} 内の全ての翻訳が完了しました。")


#TODO: 現在のmodelを2.0-flashに固定して最適化すること
if __name__ == "__main__":
    use_model="2.0_flash"
    data_folder=Path(__file__).parent / "data"
    input_folder = data_folder / "input/deep_utopia"
    save_folder=data_folder / "output/deep_utopia"
    translate_folder(use_model,save_folder,input_folder)