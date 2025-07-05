from dotenv import load_dotenv
import os,json,time,re,asyncio
from pathlib import Path
from typing import Literal
from openai import AsyncOpenAI,OpenAI
import tiktoken

'''
class GemminiTranslator:
    """Gemminiを使って翻訳を行うクラス
    """
    def __init__(self,
                 use_model:Literal["1.5_flash","1.5_pro","2.0_flash","2.0_pro"])->GenerativeModel:
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        if use_model=="1.5_flash":
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        elif use_model=="1.5_pro":
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        elif use_model=="2.0_flash":
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        elif use_model=="2.0_pro":
            self.model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        else:
            raise ValueError("use_modelが不正です。")

    def translate_folder(self, save_folder: Path, input_folder: Path,split_words:int=4000):
        """
        input_folder 以下の Markdown ファイルを翻訳し、save_folder 以下に保存する。
        ただし、ファイルの文字が多い場合はファイル内容を分割して、
        分割された各セグメントごとに翻訳を実施します。
    
        Args:
            save_folder (Path): 翻訳結果の保存先フォルダ
            input_folder (Path): 入力 Markdown ファイルが存在するフォルダ
            split_words (int): 1セグメントあたりの最大単語数の目安
        """
        for input_md_path in input_folder.glob("*.md"):
            output_md_path = save_folder / input_md_path.name
            if output_md_path.exists():
                print(f"{output_md_path.name}はすでに存在します。")
                continue
            
            input_tokens = self.token_count(input_md_path)
            print(f"{input_md_path.name} の input_tokens: {input_tokens}")
    
            # トークン数が多い場合は自動的に内容を分割する
            # ? ただし、段落ごとの分割であり、段落は\n{2,}で区切られていると仮定する
            if input_tokens > split_words: 
                print(f"{input_md_path.name} のトークン数が {split_words} を超えているため、内容を分割します。")

                seg_lens,segments = TextSplitter.split_markdown_file(input_md_path,split_words)
                print(f"分割結果: {len(segments)} 個のセグメントに分割されました。")
                
                translated_segments = []
                for i, segment in enumerate(segments):
                    print(f"セグメント {i+1} を翻訳中... 入力文書は{seg_lens[i]}単語")
                    translated_segment = self.translate_text(segment)
                    translated_segments.append(translated_segment)
                # ? セグメント間は「----------」で区切って結合
                final_translated_text = "\n\n----------\n\n".join(translated_segments)
                # 翻訳結果を保存
                with output_md_path.open("w", encoding="utf-8") as f:
                    f.write(final_translated_text)
            else:
                with input_md_path.open("r", encoding="utf-8") as f:
                    text = f.read()
                translated_text = self.translate_text(text)
                with output_md_path.open("w", encoding="utf-8") as f:
                    f.write(translated_text)
        print(f"{input_folder} 内の全ての翻訳が完了しました。")

    def token_count(self,input_path:Path)->int:
        """input_pathのテキストのトークン数をカウントする。

        Args:
            input_path (Path): token数をカウントするテキストのパス

        Returns:
            int: トークン数
        """
        use_model_name=self.model.model_name
        if use_model_name=="models/gemini-1.5-pro":
            use_model_name="gemini-1.5-pro-002"
        elif use_model_name=="models/gemini-2.0-pro-exp-02-05" or "models/gemini-2.0-flash-exp" or "models/gemini-1.5-flash":
            use_model_name="gemini-1.5-flash-002"
        else:
            raise ValueError("use_model_nameが不正です。")
        tokenizer=get_tokenizer_for_model(use_model_name)
        with open(input_path,'r') as f:
            text=f.read()
        response=tokenizer.count_tokens(text)
        return response.total_tokens

    def translate_text(self,input_md:str)->str:
        """input_md_pathのテキストを翻訳し、save_folder以下に保存する。

        Args:
            save_folder (Path): _description_
            input_md_path (Path): _description_
        """
        transltaed_md=""
        print(f"翻訳を開始します。")
        chat = self.model.start_chat()
        response_info=self._translate(chat,input_md)
        transltaed_md+=response_info[0]
        print(f"翻訳にかかった時間: {response_info[1]:.2f}秒 output_tokens: {response_info[3]}")
        use_model_name=self.model.model_name
        model_info=genai.get_model(use_model_name)
        if response_info[3]> model_info.output_token_limit:
            raise ValueError(f"output_tokenが上限まで出力されている。")
        return transltaed_md
    
    def _translate(self,chat:ChatSession,text:str)->tuple[str,float,int,int]:
        """textを翻訳する。

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
        response=chat.send_message(f"あなたは優れた翻訳者です。これから英語の長文を送るので、全文を自然な日本語に翻訳してください。\nまた、元のmdと同じ構成を保って出力することを心がけてください。\n\n<翻訳対象>\n{text}")
        translated_md=response.text
        take_time=time.time()-st
        usage = response.usage_metadata
        return translated_md,take_time,usage.prompt_token_count,usage.candidates_token_count 
'''    
    
class GPTTranslator:
    """GPTを使って翻訳を行うクラス
    """
    def __init__(self, use_model: Literal["o3-mini","o4-mini-2025-04-16"]):
        load_dotenv()
        self.client = AsyncOpenAI(api_key=os.getenv("OAI_API_KEY"))  
        self.use_model = use_model
        
    async def translate_folder(self, save_folder: Path, input_folder: Path, split_words: int = 4000):
        for input_md_path in input_folder.glob("*.md"):
            output_md_path = save_folder / input_md_path.name
            if output_md_path.exists():
                print(f"{output_md_path.name}はすでに存在します。")
                continue

            input_tokens = self.token_count(input_md_path)
            print(f"{input_md_path.name} の input_tokens: {input_tokens}")

            if input_tokens > split_words:
                print(f"{input_md_path.name} のトークン数が {split_words}トークン を超えているため、内容を{split_words}単語で分割します。")
                seg_lens, segments = TextSplitter.split_markdown_file(input_md_path, split_words)
                print(f"分割結果: {len(segments)} 個のセグメントに分割されました。")
                for i,seg_len in enumerate(seg_lens):
                    print(f"セグメント{i+1}の単語数: {seg_len} 単語")

                # 各セグメントを並列翻訳
                coros = [self.translate_text(i,seg) for i,seg in enumerate(segments)]
                translated_segments = await asyncio.gather(*coros)  # 順序は保持される
                final_text = "\n\n----------\n\n".join(translated_segments)
            else:
                with input_md_path.open("r", encoding="utf-8") as f:
                    text = f.read()
                final_text = await self.translate_text(0,text)

            with output_md_path.open("w", encoding="utf-8") as f:
                print(f"翻訳結果を{output_md_path.name}に保存します。")
                f.write(final_text)

        print(f"{input_folder} 内の全ての翻訳が完了しました。")
        
    def token_count(self,input_path:Path)->int:
        """テキストのtoken数をカウントする。

        Args:
            input_path (Path): token数を知りたいテキストのpath

        Returns:
            int: 文章のトークン数
        """
        enc = tiktoken.encoding_for_model("gpt-4o")
        with open(input_path,'r') as f:
            text=f.read()
        tokens=enc.encode(text)
        return len(tokens)
        
    async def translate_text(self, seg_idx:int,input_md: str) -> str:
        translated, used_tokens = await self._translate(input_md)
        print(f"セグメント{seg_idx+1}でのGPTの応答トークン数: {used_tokens}")
        self._check_exceeding_output_length(used_tokens)
        return translated
    
    def _check_exceeding_output_length(self,output_len:int)->None:
        """output_tokenが上限を超えているかを確認する。
        """
        max_output_token=15000
        if self.use_model =="o3-mini-2025-01-31" or self.use_model=="o4-mini-2025-04-16":
            max_output_token=100000

        if output_len> max_output_token:
            raise ValueError(f"output_tokenが上限まで出力されている。")
    
    async def _translate(self, text: str) -> tuple[str, int]:
        messages=[
                    {
                        "role": "developer",
                        "content": (
                            "あなたは優れた翻訳者です。これから英語の長文を送るので、与えられたすべての文章を自然な日本語に翻訳してください。\n"
                            "ただし、絶対に翻訳結果の日本語文章のみを出力してください。\n"
                            "また、元のマークダウンと同じ構成を保って出力することを心がけてください。"
                        ),
                    },
                    {"role": "user", "content": f"<翻訳対象>:\n{text}"},
                ]
        try:
            completion = await self.client.chat.completions.create(
                model=self.use_model,
                reasoning_effort="high",
                messages=messages,
            )
        except Exception as e:
            print(f"翻訳中にエラーが発生: {e}")
            print(f"エラー対象のメッセージ: {messages}")
            raise
        msg = completion.choices[0].message
        return msg.content, completion.usage.completion_tokens
    
class TextSplitter:
    """テキストを分割するクラス
    """
    def __init__(self):
        pass
    @classmethod    
    def split_markdown_file(cls,md_path: Path, split_words: int) -> tuple[list[int],list[str]]:
        """
        指定された Markdown ファイルを読み込み、内容を分割してリストとして返す。

        Args:
            md_path (Path): 対象の Markdown ファイルパス
            split_words (int): 1セグメントあたりの最大単語数の目安

        Returns:
            tuple[list[int], list[str]]: 各セグメントの単語数と分割されたテキストのリスト
        """
        with md_path.open("r", encoding="utf-8") as f:
            md_text = f.read()
        return cls._split_md_text(md_text, split_words=split_words)
    
    @classmethod
    def _split_md_text(cls,md_text: str, split_words: int) -> tuple[list[int],list[str]]:
        """
        Markdown の文字列を、段落ごと(\n\n)に区切りながら
        約 split_words 単語になるように分割し、セグメントのリストを返す。

        Args:
            md_text (str): 分割対象の Markdown 文字列
            split_words (int): 1セグメントあたりの最大単語数の目安

        Returns:
            tuple[list[int], list[str]]: 各セグメントの単語数と分割されたテキストのリスト
        """
        # 改行2つ以上で段落ごとに分割
        paragraphs = re.split(r"\n{2,}", md_text)
        seg_lens = []
        segments = []
        current_segment = ""
        current_word_count = 0

        for para in paragraphs:
            # 段落内の単語数を数える
            word_count = len(para.split())
            # もし現在のセグメントにこの段落を加えると split_words を超えてしまい、かつすでに何らかのテキストがあるなら…
            if current_segment and (current_word_count + word_count > split_words):
                segments.append(current_segment)
                seg_lens.append(current_word_count)
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
            seg_lens.append(current_word_count)

        return seg_lens, segments