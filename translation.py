from dotenv import load_dotenv
import os,json,time,re,asyncio
from pathlib import Path
from typing import Literal
from openai import AsyncOpenAI
import tiktoken
from google import genai

class GeminiTranslator:
    """
    Geminiを使って非同期で翻訳を行うクラス (genai.Client 使用版)
    """
    def __init__(self,
                 use_model: Literal["2.5-flash", "2.5-pro"]):
        """
        Args:
            use_model (Literal): 使用するモデルを選択します。
                - "2.5-flash": gemini-2.5-flash
                - "2.5-pro": gemini-2.5-pro
        """
        load_dotenv()
        # genai.Client() は自動的に環境変数 GEMINI_API_KEY または GOOGLE_API_KEY を読み込みます
        self.client = genai.Client()

        model_map = {
            "2.5-flash": "gemini-2.5-flash",
            "2.5-pro": "gemini-2.5-pro",
        }
        
        self.model_name = model_map.get(use_model)
        if not self.model_name:
            raise ValueError(f"指定されたモデル '{use_model}' は不正です。")

    def translate_folder(self, save_folder: Path, input_folder: Path, token_threshold: int = 15000, split_words: int = 4000):
        """
        input_folder 以下の Markdown ファイルを同期的に1つずつ翻訳し、save_folder 以下に保存する。
        ファイルサイズが大きい場合は内容を分割し、そのセグメントのみを並列で翻訳処理を行う。
    
        Args:
            save_folder (Path): 翻訳結果の保存先フォルダ
            input_folder (Path): 入力 Markdown ファイルが存在するフォルダ
            token_threshold (int): ファイルを分割するか判断するためのトークン数のしきい値
            split_words (int): ファイルを分割する際の、1セグメントあたりの最大単語数の目安
        """
        save_folder.mkdir(parents=True, exist_ok=True)
        
        # 処理するファイルをリスト化
        md_files = list(input_folder.glob("*.md"))

        for input_md_path in md_files:
            output_md_path = save_folder / input_md_path.name
            if output_md_path.exists():
                print(f"{output_md_path.name} はすでに存在します。スキップします。")
                continue
            # 各ファイルを同期的に処理するために asyncio.run を使用
            asyncio.run(self._process_single_file(input_md_path, output_md_path, token_threshold, split_words))

        print(f"{input_folder} 内の全ての翻訳が完了しました。")

    async def _process_single_file(self, input_md_path: Path, output_md_path: Path, token_threshold: int, split_words: int):
        """単一のファイルを処理する非同期ヘルパーメソッド"""
        print(f"処理開始: {input_md_path.name}")
        try:
            # token_count は同期的メソッドなので await は不要
            input_tokens = self.token_count(input_md_path)
            print(f"{input_md_path.name} のトークン数: {input_tokens}")

            if input_tokens > token_threshold: 
                print(f"トークン数がしきい値 ({token_threshold}) を超えているため、内容を分割します。")
                _, segments = TextSplitter.split_markdown_file(input_md_path, split_words)
                print(f"分割結果: {len(segments)} 個のセグメント")
                for i, seg in enumerate(segments):
                    seg_len = len(seg.split())
                    print(f"セグメント {i + 1} の単語数: {seg_len} 単語")
                
                # 各セグメントを並列翻訳
                coros = [self.translate_text(i, seg) for i, seg in enumerate(segments)]
                translated_segments = await asyncio.gather(*coros)
                final_translated_text = "\n\n----------\n\n".join(translated_segments)
            else:
                with input_md_path.open("r", encoding="utf-8") as f:
                    text = f.read()
                final_translated_text = await self.translate_text(0, text)

            with output_md_path.open("w", encoding="utf-8") as f:
                f.write(final_translated_text)
            print(f"保存完了: {output_md_path.name}")

        except Exception as e:
            print(f"エラー: {input_md_path.name} の処理中にエラーが発生しました: {e}")
            raise


    def token_count(self, input_path: Path) -> int:
        """
        指定されたテキストファイルのトークン数をカウントする。

        Args:
            input_path (Path): トークン数をカウントするテキストのパス

        Returns:
            int: トークン数
        """
        with input_path.open('r', encoding="utf-8") as f:
            text = f.read()
        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=text
        )
        return response.total_tokens

    async def translate_text(self, seg_idx: int, input_md: str) -> str:
        """
        単一のテキストセグメントを翻訳する。

        Args:
            seg_idx (int): セグメントのインデックス (ログ出力用)
            input_md (str): 翻訳対象のMarkdown文字列

        Returns:
            str: 翻訳されたテキスト
        """
        print(f"セグメント {seg_idx + 1} の翻訳を開始...")
        translated_md, take_time, _, candidates_tokens = await self._translate(input_md)
        print(f"セグメント {seg_idx + 1} の翻訳が完了。所要時間: {take_time:.2f}秒, 出力トークン数: {candidates_tokens}")
        
        # 出力トークン数の上限チェック
        model_info = self.client.models.get(model=f'models/{self.model_name}')
        if candidates_tokens >= model_info.output_token_limit:
            raise ValueError(f"出力トークンがモデルの上限 ({model_info.output_token_limit}) に達している可能性があります。")
        
        return translated_md
    
    async def _translate(self, text: str) -> tuple[str, float, int, int]:
        """
        Gemini API を呼び出してテキストを翻訳する内部メソッド。

        Args:
            text (str): 翻訳対象のテキスト

        Returns:
            tuple[str, float, int, int]: (翻訳結果, 処理時間, 入力トークン数, 出力トークン数)
        """
        st = time.time()
        prompt = (
            "あなたは優れた翻訳者です。これから英語の長文を送るので、全文を自然な日本語に翻訳してください。\n"
            "また、元のマークダウンと同じ構成を保って出力することを心がけてください。\n\n"
            f"<翻訳対象>\n{text}"
        )
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            translated_md = response.text
            usage = response.usage_metadata
            take_time = time.time() - st
            return translated_md, take_time, usage.prompt_token_count, usage.candidates_token_count
        except Exception as e:
            print(f"Gemini APIの呼び出し中にエラーが発生しました: {e}")
            raise

    
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