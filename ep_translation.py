""" 翻訳タスクを行うスクリプト
"""

from translation import GPTTranslator
from argparse import ArgumentParser
from pathlib import Path
import asyncio

# ? inputは、markdown形式のファイルを想定している
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-M","--use_model", type=str,required=True,choices=["gemini","gpt"] )
    parser.add_argument("-F","--target_data_folder",type=Path,help="data/inputフォルダ以下の処理したい相対フォルダパスを入力してください。",default=Path("tmp"))
    
    args=parser.parse_args()
    
    data_folder=Path(__file__).parent / "data"
    input_folder = data_folder / Path("input") / args.target_data_folder
    save_folder=data_folder / Path("output") / args.target_data_folder
    save_folder.mkdir(parents=True,exist_ok=True)
    # ? o4-miniは文章のチェックが厳しいためo3-miniを使用
    gpt_model="o3-mini-2025-01-31"
    gemini_model="2.0_pro"
    
    if args.use_model=="gemini":
        print("Gemini model is not implemented yet.")
        #translator=GemminiTranslator(use_model=gemini_model)
        #translator.translate_folder(save_folder,input_folder)
    elif args.use_model=="gpt":
        translator=GPTTranslator(gpt_model)
        asyncio.run(translator.translate_folder(save_folder, input_folder))