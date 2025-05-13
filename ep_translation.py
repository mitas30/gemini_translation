""" 翻訳タスクを行うスクリプト
"""

from translation import GemminiTranslator, GPTTranslator
from argparse import ArgumentParser
from pathlib import Path
import asyncio

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--use_model", type=str,required=True,choices=["gemini","gpt"] )
    
    args=parser.parse_args()
    
    data_folder=Path(__file__).parent / "data"
    target_folder = ""
    input_folder = data_folder / f"input/deep_work"
    save_folder=data_folder / f"output/deep_work"
    save_folder.mkdir(parents=True,exist_ok=True)
    gpt_model="o3-mini"
    
    if args.use_model=="gemini":
        translator=GemminiTranslator(use_model="2.0_pro")
        translator.translate_folder(save_folder,input_folder)
    elif args.use_model=="gpt":
        translator=GPTTranslator(use_model="o3-mini")
        asyncio.run(translator.translate_folder(save_folder, input_folder))