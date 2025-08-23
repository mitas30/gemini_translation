"""
End-to-End翻訳スクリプト
DOCX → Markdown → 翻訳 → DOCX の全工程を自動化

使用例:
    python e2e_translation.py -M gemini -F input_folder --input-file document.docx
    python e2e_translation.py -M gpt -F input_folder --all-docx
"""

import argparse
import asyncio
from pathlib import Path
import shutil
from typing import Optional, List

from translation import GPTTranslator, GeminiTranslator
from change_extention import (
    convert_docx_to_markdown, 
    convert_markdown_to_docx,
    convert_all_docx_files_to_markdown_in_folder
)


class E2ETranslator:
    """End-to-End翻訳を実行するクラス"""
    
    def __init__(self, 
                 model_type: str, 
                 target_folder: Path,
                 rate_limit: int = 5,
                 gemini_model: str = "2.5-flash"):
        """
        Args:
            model_type: 使用するモデル ("gemini" or "gpt")
            target_folder: 処理対象のフォルダ (data/input以下)
            rate_limit: Gemini使用時のレート制限
            gemini_model: Geminiモデルの種類 ("2.5-flash" or "2.5-pro")
        """
        self.model_type = model_type
        self.rate_limit = rate_limit
        self.gemini_model = gemini_model
        
        # パス設定
        self.data_folder = Path(__file__).parent / "data"
        self.input_folder = self.data_folder / "input" / target_folder
        self.output_folder = self.data_folder / "output" / target_folder
        
        # 出力フォルダを作成
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 翻訳器を初期化
        if model_type == "gemini":
            self.translator = GeminiTranslator(
                use_model=gemini_model, 
                calls_per_minute=rate_limit
            )
        elif model_type == "gpt":
            self.translator = GPTTranslator("o3-mini-2025-01-31", calls_per_minute=rate_limit)
        else:
            raise ValueError(f"サポートされていないモデル: {model_type}")
    
    def translate_single_docx(self, docx_file: Path) -> Path:
        """
        単一のDOCXファイルを翻訳する
        
        Args:
            docx_file: 翻訳対象のDOCXファイル
            
        Returns:
            翻訳済みDOCXファイルのパス
        """
        print(f"\n=== {docx_file.name} のE2E翻訳を開始 ===")
        
        # ステップ1: DOCX → Markdown変換
        print("ステップ1: DOCX → Markdown変換")
        md_file = docx_file.with_suffix('.md')
        convert_docx_to_markdown(docx_file, md_file)
        
        if not md_file.exists():
            raise FileNotFoundError(f"Markdown変換に失敗: {md_file}")
        
        # ステップ2: Markdown翻訳
        print("ステップ2: Markdown翻訳")
        output_md_file = self.output_folder / md_file.name
        
        # 単一ファイル翻訳を実行
        if self.model_type == "gemini":
            self.translator.translate_single_file(md_file, output_md_file)
        else:  # GPT
            asyncio.run(self.translator.translate_single_file(md_file, output_md_file))
        
        if not output_md_file.exists():
            raise FileNotFoundError(f"翻訳に失敗: {output_md_file}")
        
        # ステップ3: 翻訳済みMarkdown → DOCX変換
        print("ステップ3: 翻訳済みMarkdown → DOCX変換")
        output_docx_file = output_md_file.with_suffix('.docx')
        convert_markdown_to_docx(output_md_file, output_docx_file, search_media_in_input=True)
        
        if not output_docx_file.exists():
            raise FileNotFoundError(f"DOCX変換に失敗: {output_docx_file}")
        
        print(f"✅ 翻訳完了: {output_docx_file}")
        return output_docx_file
    
    def translate_all_docx_in_folder(self) -> List[Path]:
        """
        フォルダ内のすべてのDOCXファイルを翻訳する
        
        Returns:
            翻訳済みDOCXファイルのリスト
        """
        print(f"\n=== {self.input_folder} 内の全DOCXファイルのE2E翻訳を開始 ===")
        
        # ステップ1: フォルダ内の全DOCX → Markdown変換
        print("ステップ1: フォルダ内の全DOCX → Markdown変換")
        convert_all_docx_files_to_markdown_in_folder(self.input_folder)
        
        # ステップ2: 全Markdown翻訳
        print("ステップ2: 全Markdown翻訳")
        if self.model_type == "gemini":
            self.translator.translate_folder(self.output_folder, self.input_folder)
        else:  # GPT
            asyncio.run(self.translator.translate_folder(self.output_folder, self.input_folder))
        
        # ステップ3: 翻訳済み全Markdown → DOCX変換
        print("ステップ3: 翻訳済み全Markdown → DOCX変換")
        output_docx_files = []
        
        for md_file in self.output_folder.glob("*.md"):
            print(f"DOCX変換中: {md_file.name}")
            docx_file = md_file.with_suffix('.docx')
            try:
                convert_markdown_to_docx(md_file, docx_file, search_media_in_input=True)
                if docx_file.exists():
                    output_docx_files.append(docx_file)
                    print(f"✅ 変換完了: {docx_file.name}")
                else:
                    print(f"❌ 変換失敗: {docx_file.name}")
            except Exception as e:
                print(f"❌ 変換エラー {md_file.name}: {e}")
        
        print(f"\n🎉 E2E翻訳完了! {len(output_docx_files)}個のDOCXファイルが生成されました")
        return output_docx_files
    
    def clean_intermediate_files(self, keep_translated_md: bool = True):
        """
        中間ファイル（変換前のMarkdownファイル）をクリーンアップ
        
        Args:
            keep_translated_md: 翻訳済みMarkdownファイルを保持するか
        """
        print("\n🧹 中間ファイルのクリーンアップ中...")
        
        # 入力フォルダの変換用Markdownファイルを削除
        for md_file in self.input_folder.glob("*.md"):
            # 元々DOCXファイルから変換されたものかチェック
            corresponding_docx = md_file.with_suffix('.docx')
            if corresponding_docx.exists():
                md_file.unlink()
                print(f"削除: {md_file}")
                
                # 対応するメディアフォルダも削除
                media_folder = md_file.parent / f"{md_file.stem}_media"
                if media_folder.exists():
                    shutil.rmtree(media_folder)
                    print(f"削除: {media_folder}")
        
        # 必要に応じて翻訳済みMarkdownファイルも削除
        if not keep_translated_md:
            for md_file in self.output_folder.glob("*.md"):
                md_file.unlink()
                print(f"削除: {md_file}")


def main():
    parser = argparse.ArgumentParser(
        description="DOCX → Markdown → 翻訳 → DOCX のEnd-to-End翻訳",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイルを翻訳 (gemini-2.5-flash使用)
  python e2e_translation.py -M gemini -F my_folder --input-file document.docx
  
  # 単一ファイルを翻訳 (gemini-2.5-pro使用)
  python e2e_translation.py -M gemini -G 2.5-pro -F my_folder --input-file document.docx
  
  # フォルダ内の全DOCXファイルを翻訳 (GPT使用)
  python e2e_translation.py -M gpt -F my_folder --all-docx
  
  # 翻訳後に中間ファイルを削除
  python e2e_translation.py -M gemini -F my_folder --all-docx --cleanup
        """
    )
    
    parser.add_argument(
        "-M", "--use_model", 
        type=str, 
        required=True, 
        choices=["gemini", "gpt"],
        help="使用する翻訳モデル"
    )
    parser.add_argument(
        "-G", "--gemini_model", 
        type=str, 
        choices=["2.5-flash", "2.5-pro"],
        default="2.5-flash",
        help="Geminiモデルの種類 (デフォルト: 2.5-flash)"
    )
    parser.add_argument(
        "-F", "--target_data_folder", 
        type=Path,
        help="data/inputフォルダ以下の処理したい相対フォルダパス", 
        default=Path("tmp")
    )
    parser.add_argument(
        "-R", "--rate_limit", 
        type=int, 
        help="1分間あたりのLLM_API呼び出し回数制限", 
        default=5
    )
    
    # ファイル指定オプション
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument(
        "--input-file", 
        type=str,
        help="翻訳対象の単一DOCXファイル名（拡張子含む）"
    )
    file_group.add_argument(
        "--all-docx", 
        action="store_true",
        help="フォルダ内の全DOCXファイルを翻訳"
    )
    
    # オプション
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="翻訳後に中間ファイル（変換前Markdown）を削除"
    )
    parser.add_argument(
        "--keep-translated-md", 
        action="store_true", 
        default=True,
        help="翻訳済みMarkdownファイルを保持（デフォルト: True）"
    )
    
    args = parser.parse_args()
    
    try:
        # E2E翻訳器を初期化
        e2e_translator = E2ETranslator(
            model_type=args.use_model,
            target_folder=args.target_data_folder,
            rate_limit=args.rate_limit,
            gemini_model=args.gemini_model
        )
        
        # 翻訳実行
        if args.input_file:
            # 単一ファイル翻訳
            input_file = e2e_translator.input_folder / args.input_file
            if not input_file.exists():
                print(f"❌ エラー: ファイルが見つかりません: {input_file}")
                return 1
            
            output_file = e2e_translator.translate_single_docx(input_file)
            print(f"\n🎉 翻訳完了: {output_file}")
            
        else:
            # 全ファイル翻訳
            output_files = e2e_translator.translate_all_docx_in_folder()
            if not output_files:
                print("❌ 翻訳されたファイルがありません")
                return 1
        
        # クリーンアップ
        if args.cleanup:
            e2e_translator.clean_intermediate_files(args.keep_translated_md)
        
        print("\n✨ E2E翻訳処理が正常に完了しました！")
        return 0
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
