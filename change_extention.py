"""文書ファイルとMarkdownファイルの相互変換を行う。

現在想定している文書ファイルは、.docx形式のwordファイルと.epub形式のファイル。

Note:
    - pdf形式は、自動で変換することが難しいため、wordから開いてdocx形式に変換して保存してほしい。
"""

import subprocess
import argparse
from pathlib import Path

def convert_docx_to_markdown(input_file: Path, converted_md_save_path: Path):
    """
    DOCX → Markdown 変換。画像は <出力名>_media/ にまとめ、
    Markdown 内のリンクは相対パスにする。
    """
    workdir = converted_md_save_path.parent           
    media_dir = f"{converted_md_save_path.stem}_media"  

    cmd = [
        "pandoc",
        str(input_file),                  
        "--from=docx",
        "--to=markdown+pipe_tables+grid_tables",
        "--wrap=preserve",
        f"--extract-media={media_dir}",   
        "-o", converted_md_save_path.name 
    ]

    subprocess.run(cmd, check=True, cwd=workdir)
    print(f"変換完了: {converted_md_save_path} と {media_dir}/ が生成されました。")
        
def convert_all_docx_files_to_markdown_in_folder(target_folder: Path):
    """指定フォルダ以下のすべての.docxファイルをMarkdown形式に変換する。

    各docxファイルは同一ディレクトリに拡張子を.mdに変更したファイルとして出力されます。

    Args:
        target_folder (Path): 対象フォルダのパス
    """
    for docx_file in target_folder.rglob("*.docx"):
        output_file = docx_file.with_suffix('.md')
        print(f"変換開始: {docx_file} → {output_file}")
        convert_docx_to_markdown(docx_file, output_file)
        
def convert_markdown_to_docx(input_file: Path, output_file: Path, search_media_in_input: bool = False):
    """MarkdownファイルをDOCX形式に変換する。

    高精度な形式変換を目指し、Pandoc の変換機能を利用します。

    Args:
        input_file (Path): 入力となるMarkdownファイルのパス
        output_file (Path): 出力となるDOCXファイルのパス
        search_media_in_input (bool): 画像が見つからない場合、inputフォルダからも検索するか
    """
    import shutil
    import urllib.parse
    
    # 画像フォルダが存在するかチェック
    current_media_folder = input_file.parent / f"{input_file.stem}_media"
    
    # 出力フォルダに画像がない場合、inputフォルダから検索
    if search_media_in_input and not current_media_folder.exists():
        # data/output/folder -> data/input/folder のパスを構築
        try:
            data_folder = input_file.parents[2]  # data フォルダ
            relative_path = input_file.parent.relative_to(data_folder / "output")
            input_media_folder = data_folder / "input" / relative_path / f"{input_file.stem}_media"
            
            if input_media_folder.exists():
                print(f"inputフォルダから画像を参照: {input_media_folder}")
                # 一時的に画像フォルダをコピー
                temp_media_folder = input_file.parent / f"{input_file.stem}_media"
                shutil.copytree(input_media_folder, temp_media_folder)
                print(f"画像フォルダを一時コピー: {temp_media_folder}")
        except Exception as e:
            print(f"画像フォルダの検索中にエラー: {e}")
    
    # 基本の変換コマンド
    cmd = [
        "pandoc",
        str(input_file),
        "--from=markdown+smart",
        "--to=docx",
        "-o", str(output_file)
    ]
    
    try:
        workdir = input_file.parent
        subprocess.run(cmd, check=True, cwd=workdir)
        print(f"変換完了: {output_file} が作成されました。")
    except subprocess.CalledProcessError as e:
        print("変換中にエラーが発生しました:", e)

def convert_all_markdown_files_to_docx_in_folder(folder: Path, search_media_in_input: bool = False):
    """指定フォルダ以下のすべての.mdファイルをDOCX形式に変換する。

    各Markdownファイルは同一ディレクトリに拡張子を.docxに変更したファイルとして出力されます。

    Args:
        folder (Path): 対象フォルダのパス
        search_media_in_input (bool): 画像が見つからない場合、inputフォルダからも検索するか
    """
    for md_file in folder.rglob("*.md"):
        output_file = md_file.with_suffix('.docx')
        print(f"変換開始: {md_file} → {output_file}")
        convert_markdown_to_docx(md_file, output_file, search_media_in_input)

class EpubToMarkdownConverter:
    def __init__(self,processing_folder: Path):
        self.processing_folder = processing_folder
        
    def convert_all_epub_files_to_markdown_in_folder(self):
        """指定フォルダ以下のすべての.epubファイルをMarkdown形式に変換。

        各epubファイルは同一ディレクトリに拡張子を.mdに変更したファイルとして出力されます。
        """
        for epub_file in self.processing_folder.rglob("*.epub"):
            output_file = epub_file.with_suffix('.md')
            print(f"変換開始: {epub_file} → {output_file}")
            self._convert_epub_to_markdown(epub_file, output_file)

    def _convert_epub_to_markdown(self,input_file: Path, output_md: Path)->None:
        """EPUB を Markdown に変換する

        この関数は、EPUBファイルをMarkdown形式に変換します。

        Args:
            input_file (Path): 入力となるEPUBファイルのパス
            output_md (Path): 出力となるMarkdownファイルのパス。
        """
        workdir = output_md.parent
        media_dir = f"{output_md.stem}_media"
        cmd = [
            "pandoc",
            str(input_file),
            "--from=epub",
            "--to=markdown+pipe_tables+grid_tables",
            "--wrap=preserve",
            f"--extract-media={media_dir}",
            "-o", output_md.name
        ]
        subprocess.run(cmd, check=True, cwd=workdir)
        print(f"EPUB→Markdown 完了: {output_md} (+ {media_dir}/)")
    
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Convert file extensions")
    parser.add_argument("-M","--mode", type=str, choices=["docx2md", "md2docx","epub2md"], help="変換モードの決定",required=True)
    parser.add_argument("-F","--target_data_folder",type=Path,help="dataフォルダ以下の処理したい相対フォルダパスを入力してください。",default=Path("input/tmp"))
    parser.add_argument("--search-input", action="store_true", help="md2docx変換時に画像がない場合、inputフォルダからも検索する")
    
    args=parser.parse_args()
    target_folder= Path(__file__).parent/ "data" / args.target_data_folder
    
    if args.mode=="docx2md":
        convert_all_docx_files_to_markdown_in_folder(target_folder)
    elif args.mode=="md2docx":
        convert_all_markdown_files_to_docx_in_folder(target_folder, args.search_input)
    elif args.mode=="epub2md":
        converter = EpubToMarkdownConverter(target_folder)
        converter.convert_all_epub_files_to_markdown_in_folder()
        
    print(f"{target_folder} の変換が完了しました。")