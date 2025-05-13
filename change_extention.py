"""内容をなるべく壊さずに、docxファイルとMarkdownファイルの相互変換を行うスクリプト。
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
        
def convert_markdown_to_docx(input_file: Path, output_file: Path):
    """MarkdownファイルをDOCX形式に変換する。

    高精度な形式変換を目指し、Pandoc の変換機能を利用します。

    Args:
        input_file (Path): 入力となるMarkdownファイルのパス
        output_file (Path): 出力となるDOCXファイルのパス
    """
    cmd = [
        "pandoc",
        str(input_file),
        "--from=markdown+smart",
        "--to=docx",
        "-o", str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"変換完了: {output_file} が作成されました。")
    except subprocess.CalledProcessError as e:
        print("変換中にエラーが発生しました:", e)

def convert_all_markdown_files_to_docx_in_folder(folder: Path):
    """指定フォルダ以下のすべての.mdファイルをDOCX形式に変換する。

    各Markdownファイルは同一ディレクトリに拡張子を.docxに変更したファイルとして出力されます。

    Args:
        folder (Path): 対象フォルダのパス
    """
    for md_file in folder.rglob("*.md"):
        output_file = md_file.with_suffix('.docx')
        print(f"変換開始: {md_file} → {output_file}")
        convert_markdown_to_docx(md_file, output_file)
    

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Convert DOCX to Markdown")
    parser.add_argument("--mode", type=str, choices=["docx2md", "md2docx"], help="Conversion mode",required=True)
    target_folder= Path(__file__).parent / "data/input/deep_work"
    args=parser.parse_args()
    if args.mode=="docx2md":
        convert_all_docx_files_to_markdown_in_folder(target_folder)
    elif args.mode=="md2docx":
        convert_all_markdown_files_to_docx_in_folder(target_folder)
    print(f"{target_folder} の変換が完了しました。")