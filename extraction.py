import pymupdf4llm
from pathlib import Path
import pymupdf

def extract_text_as_md(input_pdf_path:Path)->str:
    text = pymupdf4llm.to_markdown(input_pdf_path)
    return text

def extract_text(input_file_path:Path,output_file_path:Path):
    doc = pymupdf.open(input_file_path)
    with open(output_file_path, "wb") as out:
        for page in doc: 
            text = page.get_text().encode("utf8") 
            out.write(text) 

def write_md(md_text:str,output_md_path:Path):
    with open(output_md_path,'w') as f:
        f.write(md_text)
        
def save_extracted_text_as_md(input_pdf_path:Path,output_md_path:Path):
    output_md_path.parent.mkdir(parents=True,exist_ok=True)
    md_text=extract_text_as_md(input_pdf_path)
    write_md(md_text,output_md_path)
    
def extract_correct_sequence_from_word_file(docx_path:Path,output_txt_path:Path):
    """翻訳したいワードファイルから、正しい順序で文章を抽出して、その文章をoutput_txt_pathに保存する。

    Args:
        docx_path (Path): _description_
        output_txt_path (Path): _description_

    Returns:
        str: _description_
    """
    pass

if __name__ == "__main__":
    input_pdf_path = Path(__file__).parent / "data" / "input_pdf" / "deep_utopia/Deep Utopia_3_wednesday.pdf"
    output_md_path = Path(__file__).parent / "data" / "output_md" / "deep_utopia/Deep Utopia_3_wednesday.md"
    save_extracted_text_as_md(input_pdf_path,output_md_path)