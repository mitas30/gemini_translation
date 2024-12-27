import pymupdf4llm
from pathlib import Path

def extract_text(input_pdf_path:Path)->str:
    text = pymupdf4llm.to_markdown(input_pdf_path)
    return text

def write_md(md_text:str,output_md_path:Path):
    with open(output_md_path,'w') as f:
        f.write(md_text)

if __name__ == "__main__":
    input_pdf_path = Path(__file__).parent / "data" / "input_pdf" / "part4_thought" / "ch25_Genius.pdf"
    text = extract_text(input_pdf_path)
    output_pdf_path = Path(__file__).parent / "data" / "output_md" / "part4_thought" / "ch25_Genius.md"
    output_pdf_path.parent.mkdir(exist_ok=True,parents=True)
    write_md(text,output_pdf_path)