import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

def convert_pdfs_to_images(pdf_dir, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    
    for pdf_file in os.listdir(pdf_dir):
        if not pdf_file.endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(pdf_dir, pdf_file)
        doc_name = os.path.splitext(pdf_file)[0]
        doc_output_dir = os.path.join(output_dir, doc_name)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_path = os.path.join(doc_output_dir, f"{page_num+1:03d}.jpg")
            img.save(img_path, "JPEG", quality=95)
            
        print(f"Converted {pdf_file} - {len(doc)} pages")

if __name__ == "__main__":
    convert_pdfs_to_images("data/raw", "data/processed/images")
