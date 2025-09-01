# Document processing service

import PyPDF2
import io
from typing import List, Dict, Any
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import easyocr
import tiktoken

class DocumentProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_content: bytes) -> str:
        try:
            docx_file = io.BytesIO(docx_content)
            doc = DocxDocument(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def extract_text_from_image(self, image_content: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_content))
            
            # Try OCR with pytesseract first
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    return text
            except Exception:
                pass
            
            # Fallback to EasyOCR
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image_content)
            text = " ".join([result[1] for result in results])
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        tokens = self.encoding.encode(text)
        
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            i += chunk_size - overlap
        
        return chunks
