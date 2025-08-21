import os
import PyPDF2
from docx import Document as DocxDocument
from typing import List, Dict, Any

class DocumentProcessor:
	def __init__(self) -> None:
		self.supported_types = {
			'.pdf': self._process_pdf,
			'.docx': self._process_docx,
			'.txt': self._process_txt,
		}

	def process_document(self, file_path: str) -> Dict[str, Any]:
		file_ext = os.path.splitext(file_path)[1].lower()
		if file_ext not in self.supported_types:
			raise ValueError(f"Unsupported file type: {file_ext}")
		return self.supported_types[file_ext](file_path)

	def _process_pdf(self, file_path: str) -> Dict[str, Any]:
		text_content = []
		with open(file_path, 'rb') as f:
			reader = PyPDF2.PdfReader(f)
			for page_num, page in enumerate(reader.pages, start=1):
				text = page.extract_text() or ""
				text = text.strip()
				if text:
					text_content.append({'page': page_num, 'text': text, 'word_count': len(text.split())})
		return {
			'type': 'pdf',
			'page_count': len(text_content),
			'content': text_content,
			'total_words': sum(p['word_count'] for p in text_content),
		}

	def _process_docx(self, file_path: str) -> Dict[str, Any]:
		doc = DocxDocument(file_path)
		text_content = []
		for para in doc.paragraphs:
			text = para.text.strip()
			if text:
				text_content.append({'page': 1, 'text': text, 'word_count': len(text.split())})
		return {
			'type': 'docx',
			'page_count': 1,
			'content': text_content,
			'total_words': sum(p['word_count'] for p in text_content),
		}

	def _process_txt(self, file_path: str) -> Dict[str, Any]:
		with open(file_path, 'r', encoding='utf-8') as f:
			text = f.read().strip()
		return {
			'type': 'txt',
			'page_count': 1,
			'content': [{'page': 1, 'text': text, 'word_count': len(text.split())}],
			'total_words': len(text.split()),
		}

	def chunk_text(self, content: List[Dict[str, Any]], chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
		chunks: List[Dict[str, Any]] = []
		for page_data in content:
			words = page_data['text'].split()
			start = 0
			while start < len(words):
				end = min(start + chunk_size, len(words))
				chunk_words = words[start:end]
				chunk_text = ' '.join(chunk_words)
				if chunk_text:
					chunks.append({
						'text': chunk_text,
						'page': page_data['page'],
						'chunk_index': len(chunks),
						'word_count': len(chunk_words),
						'start_word': start,
						'end_word': end,
					})
				start = end - overlap if end - overlap > 0 else end
		return chunks 
