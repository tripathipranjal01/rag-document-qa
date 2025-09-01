from setuptools import setup, find_packages

setup(
    name="rag-document-qa",
    version="1.0.0",
    description="RAG Document Q&A System",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "openai",
        "python-dotenv",
        "PyPDF2",
        "python-docx",
        "tiktoken",
        "numpy",
        "Pillow",
        "pytesseract",
        "easyocr",
        "httpx",
    ],
    python_requires=">=3.11,<3.12",
)
