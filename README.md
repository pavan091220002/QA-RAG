# QA PDF RAG System

## Overview

The **QA RAG System** is a Python-based application for processing and querying PDF documents, optimized for clinical and mental health content. It uses Retrieval-Augmented Generation (RAG) to extract text and images, perform semantic chunking, store content in a FAISS vector store, and generate accurate query responses using the google/flan-t5-base model. This system enhances document analysis by combining text and image data for comprehensive query answering.

## Key Features

- **PDF Content Extraction**: Extracts text and images from PDFs using PyMuPDF (fitz).
- **Semantic Chunking**: Splits text into contextually relevant chunks using SentenceTransformers for improved retrieval.
- **Vector Store**: Utilizes FAISS for efficient similarity search of text and image captions.
- **Image Captioning**: Generates placeholder captions for images, with potential for advanced vision-language model integration.
- **AI-Powered Query Processing**: Leverages google/flan-t5-base to provide precise answers based on retrieved content.
- **Configurable Processing**: Supports customizable chunk sizes and similarity thresholds for flexible document analysis.
- **Processing Metrics**: Tracks text and image counts for transparency in document processing.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `numpy`
  - `PyMuPDF` (fitz)
  - `Pillow` (PIL)
  - `sentence-transformers`
  - `faiss-cpu`
  - `transformers`
  - `nltk`
  - `torch`

Install dependencies using:
```bash
pip install numpy PyMuPDF Pillow sentence-transformers faiss-cpu transformers nltk torch
```

## Usage

1. **Prepare the PDF**: Place the PDF file (e.g., `9241544228_eng.pdf`) in the project directory.
2. **Run the Notebook**: Execute the Jupyter Notebook (`Untitled75.ipynb`) to:
   - Extract text and images from the PDF.
   - Chunk text semantically for context preservation.
   - Generate captions for extracted images.
   - Build a FAISS vector store for content retrieval.
   - Answer user queries using the RAG pipeline.
3. **Query the System**: Use the `query_rag` function to ask questions. Example:
   ```python
   pdf_path = "9241544228_eng.pdf"
   vector_store, doc_info = process_document(pdf_path)
   query = "What are the diagnostic criteria for OCD?"
   result = query_rag(query, vector_store)
   print(result['response'])
   ```

## Code Structure

- **Imports**: Libraries for PDF processing, embeddings, vector search, and language modeling.
- **Key Functions**:
  - `extract_content_from_pdf`: Extracts text and images, saving images to a specified directory.
  - `semantic_chunking`: Splits text into semantic chunks with configurable overlap.
  - `generate_image_caption`: Provides basic image captions (placeholder implementation).
  - `process_images`: Processes images and attaches captions.
  - `VectorStore`: Manages FAISS index for similarity search.
  - `process_document`: Coordinates PDF processing and vector store creation.
  - `query_rag`: Retrieves relevant content and generates responses using a transformer model.
- **Main Execution**: Processes a sample PDF and demonstrates querying for OCD diagnostic criteria.

## Limitations

- **Image Captioning**: Limited to placeholder captions; lacks detailed image analysis.
- **PDF Compatibility**: May encounter issues with corrupted or protected PDFs.
- **Response Quality**: Relies on extracted content quality and the google/flan-t5-base model's capabilities.
- **Single PDF Processing**: Currently supports one PDF at a time, limiting scalability.

## Future Improvements

- **Enhanced Image Captioning**: Integrate vision-language models like CLIP for detailed image descriptions.
- **Multi-PDF Support**: Enable processing of multiple PDFs concurrently.
- **Robust Error Handling**: Improve support for diverse PDF formats and error conditions.
- **Scalability**: Optimize FAISS and processing pipeline for large document collections.

## Example

To process and query a PDF:
```python
pdf_path = "9241544228_eng.pdf"
vector_store, doc_info = process_document(pdf_path)
query = "What are the diagnostic criteria for OCD?"
result = query_rag(query, vector_store)
print(result['response'])
```

## Notes

- **NLTK Data**: Ensure `punkt` and `punkt_tab` are downloaded for sentence tokenization:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('punkt_tab')
  ```
- **Model Efficiency**: Uses `google/flan-t5-base` with FP16 precision for optimized inference.
- **Customization**: Adjust `chunk_size` and `percentile_threshold` in `process_document` for specific use cases.
- **Domain Flexibility**: Optimized for clinical and mental health documents but adaptable to other domains.
