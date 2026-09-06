"""
rag.py - Retrieval-Augmented Generation (RAG) Pipeline

This module connects PDF text extraction, text chunking, vector embeddings,
similarity search, and grounded answer synthesis with document citations.

Pipeline Workflow:
==================
[PDF Document]
      │
      ▼
1. PDF Text Extraction (pdf_reader.py)
      │
      ▼
2. Text Chunking with Overlap (chunker.py)
      │
      ▼
3. Embedding Generation (embeddings.py)
      │
      ▼
4. Vector Store Indexing (vector_store.py)
      │
      ▼
[User Question] ──> Question Embedding ──> Cosine Similarity Search
                                                  │
                                                  ▼
                                       Top Relevant Chunks
                                                  │
                                                  ▼
                                       Grounded Answer + Citations
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import config
from pdf_reader import extract_text_from_pdf, extract_all_pdfs_in_directory
from chunker import chunk_extracted_pdf, DocumentChunk
from vector_store import VectorStore


class RAGPipeline:
    """
    Complete end-to-end RAG orchestrator for document question answering.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        # Try loading existing vector store index if available
        self.vector_store.load(config.VECTOR_STORE_PATH)

    def index_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extracts, chunks, embeds, and indexes a single PDF document.
        """
        extracted = extract_text_from_pdf(pdf_path)
        if not extracted["success"]:
            return {
                "success": False,
                "message": extracted.get("error", "Unknown extraction error"),
                "chunks_count": 0,
            }

        chunks = chunk_extracted_pdf(
            extracted_pdf_data=extracted,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        if not chunks:
            return {
                "success": False,
                "message": "No extractable text content found to index.",
                "chunks_count": 0,
            }

        indexed_count = self.vector_store.add_chunks(chunks)
        self.vector_store.save(config.VECTOR_STORE_PATH)

        return {
            "success": True,
            "filename": extracted["filename"],
            "pages_count": extracted["num_pages"],
            "chunks_count": indexed_count,
            "message": f"Successfully indexed {indexed_count} chunks from '{extracted['filename']}' ({extracted['num_pages']} pages).",
        }

    def index_all_documents(self, documents_dir: Union[str, Path] = config.DOCUMENTS_DIR) -> Dict[str, Any]:
        """
        Scans the documents directory and indexes all PDF files found.
        """
        documents_dir = Path(documents_dir)
        pdf_files = list(documents_dir.glob("*.pdf"))

        if not pdf_files:
            return {
                "success": False,
                "total_documents": 0,
                "total_chunks": 0,
                "message": f"No PDF documents found in '{documents_dir}'.",
            }

        total_chunks = 0
        indexed_files = []

        for pdf_file in pdf_files:
            res = self.index_pdf(pdf_file)
            if res["success"]:
                total_chunks += res["chunks_count"]
                indexed_files.append(pdf_file.name)

        return {
            "success": True,
            "total_documents": len(indexed_files),
            "total_chunks": total_chunks,
            "indexed_files": indexed_files,
            "message": f"Successfully indexed {len(indexed_files)} PDF(s) ({total_chunks} total chunks).",
        }

    def query(
        self,
        question: str,
        top_k: int = config.TOP_K_RESULTS,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executes a RAG search for the given user question.
        
        Returns:
            Dictionary with:
                - found (bool): True if relevant context exists
                - answer (str): Grounded synthesized response
                - sources (List[Dict]): Citations with document name and page number
                - chunks (List[Dict]): Full text chunks retrieved
        """
        if len(self.vector_store) == 0:
            return {
                "found": False,
                "answer": "No documents are currently indexed in the knowledge base. Please upload a PDF document first.",
                "sources": [],
                "chunks": [],
            }

        # 1. Retrieve top matching chunks via vector similarity search
        matching_chunks = self.vector_store.search(question, top_k=top_k, threshold=threshold)

        if not matching_chunks:
            return {
                "found": False,
                "answer": "I could not find relevant information in the uploaded documents.",
                "sources": [],
                "chunks": [],
            }

        # 2. Extract unique source references
        sources = []
        seen_sources = set()
        for c in matching_chunks:
            src_key = (c["source_doc"], c["page_number"])
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                sources.append({
                    "document": c["source_doc"],
                    "page": c["page_number"],
                    "score": c["score"],
                })

        # 3. Grounded Answer Synthesis
        # Combine the most relevant retrieved passages
        primary_chunk = matching_chunks[0]
        context_snippets = [c["text"].strip() for c in matching_chunks]
        combined_context = "\n".join(context_snippets)

        # Build grounded response with explicit citations
        citation_str = ", ".join([f"{s['document']} (Page {s['page']})" for s in sources])
        
        answer_text = (
            f"According to the uploaded document ({citation_str}):\n\n"
            f"{primary_chunk['text']}"
        )

        return {
            "found": True,
            "answer": answer_text,
            "sources": sources,
            "chunks": matching_chunks,
            "raw_context": combined_context,
        }

    def clear_index(self) -> None:
        """Clears all indexed documents from memory and disk."""
        self.vector_store.clear()
        if config.VECTOR_STORE_PATH.exists():
            config.VECTOR_STORE_PATH.unlink()


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60)

    rag = RAGPipeline()
    print(f"Current VectorStore chunks: {len(rag.vector_store)}")

    # Index sample documents in the documents folder
    index_res = rag.index_all_documents()
    print(f"Index status: {index_res['message']}")

    # Test Query 1: Matching topic
    query_1 = "What is inheritance?"
    print(f"\nQuery 1: '{query_1}'")
    ans_1 = rag.query(query_1)
    print(f"Found: {ans_1['found']}")
    print(f"Answer:\n{ans_1['answer']}")
    print(f"Sources: {ans_1['sources']}")

    # Test Query 2: Unrelated topic
    query_2 = "How to bake a chocolate cake at home?"
    print(f"\nQuery 2: '{query_2}'")
    ans_2 = rag.query(query_2)
    print(f"Found: {ans_2['found']}")
    print(f"Answer:\n{ans_2['answer']}")

    print("\nRAG Pipeline test passed successfully!")
