import os

from pypdf import PdfReader

from embeddings import EmbeddingModel
from vector_store import VectorStore


# ==========================================
# CONFIGURATION
# ==========================================

DOCUMENTS_FOLDER = "documents"

VECTOR_STORE_PATH = "data/all_vectors.pkl"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 150


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:

            pages.append(text)

    return "\n".join(pages)


# ==========================================
# TEXT CHUNKING
# ==========================================

def create_chunks(text):

    text = text.strip()

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# MAIN INDEXING FUNCTION
# ==========================================

def main():

    print()

    print("======================================")

    print("       PDF INDEXING SYSTEM")

    print("======================================")

    print()

    # --------------------------------------
    # Create required folders
    # --------------------------------------

    os.makedirs(
        DOCUMENTS_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    # --------------------------------------
    # Find PDF files
    # --------------------------------------

    pdf_files = [

        file

        for file in os.listdir(
            DOCUMENTS_FOLDER
        )

        if file.lower().endswith(".pdf")

    ]

    if not pdf_files:

        print(
            "❌ No PDF files found in documents folder."
        )

        print()

        print(
            "Please put your PDF files inside:"
        )

        print(
            f"   {DOCUMENTS_FOLDER}"
        )

        return

    print(
        f"📚 Found {len(pdf_files)} PDF file(s)"
    )

    print()

    # --------------------------------------
    # Load Gemini Embedding Model
    # --------------------------------------

    print(
        "🤖 Loading Gemini Embedding Model..."
    )

    embedding_model = EmbeddingModel()

    print()

    # --------------------------------------
    # Create Vector Store
    # --------------------------------------

    print(
        "🗄️ Creating vector store..."
    )

    vector_store = VectorStore()

    print()

    total_chunks = 0

    successful_pdfs = 0

    failed_pdfs = 0

    # ======================================
    # PROCESS EACH PDF
    # ======================================

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_FOLDER,
            pdf_file
        )

        print(
            "--------------------------------------"
        )

        print(
            f"📖 Processing: {pdf_file}"
        )

        print(
            "--------------------------------------"
        )

        try:

            # --------------------------------
            # Extract PDF text
            # --------------------------------

            text = extract_text_from_pdf(
                pdf_path
            )

            print(
                f"   📝 Characters: {len(text)}"
            )

            # --------------------------------
            # Create chunks
            # --------------------------------

            chunks = create_chunks(
                text
            )

            print(
                f"   🧩 Chunks: {len(chunks)}"
            )

            if not chunks:

                print(
                    "   ⚠️ No readable text found."
                )

                failed_pdfs += 1

                print()

                continue

            # --------------------------------
            # Create document objects
            # --------------------------------

            documents = []

            for chunk in chunks:

                documents.append(
                    {
                        "text": chunk,
                        "source": pdf_file
                    }
                )

            # --------------------------------
            # Create embeddings
            # --------------------------------

            embeddings = []

            print(
                "   🧠 Creating embeddings..."
            )

            for index, chunk in enumerate(
                chunks,
                start=1
            ):

                embedding = (
                    embedding_model.create_embedding(
                        chunk
                    )
                )

                embeddings.append(
                    embedding
                )

                print(
                    f"   Embedding "
                    f"{index}/{len(chunks)}"
                )

            # --------------------------------
            # Add to vector store
            # --------------------------------

            vector_store.add_documents(
                documents,
                embeddings
            )

            total_chunks += len(
                documents
            )

            successful_pdfs += 1

            print()

            print(
                f"   ✅ {pdf_file} indexed successfully."
            )

            print()

        except Exception as e:

            failed_pdfs += 1

            print()

            print(
                f"   ❌ Error processing {pdf_file}"
            )

            print(
                f"   Error: {e}"
            )

            print()

    # ======================================
    # SAVE VECTOR STORE
    # ======================================

    print(
        "======================================"
    )

    print(
        "💾 Saving vector store..."
    )

    vector_store.save(
        VECTOR_STORE_PATH
    )

    print()

    # ======================================
    # FINAL RESULT
    # ======================================

    print(
        "======================================"
    )

    print(
        "✅ INDEXING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"📚 Total PDFs found: {len(pdf_files)}"
    )

    print(
        f"✅ Successful PDFs: {successful_pdfs}"
    )

    print(
        f"❌ Failed PDFs: {failed_pdfs}"
    )

    print(
        f"🧩 Total chunks: {total_chunks}"
    )

    print(
        f"💾 Saved to: {VECTOR_STORE_PATH}"
    )

    print(
        "======================================"
    )

    print()

    print(
        "🎉 Your PDF knowledge base is ready!"
    )

    print()


# ==========================================
# PROGRAM START
# ==========================================

if __name__ == "__main__":

    main()