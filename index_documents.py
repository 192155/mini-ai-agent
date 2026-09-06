import os

from pypdf import PdfReader

from embeddings import EmbeddingModel
from vector_store import VectorStore


DOCUMENTS_FOLDER = "documents"
VECTOR_STORE_PATH = "data/all_vectors.pkl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


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


def main():

    print()
    print("======================================")
    print("       PDF INDEXING SYSTEM")
    print("======================================")
    print()

    os.makedirs(
        DOCUMENTS_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    pdf_files = [
        file
        for file in os.listdir(DOCUMENTS_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        print(
            "❌ No PDF files found in documents folder."
        )

        return

    print(
        f"📚 Found {len(pdf_files)} PDF file(s)"
    )

    print()

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    total_chunks = 0

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_FOLDER,
            pdf_file
        )

        print(
            f"📖 Processing: {pdf_file}"
        )

        try:

            text = extract_text_from_pdf(
                pdf_path
            )

            print(
                f"   Characters: {len(text)}"
            )

            chunks = create_chunks(
                text
            )

            print(
                f"   Chunks: {len(chunks)}"
            )

            if not chunks:

                print(
                    "   ⚠️ No readable text found."
                )

                continue

            documents = []

            for chunk in chunks:

                documents.append(
                    {
                        "text": chunk,
                        "source": pdf_file
                    }
                )

            texts_for_embedding = [
                item["text"]
                for item in documents
            ]

            embeddings = []

            for index, chunk in enumerate(
                texts_for_embedding,
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

                if index % 10 == 0:

                    print(
                        f"   Embeddings: "
                        f"{index}/{len(chunks)}"
                    )

            vector_store.add_documents(
                documents,
                embeddings
            )

            total_chunks += len(
                documents
            )

            print(
                f"   ✅ {pdf_file} indexed."
            )

            print()

        except Exception as e:

            print(
                f"   ❌ Error processing "
                f"{pdf_file}:"
            )

            print(
                f"   {e}"
            )

            print()

    vector_store.save(
        VECTOR_STORE_PATH
    )

    print(
        "======================================"
    )

    print(
        "✅ INDEXING COMPLETE"
    )

    print(
        f"📚 PDFs: {len(pdf_files)}"
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


if __name__ == "__main__":

    main()