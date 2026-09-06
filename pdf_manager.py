import os
import json
import hashlib
from pathlib import Path

from pypdf import PdfReader

from embeddings import EmbeddingModel
from vector_store import VectorStore


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_FOLDER = Path("documents")

VECTOR_STORE_PATH = "data/all_vectors.pkl"
METADATA_PATH = "data/pdf_metadata.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ============================================================
# FOLDER SETUP
# ============================================================

DOCUMENTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

os.makedirs(
    "data",
    exist_ok=True
)


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path):

    file_path = Path(file_path)

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# METADATA
# ============================================================

def load_metadata():

    if not os.path.exists(
        METADATA_PATH
    ):

        return {}

    try:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

            return {}

    except Exception as e:

        print(
            "Metadata loading error:",
            e
        )

        return {}


def save_metadata(metadata):

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):

    pdf_path = Path(pdf_path)

    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

            if text:

                pages.append(
                    text
                )

        except Exception as e:

            print(
                f"Page {page_number} error: {e}"
            )

    return "\n".join(
        pages
    )


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    step = chunk_size - overlap

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        start += step

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

def get_embedding_model():

    print(
        "\nLoading embedding model..."
    )

    model = EmbeddingModel()

    print(
        "Embedding model loaded."
    )

    return model


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():

    vector_store = VectorStore()

    if not os.path.exists(
        VECTOR_STORE_PATH
    ):

        return vector_store

    try:

        vector_store.load(
            VECTOR_STORE_PATH
        )

    except Exception as e:

        print(
            "Vector store loading error:",
            e
        )

        vector_store = VectorStore()

    return vector_store


# ============================================================
# SAVE VECTOR STORE
# ============================================================

def save_vector_store(
    vector_store
):

    vector_store.save(
        VECTOR_STORE_PATH
    )


# ============================================================
# CHECK PDF STATUS
# ============================================================

def get_pdf_status(
    pdf_path
):

    pdf_path = Path(
        pdf_path
    )

    if not pdf_path.exists():

        return {
            "status": "missing"
        }

    file_hash = calculate_file_hash(
        pdf_path
    )

    metadata = load_metadata()

    saved_data = metadata.get(
        pdf_path.name
    )

    if not saved_data:

        return {
            "status": "not_indexed",
            "hash": file_hash
        }

    if saved_data.get(
        "hash"
    ) == file_hash:

        return {
            "status": "indexed",
            "hash": file_hash,
            "chunks": saved_data.get(
                "chunks",
                0
            )
        }

    return {
        "status": "changed",
        "hash": file_hash
    }


# ============================================================
# CHECK IF PDF INDEXED
# ============================================================

def pdf_already_indexed(
    pdf_path
):

    status = get_pdf_status(
        pdf_path
    )

    return (
        status["status"]
        == "indexed"
    )


# ============================================================
# REMOVE PDF FROM VECTOR STORE
# ============================================================

def remove_pdf_from_vector_store(
    pdf_name
):

    vector_store = load_vector_store()

    if not vector_store.documents:

        return False

    source_text = (
        f"[SOURCE PDF: {pdf_name}]"
    )

    remaining_documents = []

    remaining_embeddings = []

    if vector_store.embeddings is None:

        return False

    for index, document in enumerate(
        vector_store.documents
    ):

        if document.startswith(
            source_text
        ):

            continue

        remaining_documents.append(
            document
        )

        remaining_embeddings.append(
            vector_store.embeddings[
                index
            ]
        )

    removed_count = (
        len(vector_store.documents)
        - len(remaining_documents)
    )

    vector_store.documents = (
        remaining_documents
    )

    if remaining_embeddings:

        import numpy as np

        vector_store.embeddings = (
            np.array(
                remaining_embeddings
            )
        )

    else:

        vector_store.embeddings = None

    if removed_count > 0:

        save_vector_store(
            vector_store
        )

    metadata = load_metadata()

    if pdf_name in metadata:

        del metadata[
            pdf_name
        ]

        save_metadata(
            metadata
        )

    print(
        f"Removed {removed_count} chunks "
        f"for {pdf_name}"
    )

    return removed_count > 0


# ============================================================
# INDEX SINGLE PDF
# ============================================================

def index_single_pdf(
    pdf_path,
    embedding_model=None,
    force=False
):

    pdf_path = Path(
        pdf_path
    )

    print(
        "\n========================================"
    )

    print(
        "             SMART PDF INDEXER"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not pdf_path.exists():

        print(
            "PDF file not found."
        )

        return False

    if pdf_path.suffix.lower() != ".pdf":

        print(
            "File is not a PDF."
        )

        return False

    print(
        f"\nPDF: {pdf_path.name}"
    )

    # --------------------------------------------------------
    # Check status
    # --------------------------------------------------------

    status = get_pdf_status(
        pdf_path
    )

    print(
        f"Status: {status['status']}"
    )

    # --------------------------------------------------------
    # Already indexed
    # --------------------------------------------------------

    if (
        status["status"]
        == "indexed"
        and not force
    ):

        print(
            "\nPDF is already indexed."
        )

        print(
            "Skipping duplicate indexing."
        )

        return True

    # --------------------------------------------------------
    # If changed, remove old chunks
    # --------------------------------------------------------

    if (
        status["status"]
        == "changed"
    ):

        print(
            "\nPDF has changed."
        )

        print(
            "Removing old indexed chunks..."
        )

        remove_pdf_from_vector_store(
            pdf_path.name
        )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    print(
        "\nExtracting PDF text..."
    )

    text = extract_pdf_text(
        pdf_path
    )

    print(
        f"Characters: {len(text)}"
    )

    if not text.strip():

        print(
            "No readable text found."
        )

        return False

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print(
        "\nCreating chunks..."
    )

    chunks = create_chunks(
        text
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    if not chunks:

        print(
            "No chunks created."
        )

        return False

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    if embedding_model is None:

        embedding_model = (
            get_embedding_model()
        )

    # --------------------------------------------------------
    # Load vector store
    # --------------------------------------------------------

    vector_store = (
        load_vector_store()
    )

    documents = []

    embeddings = []

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print(
        "\nCreating embeddings..."
    )

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        document = (
            f"[SOURCE PDF: {pdf_path.name}]\n\n"
            f"{chunk}"
        )

        embedding = (
            embedding_model.create_embedding(
                document
            )
        )

        documents.append(
            document
        )

        embeddings.append(
            embedding
        )

        if (
            index % 10 == 0
            or
            index == len(chunks)
        ):

            print(
                f"Embedded "
                f"{index}/{len(chunks)}"
            )

    # --------------------------------------------------------
    # Add to vector store
    # --------------------------------------------------------

    vector_store.add_documents(
        documents,
        embeddings
    )

    # --------------------------------------------------------
    # Save vector store
    # --------------------------------------------------------

    print(
        "\nSaving vector store..."
    )

    save_vector_store(
        vector_store
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    file_hash = calculate_file_hash(
        pdf_path
    )

    metadata[
        pdf_path.name
    ] = {

        "filename":
            pdf_path.name,

        "hash":
            file_hash,

        "chunks":
            len(chunks),

        "characters":
            len(text),

        "indexed":
            True
    }

    save_metadata(
        metadata
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "          PDF INDEXED SUCCESSFULLY"
    )

    print(
        "========================================"
    )

    print(
        f"PDF: {pdf_path.name}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        "Hash saved."
    )

    print(
        "Metadata saved."
    )

    return True


# ============================================================
# INDEX ALL PDFs
# ============================================================

def index_all_pdfs(
    force=False
):

    print(
        "\n========================================"
    )

    print(
        "          BUILDING KNOWLEDGE BASE"
    )

    print(
        "========================================"
    )

    pdf_files = [

        file

        for file in DOCUMENTS_FOLDER.iterdir()

        if (
            file.is_file()
            and
            file.suffix.lower() == ".pdf"
        )
    ]

    if not pdf_files:

        print(
            "\nNo PDF files found."
        )

        return False

    print(
        f"\nFound {len(pdf_files)} PDF(s)."
    )

    embedding_model = (
        get_embedding_model()
    )

    success_count = 0

    for pdf_file in pdf_files:

        print(
            "\n----------------------------------------"
        )

        print(
            f"Processing: {pdf_file.name}"
        )

        result = index_single_pdf(

            pdf_file,

            embedding_model=embedding_model,

            force=force
        )

        if result:

            success_count += 1

    print(
        "\n========================================"
    )

    print(
        "          KNOWLEDGE BASE READY"
    )

    print(
        "========================================"
    )

    print(
        f"Successful PDFs: "
        f"{success_count}/{len(pdf_files)}"
    )

    return success_count > 0


# ============================================================
# LIST PDFs
# ============================================================

def list_pdfs():

    pdf_files = [

        file

        for file in DOCUMENTS_FOLDER.iterdir()

        if (
            file.is_file()
            and
            file.suffix.lower() == ".pdf"
        )
    ]

    result = []

    for pdf_file in pdf_files:

        status = get_pdf_status(
            pdf_file
        )

        result.append(
            {
                "filename":
                    pdf_file.name,

                "status":
                    status.get(
                        "status",
                        "unknown"
                    ),

                "chunks":
                    status.get(
                        "chunks",
                        0
                    )
            }
        )

    return result


# ============================================================
# DELETE PDF
# ============================================================

def delete_pdf(
    pdf_name
):

    pdf_path = (
        DOCUMENTS_FOLDER
        / pdf_name
    )

    if not pdf_path.exists():

        print(
            "PDF not found."
        )

        return False

    print(
        f"\nDeleting PDF: {pdf_name}"
    )

    # Remove from vector store
    remove_pdf_from_vector_store(
        pdf_name
    )

    # Remove actual PDF
    try:

        pdf_path.unlink()

    except Exception as e:

        print(
            "PDF deletion error:",
            e
        )

        return False

    print(
        "PDF deleted successfully."
    )

    return True


# ============================================================
# REBUILD KNOWLEDGE BASE
# ============================================================

def rebuild_knowledge_base():

    print(
        "\n========================================"
    )

    print(
        "        REBUILDING KNOWLEDGE BASE"
    )

    print(
        "========================================"
    )

    # Delete old vector store
    if os.path.exists(
        VECTOR_STORE_PATH
    ):

        os.remove(
            VECTOR_STORE_PATH
        )

        print(
            "Old vector store removed."
        )

    # Delete metadata
    if os.path.exists(
        METADATA_PATH
    ):

        os.remove(
            METADATA_PATH
        )

        print(
            "Old metadata removed."
        )

    # Re-index everything
    result = index_all_pdfs(
        force=True
    )

    return result


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "             PDF MANAGER"
    )

    print(
        "========================================"
    )

    print(
        "\nPDFs in documents folder:\n"
    )

    pdfs = list_pdfs()

    if not pdfs:

        print(
            "No PDFs found."
        )

    else:

        for pdf in pdfs:

            print(
                f"📄 {pdf['filename']}"
            )

            print(
                f"   Status: {pdf['status']}"
            )

            print(
                f"   Chunks: {pdf['chunks']}"
            )

    print(
        "\nPDF Manager loaded successfully."
    )