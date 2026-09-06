from pypdf import PdfReader


def extract_pdf_text(pdf_path):
    """
    Extract text from PDF.
    """

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def create_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into smaller overlapping chunks.
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


if __name__ == "__main__":

    pdf_path = "documents/Java Notes.pdf.pdf"

    print("Reading PDF...")

    text = extract_pdf_text(pdf_path)

    print("PDF text extracted successfully.")

    print("Total characters:", len(text))

    print("\nCreating chunks...")

    chunks = create_chunks(text)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5]):

        print("\n==============================")
        print("CHUNK", i + 1)
        print("==============================")

        print(chunk[:1000])