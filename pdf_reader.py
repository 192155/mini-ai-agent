from pypdf import PdfReader


def extract_pdf_text(pdf_path):
    """
    Read a PDF and return all extracted text.
    """

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Test the PDF reader
if __name__ == "__main__":

    pdf_path = "documents/Java Notes.pdf.pdf"

    text = extract_pdf_text(pdf_path)

    print("Number of pages:", len(PdfReader(pdf_path).pages))

    print("\n========== PDF TEXT ==========\n")

    print(text[:5000])