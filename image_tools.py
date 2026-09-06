import io
from PIL import Image


def load_image(uploaded_file):
    """
    Streamlit uploaded file ko PIL Image mein convert karta hai.
    """

    try:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        return image

    except Exception as e:

        raise ValueError(
            f"Image load nahi ho payi: {e}"
        )


def get_image_info(image):
    """
    Image ki basic information return karta hai.
    """

    return {
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height
    }


def prepare_image(image):
    """
    Gemini ko bhejne se pehle image ko
    compatible format mein prepare karta hai.
    """

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    return image