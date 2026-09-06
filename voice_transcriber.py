import os
import tempfile

from faster_whisper import WhisperModel


MODEL_SIZE = "base"


_model = None


def get_whisper_model():

    global _model

    if _model is None:

        print("Loading Whisper model...")

        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

        print("Whisper model loaded.")

    return _model


def transcribe_audio(
    audio_bytes,
    audio_format="wav"
):

    if not audio_bytes:

        return "❌ Audio data is empty."

    temp_path = None

    try:

        suffix = "." + audio_format

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            temp_path = temp_file.name

        model = get_whisper_model()

        segments, info = model.transcribe(
            temp_path,
            beam_size=5
        )

        text_parts = []

        for segment in segments:

            text_parts.append(
                segment.text
            )

        text = " ".join(
            text_parts
        ).strip()

        if not text:

            return (
                "❌ I could not understand "
                "the audio."
            )

        return text

    except Exception as e:

        return (
            "❌ Voice transcription failed:\n"
            + str(e)
        )

    finally:

        if temp_path and os.path.exists(
            temp_path
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


if __name__ == "__main__":

    print(
        "Voice Transcriber loaded successfully."
    )