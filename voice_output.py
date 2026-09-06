import os
import tempfile
import subprocess


def get_available_voices():
    """
    Get available Windows Speech voices.
    """

    try:

        script = """
Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$synth.GetInstalledVoices() |
ForEach-Object {
    $_.VoiceInfo.Name
}

$synth.Dispose()
"""

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script
            ],
            capture_output=True,
            text=True,
            timeout=20
        )

        if result.returncode != 0:
            return []

        voices = []

        for line in result.stdout.splitlines():

            line = line.strip()

            if line:
                voices.append(line)

        return voices

    except Exception:

        return []


def text_to_speech(
    text,
    rate=0,
    volume=100,
    voice_name=None
):
    """
    Convert text into WAV speech using
    Windows System.Speech.
    """

    if not text:

        return False, "No text provided."

    try:

        clean_text = str(text)

        # Remove markdown formatting
        clean_text = clean_text.replace(
            "```",
            ""
        )

        clean_text = clean_text.replace(
            "**",
            ""
        )

        clean_text = clean_text.replace(
            "##",
            ""
        )

        clean_text = clean_text.replace(
            "#",
            ""
        )

        # Limit very long responses
        if len(clean_text) > 4000:

            clean_text = (
                clean_text[:4000]
                + " ..."
            )

        temp_dir = tempfile.gettempdir()

        text_file = os.path.join(
            temp_dir,
            "mini_ai_tts.txt"
        )

        wav_file = os.path.join(
            temp_dir,
            "mini_ai_agent_voice.wav"
        )

        # Save text
        with open(
            text_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                clean_text
            )

        # Escape PowerShell quotes
        safe_text_file = (
            text_file.replace(
                "'",
                "''"
            )
        )

        safe_wav_file = (
            wav_file.replace(
                "'",
                "''"
            )
        )

        voice_code = ""

        if voice_name:

            safe_voice_name = (
                voice_name.replace(
                    "'",
                    "''"
                )
            )

            voice_code = f"""
$voiceName = '{safe_voice_name}'

try {{
    $synth.SelectVoice($voiceName)
}}
catch {{
}}
"""

        powershell_script = f"""
Add-Type -AssemblyName System.Speech

$text = Get-Content -Raw -Encoding UTF8 '{safe_text_file}'

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$synth.Rate = {int(rate)}
$synth.Volume = {int(volume)}

{voice_code}

$synth.SetOutputToWaveFile('{safe_wav_file}')

$synth.Speak($text)

$synth.Dispose()
"""

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                powershell_script
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:

            return False, result.stderr

        if not os.path.exists(
            wav_file
        ):

            return False, (
                "Voice file was not created."
            )

        return True, wav_file

    except subprocess.TimeoutExpired:

        return False, (
            "Text-to-speech timed out."
        )

    except Exception as e:

        return False, str(e)