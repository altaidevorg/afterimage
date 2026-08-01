import logging
import os
from afterimage.logging import silence_noisy_third_party_loggers


def test_silence_noisy_third_party_loggers():
    silence_noisy_third_party_loggers(logging.ERROR)
    assert logging.getLogger("google_genai").level == logging.ERROR
    assert logging.getLogger("httpx").level == logging.ERROR


def test_google_genai_warning_suppressed(capsys):
    os.environ["GOOGLE_API_KEY"] = "test_google_key"
    os.environ["GEMINI_API_KEY"] = "test_gemini_key"

    silence_noisy_third_party_loggers(logging.ERROR)

    from google import genai

    _ = genai.Client()

    captured = capsys.readouterr()
    assert "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" not in captured.err
    assert "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" not in captured.out
