"""
Generate voice samples for AlphaVox

This script generates voice samples for all the different TTS options
available in AlphaVox using the gTTS library.
"""

import os
from gtts import gTTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories for voice samples
os.makedirs('static/voice_samples/languages', exist_ok=True)
os.makedirs('static/voice_samples/regions', exist_ok=True)
os.makedirs('static/voice_samples/rates', exist_ok=True)
os.makedirs('static/voice_samples/voices', exist_ok=True)

# Language samples
languages = {
    'en': 'This is an English voice sample.',
    'fr': 'Ceci est un échantillon de voix française.',
    'de': 'Dies ist eine deutsche Sprachprobe.',
    'es': 'Esta es una muestra de voz en español.',
    'it': 'Questo è un campione di voce italiana.',
    'pt': 'Esta é uma amostra de voz em português.',
    'nl': 'Dit is een Nederlands stemvoorbeeld.',
    'ja': 'これは日本語の音声サンプルです。',
    'ko': '이것은 한국어 음성 샘플입니다.',
    'zh-CN': '这是一个中文语音示例。'
}

# Region/TLD samples (for English)
regions = {
    'com': 'This is the standard US English voice.',
    'co.uk': 'This is the British English voice.',
    'ca': 'This is the Canadian English voice.',
    'com.au': 'This is the Australian English voice.',
    'co.in': 'This is the Indian English voice.',
    'ie': 'This is the Irish English voice.',
    'co.za': 'This is the South African English voice.'
}

# Voice profiles
voices = [
    {
        'id': 'us_female',
        'text': 'Hello, this is the US Female voice.',
        'tld': 'com',
        'lang': 'en'
    },
    {
        'id': 'uk_female',
        'text': 'Hello, this is the UK Female voice.',
        'tld': 'co.uk',
        'lang': 'en'
    },
    {
        'id': 'calming',
        'text': 'Hello, this is the Calming voice for anxiety reduction.',
        'tld': 'ca',
        'lang': 'en'
    },
    {
        'id': 'casual',
        'text': 'Hello, this is the Casual voice with a relaxed conversational tone.',
        'tld': 'com.au',
        'lang': 'en' 
    },
    {
        'id': 'formal',
        'text': 'Hello, this is the Formal voice with a precise, professional tone.',
        'tld': 'co.in',
        'lang': 'en'
    },
    {
        'id': 'friendly_male',
        'text': 'Hello, this is the Friendly Male voice.',
        'tld': 'ie',
        'lang': 'en'
    },
    {
        'id': 'za_voice',
        'text': 'Hello, this is the South African voice.',
        'tld': 'co.za',
        'lang': 'en'
    },
    {
        'id': 'au_female',
        'text': 'Hello, this is the Australian Female voice.',
        'tld': 'com.au',
        'lang': 'en'
    }
]

# Generate language samples
logger.info("Generating language samples...")
for lang_code, text in languages.items():
    output_path = f'static/voice_samples/languages/lang_{lang_code.replace("-", "_")}.mp3'
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(output_path)
        logger.info(f"Generated {output_path}")
    except Exception as e:
        logger.error(f"Error generating {output_path}: {e}")

# Generate region/TLD samples
logger.info("Generating region samples...")
for tld, text in regions.items():
    output_path = f'static/voice_samples/regions/tld_{tld.replace(".", "_")}.mp3'
    try:
        tts = gTTS(text=text, lang='en', tld=tld)
        tts.save(output_path)
        logger.info(f"Generated {output_path}")
    except Exception as e:
        logger.error(f"Error generating {output_path}: {e}")

# Generate rate samples
logger.info("Generating speech rate samples...")
speech_rates = {
    'slow': 'This is a slow speech rate for clear understanding.',
    'normal': 'This is a normal speech rate for everyday conversation.'
}

for rate_name, text in speech_rates.items():
    output_path = f'static/voice_samples/rates/rate_{rate_name}.mp3'
    try:
        slow = rate_name == 'slow'
        tts = gTTS(text=text, lang='en', slow=slow)
        tts.save(output_path)
        logger.info(f"Generated {output_path}")
    except Exception as e:
        logger.error(f"Error generating {output_path}: {e}")

# Generate voice profile samples
logger.info("Generating voice profile samples...")
for voice in voices:
    output_path = f'static/voice_samples/voices/{voice["id"]}.mp3'
    try:
        tts = gTTS(text=voice['text'], lang=voice['lang'], tld=voice['tld'])
        tts.save(output_path)
        logger.info(f"Generated {output_path}")
    except Exception as e:
        logger.error(f"Error generating {output_path}: {e}")

logger.info("Voice sample generation complete.")