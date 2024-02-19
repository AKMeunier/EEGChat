import os
import time
from typing import Union
import warnings

import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from gtts import gTTS
#from nemo.collections.nlp.models import PunctuationCapitalizationModel

import speech_recognition as sr
import openai

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../api_keys/google.json'

DEPRECATED_TRANSCRIPTION_MESSAGE = "The function you are trying to use is deprecated. Please reffer to " +\
                                   "`util.transcription` for audio transcription needs."

def record(wav_file, duration, device=None, fs=16000):
    """
    Record audio and save it as a wav file.
    :param wav_file: str; Location and name of .wav file
    :param duration: float; How long to record (in seconds)
    :param device: None or int; index of recording device, default device is used when None
    :param fs: sampling rate with which to record
    :return:
    """
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device, dtype=np.int16)
    sd.wait()
    wavfile.write(filename=wav_file, rate=fs, data=audio)


def mp3_to_wav(file_in, file_out, fs=16000):
    """
    Transform a mp3 file to a wav file
    :param file_in: str; file name of mp3
    :param file_out: str; file name o wav
    :param fs: sampling rate for wav
    :return:
    """
    sounds = AudioSegment.from_mp3(file_in)
    sounds = sounds.set_frame_rate(fs)
    sounds.export(file_out, format="wav")


def text_to_wav(text, wav_file, language='en'):
    """
    Creates audio from text (with google text-to-speech) and saves it to a wav file.
    Creates a temporary mp3 file which is deleted.
    :param text: str; Text to be converted
    :param wav_file: str; name and location of wav file which will be created
    :param language: str or language; Language used by google TTS (e.g. 'de' for German or 'en' for English)
    :return:
    """
    if language == 'de':
        tld = 'com'  # change tld for dialects
    else:
        tld = 'com'
    tts = gTTS(text, lang=language, tld=tld)
    # google API creates mp3, psychopy needs wav
    temp_file = wav_file[:-4] + '_temp.mp3'
    tts.save(temp_file)

    mp3_to_wav(file_in=temp_file, file_out=wav_file, fs=16000)
    os.remove(temp_file)


def wav_to_text(wav_path, language='en') -> Union[str, None]:
    """DEPRECATED, PLEASE USE `util.transcription` for your transcription needs.

    Transcribe text from a given wav file
    :param wav_path: str; path to wav file to be transcribed
    :param language: str or language; Language used by google STT (e.g. 'de' for German or 'en' for English)
    :return:
    """
    # warnings.warn(DEPRECATED_TRANSCRIPTION_MESSAGE)

    recognizer = sr.Recognizer()
    #punctuation_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

    with sr.AudioFile(wav_path) as source:
        audio_text = recognizer.listen(source)

        try:
            transcript = recognizer.recognize_google(audio_text, language=language)
            return transcript.capitalize() #punctuation_model.add_punctuation_capitalization([transcript])[0]
        except sr.UnknownValueError:
            print("Speech not intelligible.")
            return None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            raise sr.RequestError("Could not request results from Google Speech Recognition service; {0}".format(e))


def audio_data_to_text(audio_data: sr.AudioData, language="en") -> Union[str, None]:
    """DEPRECATED, PLEASE USE `util.transcription` for your transcription needs.

    Takes an AudioData object and returns its transcription or None if the audio could not be recognized

    This function can be used when you don't want to save a wav file to disk and just need to transcribe it.
    To use it you need to create an AudioData object, which just requires the signal, the sampling frequency and the
    bit depth of the signal.

    :param audio_data: the object which will be transcribed
    :param language: the language of the provided audio_data
    :return: a string containing the transcription text
    """
    # warnings.warn(DEPRECATED_TRANSCRIPTION_MESSAGE)

    recognizer = sr.Recognizer()
    try:
        transcript = recognizer.recognize_google(audio_data, language=language)
        return transcript.capitalize()
    except sr.UnknownValueError:
        print("Speech not intelligible.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        raise sr.RequestError("Could not request results from Google Speech Recognition service; {0}".format(e))


def audio_file_to_text_whisper(audio_file, openai_key, language="en", prompt=None) -> Union[str, None]:
    """DEPRECATED, PLEASE USE `util.transcription` for your transcription needs.

    Transcribes the given audio_file using openAIs whisper API.

    :param audio_file: the path to the to be transcribed audio file. Allows wav, mp3, mpeg with a size up to 25MB. For
        more information see https://platform.openai.com/docs/guides/speech-to-text/introduction
    :param openai_key: the openai api key which will be used for the request
    :param language: the language of the provided audio_data
    :param prompt: the optional prompt which will be passed to whisper. The result of the transcription can be directed
        with this parameter. For more information see https://platform.openai.com/docs/guides/speech-to-text/prompting
    :return: a string containing the transcription text
    """
    # warnings.warn(DEPRECATED_TRANSCRIPTION_MESSAGE)

    openai.api_key = openai_key

    with open(audio_file, "rb") as f:
        # the prompt should only serve as an example of how the output should look like
        if prompt is None:
            prompt = "Bear" if language == "en" else "Bär"
        try:
            return openai.Audio.transcribe("whisper-1", f, api_key=openai_key, language=language, prompt=prompt)["text"]
        except Exception:
            return None

def play_wav(wav_file, dtype='float64'):
    data, fs = sf.read(wav_file, dtype=dtype)
    sd.play(data, fs)
    sd.wait()


if __name__ == "__main__":
    test_wav = "../data/experiments/test.wav"
    text_to_wav("hello", test_wav, language='en')
    play_wav(test_wav, dtype='float64')
    pass
