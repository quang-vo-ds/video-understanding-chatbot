import datetime
import whisper
import numpy as np

class AudioTranscriber:
    def __init__(self):
        self.model = whisper.load_model("small", device="cuda")

    def invoke(self, audio_path: str, delimiter: list = ('.', '!', '?')):
        result = self.model.transcribe(audio_path, word_timestamps=True)
        segments = result.get("segments", [])

        sentences = []
        current_sentence = ""
        start_time = None
        end_time = None

        for segment in segments:
            for word_info in segment.get("words", []):
                word = word_info["word"]
                word_start = str(datetime.timedelta(seconds=np.round(word_info["start"])))
                word_end = str(datetime.timedelta(seconds=np.round(word_info["end"])))

                if start_time is None:
                    start_time = word_start

                current_sentence += word
                end_time = word_end

                if word.strip().endswith(delimiter):
                    sentences.append({
                        "sentence": current_sentence.strip(), 
                        "start_time": start_time, 
                        "end_time": end_time
                        })
                    current_sentence = ""
                    start_time = None
                    end_time = None

        # Append any leftover words as a final sentence
        if current_sentence.strip():
            sentences.append((current_sentence.strip(), start_time, end_time))

        return sentences