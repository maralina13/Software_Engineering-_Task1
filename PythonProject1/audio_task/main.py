from faster_whisper import WhisperModel
import os

def main():
    audio_path = os.path.join("audio_task", "data", "sample.wav")

    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, beam_size=3)
    print("Language:", info.language, "Prob:", round(info.language_probability, 3))

    text = []
    for seg in segments:
        text.append(seg.text)

    print("TRANSCRIPT:")
    print("".join(text).strip())

if __name__ == "__main__":
    main()
