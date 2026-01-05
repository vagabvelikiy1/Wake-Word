from src.inference import sliding_window_detect

sliding_window_detect(
    audio_path="test_audio.wav",
    model_path="wake_word_model.h5"
)
