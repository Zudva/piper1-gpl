#!/usr/bin/env python3
"""Генерация аудио из тестовых фраз с использованием ONNX модели"""

import sys
import json
import wave
import struct
from pathlib import Path

try:
    import onnxruntime as ort
    import numpy as np
except ImportError:
    print("Установка зависимостей...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime", "numpy"])
    import onnxruntime as ort
    import numpy as np

# Добавляем путь к piper
sys.path.insert(0, "/workspace/piper1-gpl/src")

from piper.phonemize_espeak import EspeakPhonemizer

def phonemize_text(text: str, voice: str = "ru", phoneme_id_map: dict = None) -> list[int]:
    """Преобразование текста в ID фонем"""
    if phoneme_id_map is None:
        raise ValueError("phoneme_id_map required")
    
    phonemizer = EspeakPhonemizer()
    phonemes_list = phonemizer.phonemize(voice, text)
    
    # Объединяем все фонемы в одну последовательность
    phonemes = []
    for sent_phonemes in phonemes_list:
        phonemes.extend(sent_phonemes)
    
    # Преобразуем фонемы в индексы используя phoneme_id_map из конфига
    phoneme_ids = []
    for phoneme in phonemes:
        if phoneme in phoneme_id_map:
            phoneme_ids.extend(phoneme_id_map[phoneme])
        else:
            # Неизвестная фонема - используем pad (0)
            print(f"⚠️ Неизвестная фонема: '{phoneme}' (код: {ord(phoneme[0]) if phoneme else 'empty'})")
            phoneme_ids.append(0)
    
    return phoneme_ids

def generate_audio(text: str, model_path: str, config_path: str, output_path: str):
    """Генерация аудио из текста"""
    
    print(f"📝 Текст: {text}")
    print(f"🎵 Модель: {model_path}")
    
    # Загрузка конфигурации
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Загрузка ONNX модели
    print("⏳ Загрузка модели...")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    # Преобразование текста в фонемы
    print("🔤 Преобразование текста...")
    phoneme_id_map = config.get("phoneme_id_map", {})
    phoneme_ids = phonemize_text(text, voice="ru", phoneme_id_map=phoneme_id_map)
    phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    phoneme_ids_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
    
    # Параметры генерации
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)  # noise, length, noise_w
    sid = None
    
    # Генерация
    print("🎤 Генерация аудио...")
    inputs = {
        'input': phoneme_ids_array,
        'input_lengths': phoneme_ids_lengths,
        'scales': scales
    }
    
    if config.get("num_speakers", 1) > 1:
        inputs['sid'] = np.array([0], dtype=np.int64)
    
    audio = session.run(None, inputs)[0].squeeze()
    
    # Сохранение WAV
    print(f"💾 Сохранение: {output_path}")
    sample_rate = config.get("audio", {}).get("sampling_rate", 22050)
    
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Нормализация и конвертация в int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    
    duration = len(audio) / sample_rate
    print(f"✅ Готово! Длительность: {duration:.2f} сек")
    print()

if __name__ == "__main__":
    # Пути
    model_path = Path("/workspace/piper1-gpl/felix_mirage_epoch749.onnx")
    config_path = Path("/workspace/piper1-gpl/felix_mirage_epoch749.onnx.json")
    phrases_file = Path("/workspace/piper1-gpl/test_phrases_ru.txt")
    output_dir = Path("/workspace/piper1-gpl/test_audio_output")
    
    # Создание папки для выходных файлов
    output_dir.mkdir(exist_ok=True)
    
    print("🎙️  ГЕНЕРАЦИЯ ТЕСТОВЫХ АУДИОФАЙЛОВ")
    print("=" * 70)
    print()
    
    # Проверка наличия модели и конфига
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"❌ Конфиг не найден: {config_path}")
        sys.exit(1)
    
    # Чтение фраз
    if phrases_file.exists():
        with open(phrases_file, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
    else:
        phrases = [
            "Привет! Меня зовут Феликс, и я говорю по-русски.",
            "Это тестовая фраза для проверки качества синтеза речи.",
            "Искусственный интеллект развивается невероятными темпами."
        ]
    
    # Генерация аудио для каждой фразы
    for i, phrase in enumerate(phrases, 1):
        output_file = output_dir / f"test_{i:02d}.wav"
        try:
            generate_audio(phrase, model_path, config_path, output_file)
        except Exception as e:
            print(f"❌ Ошибка при генерации фразы {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 70)
    print(f"🎉 Генерация завершена! Файлы сохранены в: {output_dir}")
    print(f"📁 Всего файлов: {len(list(output_dir.glob('*.wav')))}")
