#!/usr/bin/env python3
"""Генерация аудио напрямую из checkpoint (без ONNX)"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, "/workspace/piper1-gpl/src")

from piper.train.vits.lightning import VitsModel
from piper.phonemize_espeak import EspeakPhonemizer

def load_checkpoint(ckpt_path: str, config_path: str):
    """Загрузка модели из checkpoint"""
    print(f"⏳ Загрузка checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Загрузка конфига из JSON
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Создание модели из hyper_parameters
    hparams = checkpoint['hyper_parameters']
    model = VitsModel(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"✅ Модель загружена (epoch {checkpoint['epoch']})")
    return model, config

def phonemize_text(text: str, voice: str = "ru", phoneme_id_map: dict = None) -> list[int]:
    """Преобразование текста в ID фонем"""
    if phoneme_id_map is None:
        raise ValueError("phoneme_id_map required")
    
    phonemizer = EspeakPhonemizer()
    phonemes_list = phonemizer.phonemize(voice, text)
    
    # Объединяем все фонемы
    phonemes = []
    for sent_phonemes in phonemes_list:
        phonemes.extend(sent_phonemes)
    
    # Преобразуем в ID
    phoneme_ids = []
    for phoneme in phonemes:
        if phoneme in phoneme_id_map:
            phoneme_ids.extend(phoneme_id_map[phoneme])
        else:
            print(f"⚠️ Неизвестная фонема: '{phoneme}'")
            phoneme_ids.append(0)
    
    return phoneme_ids

def generate_audio(text: str, model, config: dict, output_path: str):
    """Генерация аудио из текста"""
    
    print(f"📝 Текст: {text}")
    
    # Фонемизация
    print("🔤 Преобразование текста...")
    phoneme_id_map = config['phoneme_id_map']
    phoneme_ids = phonemize_text(text, voice="ru", phoneme_id_map=phoneme_id_map)
    
    # Подготовка входных данных
    phoneme_ids_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0)
    phoneme_lengths = torch.LongTensor([len(phoneme_ids)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])  # noise, length, noise_w
    
    # Генерация
    print("🎤 Генерация аудио...")
    with torch.no_grad():
        audio = model.forward(
            phoneme_ids_tensor, 
            phoneme_lengths, 
            scales=scales
        )[0]
    
    # Сохранение
    audio_np = audio.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    sample_rate = config.get('audio', {}).get('sample_rate', 22050)
    wavfile.write(output_path, sample_rate, audio_int16)
    
    duration = len(audio_np) / sample_rate
    print(f"💾 Сохранено: {output_path}")
    print(f"✅ Длительность: {duration:.2f} сек\n")

if __name__ == "__main__":
    # Пути
    ckpt_path = "/workspace/piper1-gpl/lightning_logs/version_15/checkpoints/epoch=851-step=370000-val_loss=27.6856.ckpt"
    config_path = "/workspace/piper1-gpl/felix_mirage_epoch749.onnx.json"  # Используем конфиг от epoch 749
    phrases_file = Path("/workspace/piper1-gpl/test_phrases_ru.txt")
    output_dir = Path("/workspace/piper1-gpl/test_audio_epoch851")
    
    output_dir.mkdir(exist_ok=True)
    
    print("🎙️  ГЕНЕРАЦИЯ АУДИО ИЗ CHECKPOINT EPOCH 851")
    print("=" * 70)
    print()
    
    # Загрузка модели
    model, config = load_checkpoint(ckpt_path, config_path)
    
    # Чтение фраз
    if phrases_file.exists():
        with open(phrases_file, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
    else:
        phrases = ["Привет! Это тест модели epoch 851."]
    
    # Генерация
    for i, phrase in enumerate(phrases, 1):
        output_file = output_dir / f"test_851_{i:02d}.wav"
        try:
            generate_audio(phrase, model, config, str(output_file))
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")
            continue
    
    print(f"🎉 Готово! Создано {len(list(output_dir.glob('*.wav')))} файлов")
