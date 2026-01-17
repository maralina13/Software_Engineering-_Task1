# Программная инженерия — СЗ №1  
## 4 ML задачи: текст / аудио / изображения / видео

В репозитории реализованы 4 задачи машинного обучения на готовых предобученных моделях (без обучения/дообучения):
- Text: анализ тональности
- Audio: speech-to-text (распознавание речи)
- Image: классификация изображения (top-5)
- Video: детекция объектов по кадрам

## Зависимости и установка

### Виртуальное окружение
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Пакеты Python
```bash
pip install transformers torch torchvision opencv-python pillow numpy faster-whisper soundfile
```

## Запуск

### 1) Text — анализ тональности
```bash
python text_task/main.py
```
Ожидается: вывод label (positive/negative) и score для каждого текста.

### 2) Audio — speech-to-text
```bash
python audio_task/main.py
```
Ожидается: вывод языка (например, ru) и распознанного текста (TRANSCRIPT).

### 3) Image — классификация (top-5)
```bash
python image_task/main.py
```
Ожидается: JSON с top-5 классами (label) и вероятностями (prob).

### 4) Video — детекция объектов по кадрам
```bash
python video_task/main.py
```
Ожидается в консоли: Processed frame ...: N detections

и создаются файлы:
video_task/output/detections.json

video_task/output/frame_*.jpg (несколько кадров с рамками)
