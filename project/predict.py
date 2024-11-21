import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Папки
MODEL_DIR = "C:\\Users\\user\\Desktop\\project\\model"
DATA_DIR = "C:\\Users\\user\\Desktop\\project\\data"

# Загрузка модели и токенайзера
model_path = os.path.join(MODEL_DIR, "model.h5")
tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")

model = load_model(model_path)

with open(tokenizer_path, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Конфигурация
max_length = 10  # Должна совпадать с длиной, использованной при обучении

# Ввод задания
while True:
    task_file = input("Введите имя файла с заданием (или 'exit' для выхода): ")
    if task_file.lower() == "exit":
        break

    try:
        # Чтение текста из файла
        filepath = os.path.join(DATA_DIR, task_file)
        with open(filepath, "r", encoding="utf-8") as file:
            task_text = file.read().strip()

        # Токенизация и предобработка
        sequence = tokenizer.texts_to_sequences([task_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Предсказание
        prediction = model.predict(padded_sequence)
        print(f"Оценка сложности задания: {prediction[0][0]:.2f}")

    except FileNotFoundError:
        print("Файл не найден. Попробуйте ещё раз.")
