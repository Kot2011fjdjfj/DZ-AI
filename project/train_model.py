import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Данные: {имя файла: оценка сложности}
data = {
    "data1.txt": 7,
    "data2.txt": 8,
    "data3.txt": 6,
    "data4.txt": 5,
    "data5.txt": 7,
    "data6.txt": 8,
    "data7.txt": 4,
    "data8.txt": 7,
    "data9.txt": 6,
    "data10.txt": 8,
    "data11.txt": 6,
    "data12.txt": 3,
    "data13.txt": 1,
    "data14.txt": 2,
    "data15.txt": 6,
    "data16.txt": 4,
    "data17.txt": 4,
    "data18.txt": 5,
    "data19.txt": 6,
    "data20.txt": 4,
    "data21.txt": 7,
    "data22.txt": 9,
    "data23.txt": 8,
    "data24.txt": 10,
    "data25.txt": 8,
    "data26.txt": 1,
    "data27.txt": 3,
    "data28.txt": 7,
}

# Папки
DATA_DIR = "C:\\Users\\user\\Desktop\\project\\data"
MODEL_DIR = "C:\\Users\\user\\Desktop\\project\\model"

# Создаём папку для модели, если её нет
os.makedirs(MODEL_DIR, exist_ok=True)

# Считываем тексты и оценки сложности
tasks = []
labels = []

for filename, complexity in data.items():
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        tasks.append(file.read().strip())
    labels.append(complexity)

labels = np.array(labels)  # Преобразуем оценки сложности в массив

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tasks)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(tasks)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Создание модели
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)  # Одно значение сложности
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Обучение модели
model.fit(padded_sequences, labels, epochs=5000, verbose=2)

# Сохранение модели и токенайзера
model.save(os.path.join(MODEL_DIR, "model.h5"))

with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
    import json
    f.write(json.dumps(tokenizer.to_json()))

print(f"Модель и токенайзер успешно сохранены в папке '{MODEL_DIR}'.")
