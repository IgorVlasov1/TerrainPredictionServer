from flask import Flask, request
from keras.models import load_model
import configparser
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO
from flask_cors import CORS

# Загрузка модели
model = load_model("terrainPrediction.h5")

# Определяем словарь labels
labels = {
    0: "Здания",
    1: "Лес",
    2: "Ледник",
    3: "Гора",
    4: "Море",
    5: "Улица",
}

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    # Получаем файлы изображений из запроса
    image_files = request.files.getlist("image")

    labels = []
    probabilities = []

    for image_file in image_files:
        # Преобразуем FileStorage в BytesIO
        image_stream = BytesIO(image_file.read())

        # Загружаем изображения и преобразуем в массив
        image = load_img(image_stream, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Предсказываем
        prediction = model.predict(image)

        # Преобразуем предсказание в метку класса и вероятность
        label, probability = decode_prediction(prediction)

        labels.append(label)
        probabilities.append(probability)

    # Вовзращаем метку и предсказание на клиент
    return {"labels": labels, "probabilities": probabilities}


def decode_prediction(prediction):
    # Получаем индекс класса с наибольшей вероятностью
    class_index = np.argmax(prediction)

    # Сопоставляем индекс класса с меткой класса
    label = labels[class_index]

    # Получаем вероятность для предсказанного класса и преобразуем ее в float
    probability = float(prediction[0][class_index])

    return label, probability


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
