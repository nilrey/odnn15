# Video Analytics Tools - детекция, трекинг и анализ объектов в видео

Набор инструментов для обработки видео с использованием YOLO моделей. Поддерживает детекцию и трекинг объектов (автомобили, люди), построение тепловых карт активности, отслеживание поз человека, добавление шумов и размытия. Интеграция с ClearML для логирования метрик, артефактов и визуализации результатов.

Стек: Python 3.10+, OpenCV, YOLO (ultralytics), ClearML, NumPy.

## Структура модулей

video_analytics/

├── gaussian_blur.py      # добавление размытия и шума в видео

├── hitmap.py             # трекинг авто + тепловая карта плотности

├── tracking_pose.py      # трекинг людей + отображение скелета (pose)

├── tracking_cml.py       # универсальный трекинг с логированием в ClearML

└── models/               # директория с .pt файлами моделей YOLO


## Структура директорий (для текущих путей в проекте):
.

├── data/

│   ├── input/

│   │   ├── cars_1.mp4

│   │   ├── cars_1_1.mp4

│   │   ├── people_1.mp4

│   │   └── 001/

│   │       └── *.mp4

│   └── output/

│       ├── noise-gauss-*.mp4

│       ├── hitmap-*.mp4

│       ├── heatmap_*.png

│       ├── out-people-*.mp4

│       └── 001/

│           └── out-*.mp4

├── models/

│   ├── yolov8n.pt

│   ├── yolov8x-pose.pt

│   └── yolo12x.pt

├── gaussian_blur.py

├── hitmap.py

├── tracking_pose.py

└── tracking_cml.py

## Общая функциональность

1. Гауссово размытие и шум (gaussian_blur.py)
2. Трекинг автомобилей с тепловой картой (hitmap.py)
3. Трекинг людей с отображением позы (tracking_pose.py)
4. Универсальный трекинг объектов с логированием в ClearML (tracking_cml.py)

## Установка

Python 3.10 или выше

Установка зависимостей:
pip install opencv-python ultralytics clearml numpy

Настройка ClearML (для tracking_cml.py и hitmap.py):
clearml-init

## Модуль 1: gaussian_blur.py

Назначение: добавление гауссовского размытия и нормального шума к видео.

Функция add_gaussian_blur_and_noise(input_video, output_video, blur_kernel=(5,5), noise_intensity=25)

Параметры:
- input_video: путь к исходному видео
- output_video: путь для сохранения
- blur_kernel: кортеж (x, y) размер ядра размытия (нечётные числа)
- noise_intensity: интенсивность шума (0-255)

Запуск:
python gaussian_blur.py

Перед запуском изменить в блоке __main__:
- VIDEO_FILE_NAME = "cars_1"
- NOISE = 25
- BLUR_SIZE = 15
- input_file = f'data/input/{VIDEO_FILE_NAME}.mp4'
- output_file = f'data/output/noise-gauss-{VIDEO_FILE_NAME}-n{NOISE}-b{BLUR_SIZE}.mp4'

Входные данные: data/input/*.mp4

Выходные данные: data/output/*.mp4

## Модуль 2: hitmap.py

Назначение: трекинг автомобилей с построением тепловой карты плотности. Логирование в ClearML.

Функция main(model_name, input_file, output_file, confidence)

Параметры:
- model_name: имя файла модели (без .pt) из папки models/
- input_file: имя входного видео (с расширением)
- output_file: имя выходного видео
- confidence: порог уверенности (0.0-1.0)

Детектируемые классы: 0, 2, 3, 5, 6, 7, 8 (person, car, motorcycle, bus, truck)

Особенности:
- трекинг с persist=True
- тепловая карта с затуханием (decay 0.95)
- гауссово ядро для каждой детекции
- наложение heatmap с прозрачностью 0.5
- легенда heatmap (градиент от Low к High)
- сохранение финальной heatmap как PNG
- ClearML: логи скаляров (объекты на фрейм, среднее, общее), гистограммы распределения, артефакты (heatmap, видео)

Запуск:
python hitmap.py

Перед запуском изменить в блоке __main__:
- model_name = "yolov8n"
- confidence = 0.5
- input_name = "cars_1_1"
- output_name = f"hitmap-{input_name}-{model_name}-conf-{confidence}"

Входные данные: data/input/*.mp4
Выходные данные: data/output/*.mp4, data/output/heatmap_*.png

## Модуль 3: tracking_pose.py

Назначение: трекинг людей с отображением позы (скелета) и ключевых точек.

Модель: YOLOv8x-pose (ключевые точки тела)

Особенности:
- отображение bounding box (опционально)
- отображение скелета (17 ключевых точек, 12 соединений)
- выделение области головы отдельным прямоугольником
- уникальный цвет для каждого отслеживаемого объекта (на основе ID)
- ширина линии: 2 пикселя

Параметры в коде:
- MODEL_TAG = "yolov8x-pose"
- SHOW_BOUNDING_BOXES = False
- SHOW_POSE = True
- STROKE_WIDTH = 2
- confidence = 0.5 (встроено)

Запуск:

python tracking_pose.py

Входные данные: data/input/people_1.mp4 (путь зашит в коде)

Выходные данные: data/output/out-people-yolov8x-pose-conf-05.mp4

Перед запуском изменить:

video_path = 'data/input/people_1.mp4'

## Модуль 4: tracking_cml.py

Назначение: универсальный трекинг объектов с детальной аналитикой и логированием в ClearML. Основной модуль для production сценариев.

Функция main(cml_project_name, cml_task_name, model_name, input_video, output_video, confidence, date_stamp)

Параметры:
- cml_project_name: проект в ClearML
- cml_task_name: имя задачи в ClearML
- model_name: имя файла модели (yolo12x, yolov8n и т.д.)
- input_video: полный путь к входному видео
- output_video: полный путь для сохранения
- confidence: порог уверенности
- date_stamp: временная метка (для отображения)

Детектируемые классы: {0, 2, 3, 5, 6, 7, 8} (person + автомобили)

Логируемые метрики (ClearML):
- объекты на фрейм (скаляр)
- общее количество обнаруженных объектов (скаляр)
- среднее количество объектов на фрейм (скаляр)
- трекируемые объекты (суммарно) (скаляр)
- гистограмма объектов на фрейме
- гистограмма трекируемых объектов (нарастающий итог)
- гистограмма изменений между фреймами
- гистограмма распределения confidence по диапазонам (0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0)
- single values: всего фреймов, всего объектов, трекируемых объектов, объекты по категориям (person, car), объекты по диапазонам confidence
- артефакт: processed_video

Запуск:

python tracking_cml.py

Перед запуском настроить:
- cml_project_name = "ICIE Detection Project New"
- model_name = "yolo12x"
- confidence = 0.5
- path_input = текущая_директория/data/input/001

Модуль автоматически обрабатывает все .mp4 файлы из data/input/001/

Входные данные: data/input/001/*.mp4

Выходные данные: data/output/001/out-*.mp4

## Формат входных данных

Все модули ожидают видео в формате MP4, H.264 кодировка.


## Решаемые задачи

1. Добавить шум и размытие: gaussian_blur.py

2. Построить тепловую карту для автомобилей: hitmap.py

3. Отследить позы людей: tracking_pose.py

4. Пакетная обработка с определением людей и машин, с логированием в ClearML: tracking_cml.py

## ClearML интеграция

Для hitmap.py и tracking_cml.py требуется настроенный ClearML сервер или облачный агент.

Переменные окружения (опционально):
- CLEARML_API_HOST
- CLEARML_FILES_HOST
- CLEARML_WEB_HOST

Логирование:
- скалярные метрики: report_scalar
- гистограммы: report_histogram
- артефакты: upload_artifact
- параметры задачи: set_parameter

## Примечания

- Все модули используют абсолютные или относительные пути. Рекомендуется запускать из корневой директории проекта.
- Модели YOLO должны находиться в папке models/
- Для pose tracking требуется модель с ключевыми точками (yolov8x-pose.pt)
- Размер кадра при обработке через model.track() приводится к кратному 32 для совместимости с YOLO
- Выходные видео сохраняются с кодеком mp4v
- При отсутствии детекций тепловая карта отображается как чёрное поле
- Confidence распределение в tracking_cml.py рассчитывается для всех обнаруженных объектов за всё видео
