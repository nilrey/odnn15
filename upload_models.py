from clearml import Model
import os

# Путь к папке с локальными моделями
MODELS_DIR = "d:/Projects/git-practice/odnn15/models"  # или полный путь
PROJECT_NAME = "ICIE Detection Project"

# Названия моделей для регистрации
MODELS_TO_REGISTER = ["yolo12n.pt", "yolov8n.pt"]

for filename in MODELS_TO_REGISTER:
    local_path = os.path.join(MODELS_DIR, filename)
    model_name = os.path.splitext(filename)[0]  # yolo12n, yolov8n

    # Регистрируем модель в ClearML
    model = Model.create(
        name=model_name,
        project=PROJECT_NAME,
        description=f"Модель {model_name} для обнаружения объектов",
    )

    print(f"✅ Модель {model_name} зарегистрирована в ClearML (ID: {model.id})")