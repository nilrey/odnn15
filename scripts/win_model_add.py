from clearml import OutputModel, Task

# Создаем задачу для модели (если ещё нет)
task = Task.init(
    project_name="Research Yolo-models Car-Person Detection",
    task_name="Car-Person Detection",
    task_type=Task.TaskTypes.training
)

# Создаем OutputModel для хранения весов
model = OutputModel(
    task=task,
    name="YOLO 8 Nano Original",
    framework="PyTorch"  # или "ONNX", "TensorFlow" в зависимости от модели
)

# Добавляем файлы модели
model.update_weights(
    weights_path="models/",  # путь к файлам модели
    auto_delete_file=False,
    target_filename=None  # сохранить оригинальные имена
)

print("✅ Модель загружена в раздел Models")