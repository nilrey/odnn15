# create example dataset
from clearml import Dataset

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="Research Yolo-models Car-Person Detection", 
    dataset_name="Car-Person Detection"
)

dataset.add_files(
    path="data/input/",
    verbose=False,                        # написание вывода в консоль
    wildcard="*.mp4",                     # только MP4 файлы
    recursive=False,                      # искать во вложенных папках
    # local_base_folder="",            # базовый путь для относительных путей
    max_workers=2                         # количество потоков, с помощью которых будут добавлены файлы. По умолчанию используется количество логических ядер
)

# Upload dataset to ClearML server (customizable)
dataset.upload()

# commit dataset changes
dataset.finalize()