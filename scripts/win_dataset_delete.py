# create example dataset
from clearml import Dataset

# Delete a dataset with ClearML`s Dataset class
dataset = Dataset.get(
    dataset_id=""
)

dataset.delete()
 