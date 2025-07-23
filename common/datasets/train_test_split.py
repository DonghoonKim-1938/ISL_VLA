import numpy as np
import os

from common.datasets.sample_dataset import sample_dataset
from common.datasets.create_meta import create_meta

def get_split_indices(dataset_size, test_size):
    step_size = int(1/test_size)
    test_index_size = int(dataset_size * test_size)

    total_index = np.arange(dataset_size)
    test_index = np.random.randint(step_size, size=test_index_size) + step_size * np.arange(test_index_size)
    train_index = np.setdiff1d(total_index, test_index)

    return train_index.tolist(), test_index.tolist()


def split_train_test(root, dataset_size, test_size=0.1):
    train_index, test_index = get_split_indices(dataset_size, test_size)

    train_root = os.path.join(root, "train")
    test_root = os.path.join(root, "test")

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    sample_dataset(train_index, root, train_root)
    sample_dataset(test_index, root, test_root)

    create_meta(train_root, len(train_index))
    create_meta(test_root, len(test_index))


if __name__=="__main__":
    root_dir = "/datasets/piper_grape0711/lerobot_5hz"
    dataset_size = 2160
    split_train_test(root_dir, dataset_size, test_size=0.1)