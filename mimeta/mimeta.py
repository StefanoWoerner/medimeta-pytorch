import os

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from torchcross.data.base import TaskSource
from torchcross.data.task import TaskTarget
from .logger import logger


def default_transform(input_size):
    return transforms.Compose(
        [
            transforms.Resize(input_size[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * input_size[0], (0.5,) * input_size[0]),
        ]
    )


class MIMeta(TaskSource):
    _data_path: str = None
    _dataset_dir_mapping: dict[str, str] = None
    _available_tasks: dict[str, list[str]] = None

    def __init__(
        self,
        data_path,
        dataset_name,
        task_name,
        original_split=None,
        transform=None,
        use_hdf5=True,
    ):
        logger.info(
            f"Initializing MIMeta with data_path={data_path}, dataset_name={dataset_name}, "
            f"task_name={task_name}, original_split={original_split}, transform={transform}, "
            f"use_hdf5={use_hdf5}"
        )
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.original_split = original_split
        self.transform = transform

        available_datasets = self.get_available_datasets(data_path)
        if dataset_name not in available_datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}"
            )

        dataset_subdir = self._dataset_dir_mapping[dataset_name]

        # load info.yaml file and parse task information
        info_file = os.path.join(data_path, dataset_subdir, "info.yaml")
        with open(info_file, "r") as f:
            info = yaml.load(f, Loader=yaml.FullLoader)
            task_info = next(
                (t for t in info["tasks"] if t["task_name"] == task_name), None
            )
            if task_info is None:
                raise ValueError(
                    f"No task with name '{task_name}' found for dataset '{dataset_name}'"
                )
            self.classes = task_info["labels"]
            self.task_target = TaskTarget[task_info["task_target"]]
            self.input_size = info["input_size"]

        # set number of channels based on input size
        self.num_channels = self.input_size[0]

        # set transform to default if none is provided
        if self.transform is None:
            self.transform = default_transform(self.input_size)

        # set up paths and load label data
        self.image_dir = os.path.join(data_path, dataset_subdir, "images")
        self.label_file = os.path.join(
            data_path, dataset_subdir, "task_labels", f"{task_name}.npy"
        )
        self.labels: np.ndarray = np.load(self.label_file)
        if self.task_target == TaskTarget.BINARY_CLASSIFICATION:
            self.labels = self.labels[:, np.newaxis]

        self.hdf5_path = os.path.join(data_path, dataset_subdir, "images.hdf5")
        self.use_hdf5 = use_hdf5
        if self.use_hdf5:
            self.hdf5_images = h5py.File(self.hdf5_path, "r")["images"]

        if self.original_split is not None:
            split_file = os.path.join(
                data_path, dataset_subdir, "original_splits", f"{original_split}.txt"
            )
            if not os.path.exists(split_file):
                available_splits = [k for k, v in info["num_samples"].items() if v > 0]
                raise ValueError(
                    f"Split '{original_split}' not found for dataset {dataset_name}. "
                    f"Available splits: {available_splits}"
                )
            with open(split_file, "r") as f:
                split_paths = [x.strip() for x in f.readlines()]
            self.split_indices = [
                int(p.split("/")[-1].split(".")[0]) for p in split_paths
            ]
            self.labels = self.labels[self.split_indices]

        self.task_identifier = f"{dataset_name}: {task_name}"

    def __getitem__(self, index):
        img_index = index if self.original_split is None else self.split_indices[index]
        if self.use_hdf5:
            image_array = self.hdf5_images[img_index, ...]
            image = Image.fromarray(image_array)
        else:
            image_path = os.path.join(self.image_dir, f"{img_index:06d}.tiff")
            image = Image.open(image_path)
        if image.mode == "RGB" and self.num_channels == 1:
            # convert RGB images to grayscale
            image = image.convert("L")
        elif image.mode == "L" and self.num_channels == 3:
            # convert grayscale images to RGB
            image = image.convert("RGB")
            # image = Image.merge("RGB", [image] * 3)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, torch.as_tensor(label)

    def __len__(self):
        return len(self.labels)

    @classmethod
    def get_available_datasets(cls, data_path: str) -> list[str]:
        if cls._dataset_dir_mapping is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        return list(cls._dataset_dir_mapping.keys())

    @classmethod
    def get_available_tasks(cls, data_path: str) -> dict[str, list[str]]:
        if cls._available_tasks is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        return cls._available_tasks

    @classmethod
    def _read_dataset_info(cls, data_path: str):
        cls._dataset_dir_mapping = {}
        cls._available_tasks = {}
        for subdir in os.listdir(data_path):
            subdir_path = os.path.join(data_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            info_path = os.path.join(subdir_path, "info.yaml")
            if not os.path.isfile(info_path):
                continue
            with open(info_path, "r") as f:
                info = yaml.load(f, Loader=yaml.FullLoader)
                cls._dataset_dir_mapping[info["name"]] = subdir
                cls._available_tasks[info["name"]] = [
                    t["task_name"] for t in info["tasks"]
                ]
        cls._data_path = data_path
