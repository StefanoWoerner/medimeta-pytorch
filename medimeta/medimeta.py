import os
import warnings
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from torchcross.data import TaskSource, TaskDescription
from torchcross.data import TaskTarget
from .logger import logger


def default_transform(input_size):
    return transforms.Compose(
        [
            transforms.Resize(input_size[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * input_size[0], (0.5,) * input_size[0]),
        ]
    )


class MedIMeta(TaskSource):
    """A PyTorch Dataset and TorchCross TaskSource for the MedIMeta
    dataset.

    Args:
        data_path: The path to the MedIMeta data directory containing the
            dataset directories.
        dataset_id: The ID of the dataset to use.
        task_name: The name of the task to use.
        split: The split(s) to use for the task. If a list is provided,
            the samples from all splits in the list will be
            concatenated. Only one of 'split' and 'original_split' can
            be specified.
        original_split: The original split to use for the task. If a
            list is provided, the samples from all splits in the list
            will be concatenated. Only one of 'split' and
            'original_split' can be specified.
        transform: The transform to apply to the images.
        use_hdf5: Whether to use the HDF5 file for loading images. If
            False, the images will be loaded from TIFF files. Defaults
            to True.

    Raises:
        ValueError: If the dataset with the specified ID does not exist.
        ValueError: If no task with the specified name exists for the
            dataset.
        ValueError: If both 'split' and 'original_split' are specified.
        ValueError: If the specified split(s) or original split(s) do
            not exist for the dataset.
    """

    _data_path: str = None
    _infos: dict[str, dict] = None
    _dataset_dir_mapping: dict[str, str] = None
    _available_tasks: dict[str, list[str]] = None

    def __init__(
        self,
        data_path: str,
        dataset_id: str,
        task_name: str,
        split: str | list[str] | None = None,
        original_split: str | list[str] | None = None,
        transform: Callable[[Image], Any] | None = None,
        use_hdf5: bool = True,
    ):
        logger.info(
            f"Initializing MedIMeta with data_path={data_path}, dataset_id={dataset_id}, "
            f"task_name={task_name}, original_split={original_split}, transform={transform}, "
            f"use_hdf5={use_hdf5}"
        )
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.task_name = task_name
        self.split = split
        self.original_split = original_split
        self.transform = transform

        # get dataset info (also checks if dataset exists)
        self.info = self.get_info_dict(data_path, dataset_id)

        self.dataset_subdir = self._dataset_dir_mapping[self.dataset_id]

        self.dataset_name = self.info["name"]

        # parse task information
        task_info = next(
            (t for t in self.info["tasks"] if t["task_name"] == task_name), None
        )
        if task_info is None:
            raise ValueError(
                f"No task with name '{task_name}' found for dataset '{self.dataset_name}'"
            )
        self.task_target = TaskTarget[task_info["task_target"]]
        self.classes = task_info["labels"]
        self.task_identifier = f"{self.dataset_name}: {task_name}"
        self.domain_identifier = self.info["domain"]
        self.input_size = self.info["input_size"]
        self.task_description = TaskDescription(
            self.task_target, self.classes, self.task_identifier, self.domain_identifier
        )

        # set number of channels based on input size
        self.num_channels = self.input_size[0]

        # set transform to default if none is provided
        if self.transform is None:
            self.transform = default_transform(self.input_size)

        # set up paths and load label data
        self.image_dir = os.path.join(data_path, self.dataset_subdir, "images")
        self.label_file = os.path.join(
            data_path, self.dataset_subdir, "task_labels", f"{task_name}.npy"
        )
        self.labels: np.ndarray = np.load(self.label_file)
        if self.task_target == TaskTarget.BINARY_CLASSIFICATION:
            self.labels = self.labels[:, np.newaxis]

        self.hdf5_path = os.path.join(data_path, self.dataset_subdir, "images.hdf5")
        self.use_hdf5 = use_hdf5
        if self.use_hdf5:
            self.hdf5_images = h5py.File(self.hdf5_path, "r")["images"]

        # filter labels by split
        self.use_split_indices = False
        if split is not None and original_split is not None:
            raise ValueError(
                "Only one of 'split' and 'original_split' can be specified."
            )
        if self.split is not None:
            self.split_indices = self.get_split_indices(self.split)
            self.labels = self.labels[self.split_indices]
            self.use_split_indices = True
        if self.original_split is not None:
            self.split_indices = self.get_split_indices(
                self.original_split, use_original_split=True
            )
            self.labels = self.labels[self.split_indices]
            self.use_split_indices = True

    def get_split_indices(self, split: str | list[str], use_original_split=False):
        if isinstance(split, str):
            split = [split]
        split_indices = []
        for s in split:
            split_indices.extend(self._get_split_indices(s, use_original_split))
        return sorted(split_indices)

    def _get_split_indices(self, split: str, use_original_split=False):
        if use_original_split:
            strings = "original_splits", "Original split", "original split"
        else:
            strings = "splits", "Split", "split"

        split_file = os.path.join(
            self.data_path, self.dataset_subdir, strings[0], f"{split}.txt"
        )
        if not os.path.exists(split_file):
            available_splits = [
                k for k, v in self.info[f"{strings[0]}_num_samples"].items() if v > 0
            ]
            raise ValueError(
                f"{strings[1]} '{split}' not found for dataset {self.dataset_name}. "
                f"Available {strings[2]}s: {available_splits}"
            )
        with open(split_file, "r") as f:
            split_paths = [x.strip() for x in f.readlines()]
        return [int(p.split("/")[-1].split(".")[0]) for p in split_paths]

    def __getitem__(self, index):
        img_index = index if not self.use_split_indices else self.split_indices[index]
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
        if cls._infos is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        return list(cls._infos.keys())

    @classmethod
    def get_available_tasks(cls, data_path: str) -> dict[str, list[str]]:
        if cls._available_tasks is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        return cls._available_tasks

    @classmethod
    def get_info_dict(cls, data_path: str, dataset_id: str) -> dict:
        if cls._infos is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        if dataset_id not in cls._infos:
            raise ValueError(
                f"Dataset with ID '{dataset_id}' not found. "
                f"Available datasets (name and ID): {cls.get_available_datasets(data_path)}"
            )
        return cls._infos[dataset_id]

    @classmethod
    def _read_dataset_info(cls, data_path: str):
        cls._dataset_dir_mapping = {}
        cls._available_tasks = {}
        cls._infos = {}
        for subdir in os.listdir(data_path):
            subdir_path = os.path.join(data_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            info_path = os.path.join(subdir_path, "info.yaml")
            if not os.path.isfile(info_path):
                continue
            with open(info_path, "r") as f:
                info = yaml.load(f, Loader=yaml.FullLoader)
                if info["id"] != subdir:
                    warnings.warn(
                        f"Dataset ID '{info['id']}' does not match directory name '{subdir}'",
                        RuntimeWarning,
                    )
                cls._infos[info["id"]] = info
                cls._dataset_dir_mapping[info["id"]] = subdir
                cls._available_tasks[info["id"]] = [
                    t["task_name"] for t in info["tasks"]
                ]
        cls._data_path = data_path
