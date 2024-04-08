import os
import pickle as pkl
from typing import Optional, Callable

from torchcross.data.metadataset import MetaDataset
from torchcross.data.task import Task, TaskDescription
from torchcross.utils.collate_fn import identity
from .logger import logger
from .medimeta import MedIMeta


class PickledMedIMetaTaskDataset(MetaDataset):
    def __init__(
        self,
        task_lists_data_path: str,
        medimeta_data_path: str,
        dataset_id: str,
        task_name: str,
        n_support: int | tuple[int, int],
        n_query: int,
        length: int,
        split: str | list[str] | None = None,
        collate_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        logger.info(
            f"Initializing PickledMedIMetaTaskDataset with task_lists_data_path={task_lists_data_path}, "
            f"medimeta_data_path={medimeta_data_path}, dataset_name={dataset_id}, task_name={task_name}, "
            f"n_support={n_support}, n_query={n_query}, length={length}, "
            f"collate_fn={collate_fn}, transform={transform}"
        )
        if isinstance(n_support, int):
            file_name = f"{task_name}_{n_support}_{n_query}_{length}.pkl"
        else:
            file_name = (
                f"{task_name}_{n_support[0]}-{n_support[1]}_{n_query}_{length}.pkl"
            )
        if split is not None:
            if isinstance(split, str):
                file_name = f"{file_name[:-4]}_{split}.pkl"
            else:
                file_name = f"{file_name[:-4]}_{'-'.join(split)}.pkl"
        file_path = os.path.join(task_lists_data_path, dataset_id, file_name)
        if not os.path.exists(file_path):
            raise ValueError(f"Task list file {file_path} not found.")
        with open(file_path, "rb") as f:
            self.task_list: list[Task] = pkl.load(f)
        self.task_source = MedIMeta(
            data_path=medimeta_data_path,
            dataset_id=dataset_id,
            task_name=task_name,
            split=split,
            transform=transform,
        )
        self.collate_fn = collate_fn if collate_fn is not None else identity

    def __getitem__(self, index) -> Task:
        task: Task = self.task_list[index]
        support = [(self.task_source[i][0], l) for i, l in task.support]
        query = [(self.task_source[i][0], l) for i, l in task.query]
        support = self.collate_fn(support)
        query = self.collate_fn(query)
        return Task(support, query, task.description)

    def __len__(self) -> int:
        return len(self.task_list)


class MultiPickledMedIMetaTaskDataset(MetaDataset):
    def __init__(
        self,
        task_lists_data_path: str,
        medimeta_data_path: str,
        task_ids: list[tuple[str, str]],
        n_support: int | tuple[int, int],
        n_query: int,
        length: int,
        split: str | list[str] | None = None,
        collate_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        logger.info(
            f"Initializing MultiPickledMedIMetaTaskDataset with task_lists_data_path={task_lists_data_path}, "
            f"medimeta_data_path={medimeta_data_path}, task_ids={task_ids}, "
            f"n_support={n_support}, n_query={n_query}, length={length}, split={split}, "
            f"collate_fn={collate_fn}, transform={transform}"
        )
        self.task_list: list[tuple[Task, int]] = []
        for i, (dataset_id, task_name) in enumerate(task_ids):
            if isinstance(n_support, int):
                file_name = f"{task_name}_{n_support}_{n_query}_{length}.pkl"
            else:
                file_name = (
                    f"{task_name}_{n_support[0]}-{n_support[1]}_{n_query}_{length}.pkl"
                )
            if split is not None:
                if isinstance(split, str):
                    file_name = f"{file_name[:-4]}_{split}.pkl"
                else:
                    file_name = f"{file_name[:-4]}_{'-'.join(split)}.pkl"
            file_path = os.path.join(task_lists_data_path, dataset_id, file_name)
            if not os.path.exists(file_path):
                raise ValueError(f"Task list file {file_path} not found.")
            with open(file_path, "rb") as f:
                self.task_list.extend(zip(pkl.load(f), [i] * length))
        self.task_sources = [
            MedIMeta(
                data_path=medimeta_data_path,
                dataset_id=dataset_id,
                task_name=task_name,
                split=split,
                transform=transform,
            )
            for dataset_id, task_name in task_ids
        ]
        self.collate_fn = collate_fn if collate_fn is not None else identity

    def __getitem__(self, index) -> Task:
        task, task_source_index = self.task_list[index]
        task_source = self.task_sources[task_source_index]
        support = [(task_source[i][0], l) for i, l in task.support]
        query = [(task_source[i][0], l) for i, l in task.query]
        support = self.collate_fn(support)
        query = self.collate_fn(query)
        return Task(support, query, task.description)

    def __len__(self) -> int:
        return len(self.task_list)
