from typing import Optional, Callable

from torchcross.data import RandomInterleaveDataset, BatchedTaskSource
from .logger import logger
from .medimeta import MedIMeta


class MultiMedIMetaBatchTaskSource(RandomInterleaveDataset):
    def __init__(
        self,
        medimeta_data_path: str,
        task_ids: list[tuple[str, str]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        splits: list[str] | str | None = None,
        original_splits: list[str] | str | None = None,
        transform: Optional[Callable] = None,
    ):
        logger.info(
            f"Initializing MultiMedIMetaBatchTaskSource with medimeta_data_path={medimeta_data_path}, "
            f"task_ids={task_ids}, batch_size={batch_size}, shuffle={shuffle}, "
            f"drop_last={drop_last}, collate_fn={collate_fn}, original_splits={original_splits}, "
            f"transform={transform}"
        )
        if splits is None or isinstance(splits, str):
            splits = [splits] * len(task_ids)
        elif len(splits) == len(task_ids):
            pass
        else:
            splits = [splits] * len(task_ids)
        if original_splits is None or isinstance(original_splits, str):
            original_splits = [original_splits] * len(task_ids)
        elif len(original_splits) == len(task_ids):
            pass
        else:
            original_splits = [original_splits] * len(task_ids)

        unbatched_task_sources = [
            MedIMeta(
                data_path=medimeta_data_path,
                dataset_id=dataset_id,
                task_name=task_name,
                split=split,
                original_split=original_split,
                transform=transform,
            )
            for (dataset_id, task_name), split, original_split in zip(
                task_ids, splits, original_splits
            )
        ]
        batched_task_sources = [
            BatchedTaskSource(
                task_source=task_source,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                with_task_description=True,
            )
            for task_source in unbatched_task_sources
        ]
        super().__init__(batched_task_sources)
