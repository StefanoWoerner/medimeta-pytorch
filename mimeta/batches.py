from typing import Optional, Callable

from torchcross.data.base import RandomChainTaskSource, BatchedTaskSource
from .logger import logger
from .mimeta import MIMeta


class MultiMIMetaBatchTaskSource(RandomChainTaskSource):
    def __init__(
        self,
        mimeta_data_path: str,
        task_ids: list[tuple[str, str]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        original_splits: list[str] | str | None = None,
        transform: Optional[Callable] = None,
    ):
        logger.info(
            f"Initializing MultiMIMetaBatchTaskSource with mimeta_data_path={mimeta_data_path}, "
            f"task_ids={task_ids}, batch_size={batch_size}, shuffle={shuffle}, "
            f"drop_last={drop_last}, collate_fn={collate_fn}, original_splits={original_splits}, "
            f"transform={transform}"
        )
        if original_splits is None or isinstance(original_splits, str):
            original_splits = [original_splits] * len(task_ids)
        else:
            assert len(original_splits) == len(task_ids)
        unbatched_task_sources = [
            MIMeta(
                data_path=mimeta_data_path,
                dataset_name=dataset_name,
                task_name=task_name,
                original_split=original_split,
                transform=transform,
            )
            for (dataset_name, task_name), original_split in zip(
                task_ids, original_splits
            )
        ]
        batched_task_sources = [
            BatchedTaskSource(
                task_source=task_source,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
            for task_source in unbatched_task_sources
        ]
        super().__init__(batched_task_sources)
