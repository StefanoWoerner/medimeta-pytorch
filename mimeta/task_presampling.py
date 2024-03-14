import os
import pickle as pkl

from mimeta import MIMeta
from torchcross.data.metadataset import (
    FewShotMetaDataset,
    SubTaskRandomFewShotMetaDataset,
)
from torchcross.data.task import Task

overwrite = False


def available_tasks(data_path) -> list[tuple[str, str]]:
    task_dict = MIMeta.get_available_tasks(data_path)
    return [(dataset, task) for dataset, tasks in task_dict.items() for task in tasks]


def create_few_shot_tasks(
    data_path,
    dataset_id: str,
    task_name: str,
    n_support: int,
    n_query: int,
    length: int,
    split: str | list[str] | None = None,
) -> list[Task]:
    task_source = MIMeta(data_path, dataset_id, task_name, split=split)
    few_shot = FewShotMetaDataset(
        task_source, None, n_support, n_query, length=length, output_indices=True
    )
    print(
        f"Creating {length} few-shot tasks for: {dataset_id} {task_name} {n_support} {n_query}"
    )
    try:
        print("Length: ", len(few_shot))
    except ValueError:
        print("Length: None")
    task_list = [t for t in few_shot]
    print("Total: ", len(task_list))
    return task_list


def create_random_few_shot_tasks(
    data_path,
    dataset_id: str,
    task_name: str,
    n_support_min: int,
    n_support_max: int,
    n_query: int,
    length: int,
    split: str | list[str] | None = None,
) -> list[Task]:
    task_source = MIMeta(data_path, dataset_id, task_name, split=split)
    few_shot = SubTaskRandomFewShotMetaDataset(
        task_source,
        None,
        n_support_samples_per_class_min=n_support_min,
        n_support_samples_per_class_max=n_support_max,
        n_query_samples_per_class=n_query,
        length=length,
        output_indices=True,
    )
    print(
        f"Creating {length} few-shot tasks for: {dataset_id} {task_name} {n_support_min}-{n_support_max} {n_query}"
    )
    try:
        print("Length: ", len(few_shot))
    except ValueError:
        print("Length: None")
    task_list = [t for t in few_shot]
    print("Total: ", len(task_list))
    return task_list


def save_few_shot_tasks(data_path, save_path=None, split=None):
    n_query = 10
    length = 100
    os.makedirs(save_path, exist_ok=True)
    # create few-shot instances for all tasks and all nshot values
    # and save them to pkl files
    for dataset, task in available_tasks(data_path):
        for n_support in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
            few_shot_tasks = create_few_shot_tasks(
                data_path, dataset, task, n_support, n_query, length, split
            )
            if few_shot_tasks is None:
                continue
            file_name = f"{task}_{n_support}_{n_query}_{length}.pkl"
            if split is not None:
                if isinstance(split, str):
                    file_name = f"{file_name[:-4]}_{split}.pkl"
                else:
                    file_name = f"{file_name[:-4]}_{'-'.join(split)}.pkl"
            file_path = os.path.join(save_path, dataset, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(file_path) and not overwrite:
                raise FileExistsError(
                    f"File {file_name} already exists. Set overwrite to True to overwrite."
                )
            with open(file_path, "wb") as f:
                pkl.dump(few_shot_tasks, f)


def save_random_few_shot_tasks(data_path, save_path=None, split=None):
    n_query = 10
    length = 1000
    n_support_min = 1
    n_support_max = 10
    os.makedirs(save_path, exist_ok=True)
    for dataset, task in available_tasks(data_path):
        few_shot_tasks = create_random_few_shot_tasks(
            data_path,
            dataset,
            task,
            n_support_min,
            n_support_max,
            n_query,
            length,
            split,
        )
        if few_shot_tasks is None:
            continue
        file_name = f"{task}_{n_support_min}-{n_support_max}_{n_query}_{length}.pkl"
        if split is not None:
            if isinstance(split, str):
                file_name = f"{file_name[:-4]}_{split}.pkl"
            else:
                file_name = f"{file_name[:-4]}_{'-'.join(split)}.pkl"
        file_path = os.path.join(save_path, dataset, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(
                f"File {file_name} already exists. Set overwrite to True to overwrite."
            )
        with open(file_path, "wb") as f:
            pkl.dump(few_shot_tasks, f)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MIMeta")
    parser.add_argument("--save_path", type=str, default="data/MIMeta_presampled")
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path
    split = args.split
    if split is not None:
        split = split.split("-")
        if isinstance(split, list) and len(split) == 1:
            split = split[0]

    print("Available tasks:")
    print(available_tasks(data_path))
    save_few_shot_tasks(data_path, save_path, split)
    save_random_few_shot_tasks(data_path, save_path, split)


if __name__ == "__main__":
    main()
