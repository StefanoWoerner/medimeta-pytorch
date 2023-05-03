import os
import pickle as pkl

from torchcross.data.meta import FewShotMetaDataset, SubTaskRandomFewShotMetaDataset
from mimeta.mimeta import MIMeta
from torchcross.data.task import Task


overwrite = False


def available_tasks(data_path) -> list[tuple[str, str]]:
    task_dict = MIMeta.get_available_tasks(data_path)
    return [(dataset, task) for dataset, tasks in task_dict.items() for task in tasks]


def create_few_shot_tasks(
    data_path,
    dataset_name: str,
    task_name: str,
    n_support: int,
    n_query: int,
    length: int,
) -> list[Task]:
    task_source = MIMeta(data_path, dataset_name, task_name)
    few_shot = FewShotMetaDataset(
        task_source, None, n_support, n_query, length=length, output_indices=True
    )
    print(
        f"Creating {length} few-shot tasks for: {dataset_name} {task_name} {n_support} {n_query}"
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
    dataset_name: str,
    task_name: str,
    n_support_min: int,
    n_support_max: int,
    n_query: int,
    length: int,
) -> list[Task]:
    task_source = MIMeta(data_path, dataset_name, task_name)
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
        f"Creating {length} few-shot tasks for: {dataset_name} {task_name} {n_support_min}-{n_support_max} {n_query}"
    )
    try:
        print("Length: ", len(few_shot))
    except ValueError:
        print("Length: None")
    task_list = [t for t in few_shot]
    print("Total: ", len(task_list))
    return task_list


def save_few_shot_tasks(data_path, save_path=None):
    n_query = 20
    length = 400
    os.makedirs(save_path, exist_ok=True)
    # create few-shot instances for all tasks and all nshot values
    # and save them to pkl files
    for dataset, task in available_tasks(data_path):
        for n_support in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
            few_shot_tasks = create_few_shot_tasks(
                data_path, dataset, task, n_support, n_query, length
            )
            if few_shot_tasks is None:
                continue
            file_name = f"{dataset}_{task}_{n_support}_{n_query}_{length}.pkl"
            if os.path.exists(os.path.join(save_path, file_name)) and not overwrite:
                raise FileExistsError(
                    f"File {file_name} already exists. Set overwrite to True to overwrite."
                )
            with open(os.path.join(save_path, file_name), "wb") as f:
                pkl.dump(few_shot_tasks, f)


def save_random_few_shot_tasks(data_path, save_path=None):
    n_query = 10
    length = 20000
    n_support_min = 1
    n_support_max = 10
    os.makedirs(save_path, exist_ok=True)
    for dataset, task in available_tasks(data_path):
        few_shot_tasks = create_random_few_shot_tasks(
            data_path, dataset, task, n_support_min, n_support_max, n_query, length
        )
        if few_shot_tasks is None:
            continue
        file_name = (
            f"{dataset}_{task}_{n_support_min}-{n_support_max}_{n_query}_{length}.pkl"
        )
        if os.path.exists(os.path.join(save_path, file_name)) and not overwrite:
            raise FileExistsError(
                f"File {file_name} already exists. Set overwrite to True to overwrite."
            )
        with open(os.path.join(save_path, file_name), "wb") as f:
            pkl.dump(few_shot_tasks, f)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MIMeta")
    parser.add_argument("--save_path", type=str, default="data/MIMeta_presampled")
    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path

    print("Available tasks:")
    print(available_tasks(data_path))
    save_few_shot_tasks(data_path, save_path)
    save_random_few_shot_tasks(data_path, save_path)


if __name__ == "__main__":
    main()
