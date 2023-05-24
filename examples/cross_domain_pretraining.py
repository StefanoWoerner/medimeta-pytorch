from functools import partial

import lightning.pytorch as pl
import torch
import torch.nn as _nn
import torchvision.models as _models
from torch.utils.data import DataLoader

import torchcross as tx
from mimeta import MIMeta, MultiMIMetaBatchTaskSource, get_available_tasks
from torchcross.data.metadataset import FewShotMetaDataset
from torchcross.data.task_source import SubsetTaskSource, BatchedTaskSource
from torchcross.models.lightning import SimpleCrossDomainClassifier
from torchcross.utils.collate_fn import identity


def resnet18_backbone(pretrained=False):
    weights = _models.ResNet18_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet18(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def main(args):
    data_path = args.data_path
    target_dataset_name = args.target_dataset
    target_task_name = args.target_task
    num_workers = args.num_workers

    batch_size = 64

    task_dict = get_available_tasks(data_path)
    train_val_tasks = [
        (ds, t)
        for ds, tasks in task_dict.items()
        for t in tasks
        if ds != target_dataset_name
    ]

    # Get the validation splits for each task in the train-val set
    train_val_info = [MIMeta.get_info_dict(data_path, ds) for ds, t in train_val_tasks]
    available_splits = [
        [k for k, v in info["original_splits_num_samples"].items() if v > 0]
        for info in train_val_info
    ]
    val_splits = [splits[-1] for splits in available_splits]

    # Create the cross-domain batch task sources
    train_dataset = MultiMIMetaBatchTaskSource(
        data_path,
        train_val_tasks,
        batch_size,
        collate_fn=tx.utils.collate_fn.stack,
        original_splits="train",
    )
    val_dataset = MultiMIMetaBatchTaskSource(
        data_path,
        train_val_tasks,
        batch_size,
        collate_fn=tx.utils.collate_fn.stack,
        original_splits=val_splits,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    hparams = {
        "lr": 1e-3,
    }

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    # Create the lighting model with pre-trained resnet18 backbone
    model = SimpleCrossDomainClassifier(resnet18_backbone(pretrained=True), optimizer)

    # Create the lightning trainer
    trainer = pl.Trainer(
        inference_mode=False,
        max_epochs=10,
        check_val_every_n_epoch=1,
        val_check_interval=1000,
        limit_val_batches=100,
    )

    # Pre-train the model on all the tasks in MIMeta except the target task
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model
    # Uncomment the following line to save the pretrained model
    # torch.save(model.state_dict(), "model.pt")

    # Create one 10-shot task from the target dataset
    target_task_source = MIMeta(data_path, target_dataset_name, target_task_name)
    target_few_shot = FewShotMetaDataset(
        target_task_source,
        collate_fn=None,
        n_support_samples_per_class=10,
        n_query_samples_per_class=10,
        filter_classes_min_samples=20,
        length=1,
        output_indices=True,
    )
    task = next(iter(target_few_shot))

    target_train_indices = [i for i, _ in task.support]
    target_test_indices = [i for i, _ in task.query]

    # Create target training and test dataloaders
    target_train_dataset = BatchedTaskSource(
        task_source=SubsetTaskSource(target_task_source, target_train_indices),
        batch_size=batch_size,
        collate_fn=tx.utils.collate_fn.stack,
        with_task_description=True,
    )
    target_test_dataset = BatchedTaskSource(
        task_source=SubsetTaskSource(target_task_source, target_test_indices),
        batch_size=batch_size,
        collate_fn=tx.utils.collate_fn.stack,
        with_task_description=True,
    )

    target_train_dataloader = DataLoader(
        target_train_dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    target_test_dataloader = DataLoader(
        target_test_dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    target_trainer = pl.Trainer(inference_mode=False, max_epochs=100)

    # Fine-tune the model on the target task
    # We are using the test set as the validation set here for simplicity
    target_trainer.fit(model, target_train_dataloader, target_test_dataloader)

    # Test the model after fine-tuning on the target task
    target_trainer.test(model, target_test_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MIMeta")
    parser.add_argument("--target_dataset", type=str, default="OCT")
    parser.add_argument("--target_task", type=str, default="disease")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args)
