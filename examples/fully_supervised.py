from functools import partial

import lightning.pytorch as pl
import torch
import torch.nn as _nn
import torchvision.models as _models
from torch.utils.data import DataLoader

from medimeta import MedIMeta
from torchcross.data import TaskDescription
from torchcross.models.lightning import SimpleClassifier


def resnet18_backbone(pretrained=False):
    weights = _models.ResNet18_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet18(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def main(args):
    data_path = args.data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    num_workers = args.num_workers

    batch_size = 64

    # Create train and validation datasets for the target task
    train_dataset = MedIMeta(
        data_path, target_dataset_id, target_task_name, original_split="train"
    )
    dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
    available_splits = [
        k for k, v in dataset_info["original_splits_num_samples"].items() if v > 0
    ]
    val_split_name = available_splits[-1]
    val_dataset = MedIMeta(
        data_path, target_dataset_id, target_task_name, original_split=val_split_name
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    hparams = {
        "lr": 1e-3,
    }

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    task_description = train_dataset.task_description

    # Create the lighting model with pre-trained resnet18 backbone
    model = SimpleClassifier(
        resnet18_backbone(pretrained=True), task_description, optimizer
    )

    # Create the lightning trainer
    trainer = pl.Trainer(
        max_steps=100_000,
        check_val_every_n_epoch=None,
        val_check_interval=1000,
        limit_val_batches=100,
    )

    # Train the model on the target task
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="oct")
    parser.add_argument("--target_task", type=str, default="disease")
    parser.add_argument("--num_workers", type=int, default=8)

    main(parser.parse_args())
