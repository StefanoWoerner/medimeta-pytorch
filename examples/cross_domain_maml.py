from functools import partial

import lightning.pytorch as pl
import torch
import torch.nn as _nn
import torchopt
import torchvision.models as _models
from torch.utils.data import DataLoader

import torchcross as tx
from mimeta import (
    get_available_tasks,
    MultiPickledMIMetaTaskDataset, PickledMIMetaTaskDataset,
)
from torchcross.models.lightning import CrossDomainMAML
from torchcross.utils.collate_fn import identity


def resnet18_backbone(pretrained=False):
    weights = _models.ResNet18_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet18(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def main(args):
    data_path = args.data_path
    presampled_data_path = args.presampled_data_path
    target_dataset_name = args.target_dataset
    target_task_name = args.target_task
    validation_dataset_name = args.validation_dataset
    validation_task_name = args.validation_task
    num_workers = args.num_workers

    batch_size = 64

    task_dict = get_available_tasks(data_path)
    train_tasks = [
        (ds, t)
        for ds, tasks in task_dict.items()
        for t in tasks
        if ds != target_dataset_name
    ]

    # Create the cross-domain meta-dataset from pre-sampled tasks
    train_dataset = MultiPickledMIMetaTaskDataset(
        presampled_data_path,
        data_path,
        train_tasks,
        n_support=(1, 10),
        n_query=10,
        length=1000,
        collate_fn=tx.utils.collate_fn.stack,
    )
    val_dataset = PickledMIMetaTaskDataset(
        presampled_data_path,
        data_path,
        validation_dataset_name,
        validation_task_name,
        n_support=5,
        n_query=20,
        length=400,
        collate_fn=tx.utils.collate_fn.stack,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    # Create optimizer
    outer_optimizer = partial(torch.optim.Adam, lr=0.001)
    inner_optimizer = partial(torchopt.MetaSGD, lr=0.1)
    eval_inner_optimizer = partial(torch.optim.SGD, lr=0.1)
    num_inner_steps = 4
    eval_num_inner_steps = 32

    # Create the lighting model with pre-trained resnet18 backbone
    model = CrossDomainMAML(resnet18_backbone(pretrained=True),
                            outer_optimizer,
                            inner_optimizer,
                            eval_inner_optimizer,
                            num_inner_steps,
                            eval_num_inner_steps)



    # Create the lightning trainer
    trainer = pl.Trainer(
        inference_mode=False,
        max_epochs=1,
        check_val_every_n_epoch=1,
        val_check_interval=1000,
        limit_val_batches=100,
    )

    # Pre-train the model on all the tasks in MIMeta except the target task
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model
    # Uncomment the following line to save the pretrained model
    # torch.save(model.state_dict(), "model.pt")

    # Create the test dataloader
    test_dataset = PickledMIMetaTaskDataset(
        presampled_data_path,
        data_path,
        target_dataset_name,
        target_task_name,
        n_support=5,
        n_query=20,
        length=400,
        collate_fn=tx.utils.collate_fn.stack,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    # Meta-test the model on the target task
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MIMeta")
    parser.add_argument(
        "--presampled_data_path", type=str, default="data/MIMeta_presampled"
    )
    parser.add_argument("--target_dataset", type=str, default="OCT")
    parser.add_argument("--target_task", type=str, default="disease")
    parser.add_argument("--validation_dataset", type=str, default="OCT")
    parser.add_argument("--validation_task", type=str, default="disease")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args)
