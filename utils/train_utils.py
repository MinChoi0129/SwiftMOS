import os
import re
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import data_MOS
from networks import MainNetwork
from utils import builder


def get_next_case_path(log_root="logs", tags=None):
    os.makedirs(log_root, exist_ok=True)
    tags = tags or []

    # case_번호 추출
    existing = [d for d in os.listdir(log_root) if re.match(r"case_\d{2}", d)]
    numbers = [int(re.findall(r"\d{2}", name)[0]) for name in existing] if existing else [0]
    next_num = max(numbers, default=0) + 1
    case_prefix = f"case_{next_num:02d}"

    # 태그 연결
    if tags:
        tag_str = "_".join(tags)
        folder_name = f"{case_prefix}_{tag_str}"
    else:
        folder_name = case_prefix

    return os.path.join(log_root, folder_name)


def reduce_tensor(inp):
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def load_checkpoint(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]


def get_dataloaders(pDataset, pGen):
    # 데이터로더 준비
    train_dataset = data_MOS.DataloadTrain(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=pGen.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=pDataset.Train.num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_dataset = data_MOS.DataloadVal(pDataset.Val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=pDataset.Val.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def get_networks_optimizer_scheduler(pModel, pOpt, train_loader, device, local_rank):
    base_net = MainNetwork.MOSNet(pModel)
    optimizer = builder.get_optimizer(pOpt, base_net)
    per_epoch_num_iters = len(train_loader)
    scheduler = builder.get_scheduler(optimizer, pOpt, per_epoch_num_iters)

    base_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_net)
    model = torch.nn.parallel.DistributedDataParallel(
        base_net.to(device),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    return base_net, model, optimizer, scheduler


def set_starting_condition(args, model_prefix, pModel, pOpt, base_net, optimizer, scheduler, rank, logger):
    if args.keep_training:
        checkpoint_file = os.path.join(model_prefix, "{}-checkpoint.pth".format(pModel.pretrain.pretrain_epoch))
        if os.path.exists(checkpoint_file):
            start_epoch = load_checkpoint(checkpoint_file, base_net, optimizer, scheduler) + 1
            if rank == 0:
                logger.info("Checkpoint {} 불러와서 {} epoch 부터 재개합니다.".format(checkpoint_file, start_epoch))
        else:
            if rank == 0:
                logger.info("Checkpoint {}를 찾을 수 없습니다. 파라미터 처음부터 학습합니다.".format(checkpoint_file))
            start_epoch = pOpt.schedule.begin_epoch
    else:
        if rank == 0:
            logger.info("Checkpoint를 지정하지 않아 처음부터 학습합니다.")
        start_epoch = pOpt.schedule.begin_epoch

    return start_epoch
