import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb
import sys  # 추가됨
import traceback
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
import warnings
from evaluate import val
from utils.metric import MultiClassMetric
from models import *

import tqdm
import logging
import importlib
from utils.logger import config_logger
from utils import builder

from torch.utils.tensorboard import SummaryWriter

from utils.pretty_print import shprint


def reduce_tensor(inp):
    """
    모든 프로세스의 loss 값을 평균 내어 rank 0에서 확인할 수 있도록 합니다.
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]


def train(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, writer, log_frequency):
    rank = torch.distributed.get_rank()
    model.train()
    # rank 0인 경우 tqdm 진행바 생성, 나머지는 단순 반복
    if rank == 0:
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{end_epoch}")
    else:
        pbar = enumerate(train_loader)

    for i, (pcds_xyzi, pcds_coord, pcds_polar_coord, pcds_target, pcds_bev_target, _) in pbar:
        loss = model(pcds_xyzi, pcds_coord, pcds_polar_coord, pcds_target, pcds_bev_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        reduced_loss = reduce_tensor(loss)
        if rank == 0:
            pbar.set_postfix(loss="%.4f" % (reduced_loss.item() / torch.distributed.get_world_size()))
            if i % log_frequency == 0:
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                log_str = "Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}; loss: {}".format(
                    epoch,
                    end_epoch,
                    i,
                    len(train_loader),
                    current_lr,
                    reduced_loss.item() / torch.distributed.get_world_size(),
                )
                logger.info(log_str)

            global_step = epoch * len(train_loader) + i  # epoch와 iteration을 조합하여 global step 계산
            if writer:
                writer.add_scalar(
                    "Train/Loss",
                    reduced_loss.item() / torch.distributed.get_world_size(),
                    global_step,
                )


def main(args, config):
    # config 파싱
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    os.system("mkdir -p {}".format(model_prefix))
    config_logger(os.path.join(save_path, "train_log.txt"))
    logger = logging.getLogger()

    device = torch.device("cuda:{}".format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))

    seed = rank * pDataset.Train.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #######################################################################################################
    # 데이터로더 준비
    train_dataset = eval("datasets.{}.DataloadTrain".format(pDataset.Train.data_src))(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=pGen.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=pDataset.Train.num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_dataset = eval("datasets.{}.DataloadVal".format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=pDataset.Val.num_workers, pin_memory=True
    )
    #######################################################################################################

    base_net = TripleMOS.AttNet(pModel)

    optimizer = builder.get_optimizer(pOpt, base_net)
    per_epoch_num_iters = len(train_loader)
    scheduler = builder.get_scheduler(optimizer, pOpt, per_epoch_num_iters)

    # 동기화 배치 정규화 적용 및 분산 학습 설정
    base_net = nn.SyncBatchNorm.convert_sync_batchnorm(base_net)
    model = torch.nn.parallel.DistributedDataParallel(
        base_net.to(device), device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

    # 학습 재개: checkpoint에서 전체 상태 로드
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

    try:
        for epoch in range(start_epoch, pOpt.schedule.end_epoch):
            model.train()
            train_sampler.set_epoch(epoch)

            train(
                epoch,
                pOpt.schedule.end_epoch,
                args,
                model,
                train_loader,
                optimizer,
                scheduler,
                logger,
                writer,
                pGen.log_frequency,
            )

            if rank == 0:
                # 체크포인트 저장: epoch, 모델, optimizer, scheduler 상태 포함
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                checkpoint_path = os.path.join(model_prefix, "{}-checkpoint.pth".format(epoch))
                save_checkpoint(checkpoint, checkpoint_path)
                logger.info("Epoch {} 체크포인트 저장: {}".format(epoch, checkpoint_path))

                if epoch >= args.start_validating_epoch and epoch % 10 in [0, 1, 2]:
                    # 평가 수행
                    v_model = eval(pModel.prefix)(pModel)
                    v_model.cuda()
                    v_model.eval()
                    eval_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    v_model.load_state_dict(eval_checkpoint["model_state_dict"])
                    logger.info("{} 체크포인트를 이용하여 평가합니다.".format(checkpoint_path))
                    val(epoch, v_model, val_loader, pGen.category_list, save_path, writer, rank, save_label=False)
    except KeyboardInterrupt:
        if rank == 0:
            logger.info("KeyboardInterrupt 감지됨. 안전하게 종료합니다...")
    except Exception as e:
        if rank == 0:
            traceback.print_exc()  # 전체 Traceback 출력
    finally:
        # writer 종료 및 프로세스 그룹 해제
        if writer is not None:
            writer.close()
        torch.distributed.destroy_process_group()
        if rank == 0:
            logger.info("리소스(CPU, GPU, RAM) 모두 해제되었습니다.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="lidar segmentation")
    parser.add_argument("--config", help="config file path", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--start_validating_epoch", type=int, default=0)
    parser.add_argument("--keep_training", action="store_true")

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    try:
        main(args, config)
    except KeyboardInterrupt:
        # 메인에서도 KeyboardInterrupt를 catch하여 안전 종료
        print("Graceful Shutdown")
        traceback.print_exc()
    finally:
        sys.exit(0)
