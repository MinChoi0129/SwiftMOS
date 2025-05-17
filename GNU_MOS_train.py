import torch, argparse, random, sys, traceback, warnings, logging, tqdm, importlib, os
import numpy as np
from GNU_MOS_evaluate import val
from networks import MainNetwork
from utils.logger import config_logger
from torch.utils.tensorboard import SummaryWriter

from utils.train_utils import (
    get_dataloaders,
    get_networks_optimizer_scheduler,
    get_next_case_path,
    reduce_tensor,
    set_starting_condition,
)


def save_checkpoint_and_eval_using_it(
    epoch, model, optimizer, scheduler, model_prefix, logger, pModel, val_loader, pGen, save_path, writer, rank
):
    if rank != 0:
        return

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    checkpoint_path = os.path.join(model_prefix, "{}-checkpoint.pth".format(epoch))

    torch.save(checkpoint, checkpoint_path)
    logger.info("Epoch {} 체크포인트 저장: {}".format(epoch, checkpoint_path))

    run_EVAL = False
    if epoch >= args.start_validating_epoch:
        if epoch <= 10 and epoch in [0, 2, 9]:
            run_EVAL = True
        elif epoch < 40 and epoch % 10 in [4, 9]:
            run_EVAL = True
        elif epoch % 10 in [0, 2, 4, 6, 8]:
            run_EVAL = True

    if run_EVAL:
        v_model = MainNetwork.MOSNet(pModel)
        v_model.cuda()
        v_model.eval()
        eval_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        v_model.load_state_dict(eval_checkpoint["model_state_dict"])
        logger.info("{} 체크포인트를 이용하여 평가합니다.".format(checkpoint_path))
        val(epoch, v_model, val_loader, pGen.category_list, save_path, writer, save_label=False)


def train_one_epoch(epoch, end_epoch, model, train_loader, optimizer, scheduler, logger, writer, log_frequency):
    rank = torch.distributed.get_rank()
    model.train()

    pbar = (
        tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{end_epoch}",
        )
        if rank == 0
        else enumerate(train_loader)
    )

    for i, (xyzi, descartes_coord, sphere_coord, label, bev_label, meta_list_raw) in pbar:
        loss, loss_2d, loss_3d = model(xyzi, descartes_coord, sphere_coord, label, bev_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        reduced_losses = {
            "total_loss": reduce_tensor(loss),
            "loss_2d": reduce_tensor(loss_2d),
            "loss_3d": reduce_tensor(loss_3d),
        }

        if rank == 0:
            pbar.set_postfix(loss="%.4f" % (reduced_losses["total_loss"].item() / torch.distributed.get_world_size()))
            if i % log_frequency == 0:
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                log_str = "Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}; loss: {}".format(
                    epoch,
                    end_epoch,
                    i,
                    len(train_loader),
                    current_lr,
                    reduced_losses["total_loss"].item() / torch.distributed.get_world_size(),
                )
                logger.info(log_str)

            global_step = epoch * len(train_loader) + i  # epoch와 iteration을 조합하여 global step 계산
            if writer:
                for key, value in reduced_losses.items():
                    writer.add_scalar(
                        f"Train/{key}",
                        value.item() / torch.distributed.get_world_size(),
                        global_step,
                    )


def main(args, config):
    # 설정 파일 불러오기
    pGen, pDataset, pModel, pOpt = config.get_config()
    prefix = pGen.name

    # 저장 경로 생성
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")
    os.system("mkdir -p {}".format(model_prefix))

    # 분산 처리 설정
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # 랜덤 시드 설정
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 로그 파일 생성
    log_path = os.path.join(save_path, "train_log.txt")
    config_logger(log_path, is_master=(rank == 0))
    logger = logging.getLogger(__name__)  # 모든 rank 에서 동일

    # 데이터로더 설정
    train_loader, val_loader, train_sampler = get_dataloaders(pDataset=pDataset, pGen=pGen)

    # 네트워크, 최적화, 스케줄러 설정
    base_net, model, optimizer, scheduler = get_networks_optimizer_scheduler(
        pModel=pModel, pOpt=pOpt, train_loader=train_loader, device=device, local_rank=local_rank
    )

    # 텐서보드 설정
    writer = None
    if rank == 0:
        log_dir = get_next_case_path(os.path.join(save_path, "logs"), tags=[])
        writer = SummaryWriter(log_dir=log_dir)
        logger.info("*" * 80)
        logger.info(
            "Total trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
        )

    # 시작 에포크 설정
    start_epoch = set_starting_condition(
        args, model_prefix, pModel, pOpt, base_net, optimizer, scheduler, rank, logger
    )

    # ***************************************************************************************************** #

    try:
        for epoch in range(start_epoch, pOpt.schedule.end_epoch):
            model.train()
            train_sampler.set_epoch(epoch)

            train_one_epoch(
                epoch,
                pOpt.schedule.end_epoch,
                model,
                train_loader,
                optimizer,
                scheduler,
                logger,
                writer,
                pGen.log_frequency,
            )

            save_checkpoint_and_eval_using_it(
                epoch,
                model,
                optimizer,
                scheduler,
                model_prefix,
                logger,
                pModel,
                val_loader,
                pGen,
                save_path,
                writer,
                rank,
            )

        logger.info(f"학습 완료")

    except KeyboardInterrupt:
        print("Graceful Shutdown...")

    else:
        if rank == 0:
            traceback.print_exc()

    finally:
        if writer is not None:
            writer.close()
        torch.distributed.destroy_process_group()
        logger.info("리소스(CPU, GPU, RAM) 모두 해제되었습니다.")
        sys.exit(0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="lidar segmentation")
    parser.add_argument("--config", help="config file path", type=str)
    parser.add_argument("--start_validating_epoch", type=int, default=0)
    parser.add_argument("--keep_training", action="store_true")
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    main(args, config)
