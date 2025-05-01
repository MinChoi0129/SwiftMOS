import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.pretty_print import shprint
import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.enabled = True


def mapping(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]


def val(epoch, model, val_loader, category_list, save_path, writer, save_label=True):
    criterion_cate = MultiClassMetric(category_list)
    model.eval()

    # 결과 로그를 기록할 파일
    f = open(os.path.join(save_path, "val_log.txt"), "a")
    with torch.no_grad():
        deep_64 = None
        for i, (
            xyzi,
            c_coord,
            p_coord,
            label,
            bev_label,
            valid_mask_list,
            pad_length_list,
            meta_list_raw,
        ) in tqdm.tqdm(enumerate(val_loader)):
            #######################################################################################################
            pred_cls, deep_64 = model.infer(
                xyzi.squeeze(0).cuda(),
                c_coord.squeeze(0).cuda(),
                p_coord.squeeze(0).cuda(),
                deep_64,
            )
            #######################################################################################################
            pred_cls = F.softmax(
                pred_cls[0].squeeze(-1), dim=0
            ).T.contiguous()  # 160000, 3

            label = label[0, :, 0].contiguous()  # 160000,
            criterion_cate.addBatch(label.cpu(), pred_cls.cpu())
            #######################################################################################################

            # --------------------------------#
            if save_label:
                valid_mask = valid_mask_list[0].reshape(-1)

                new_pred_cls = np.zeros((valid_mask_list[0].shape[1]))
                _, pred_cls = torch.max(pred_cls, dim=1)
                pred_cls = pred_cls[: pred_cls.shape[0] - pad_length_list[0][0]]
                new_pred_cls[valid_mask] = pred_cls.cpu().numpy()
                new_pred_cls = new_pred_cls.astype("uint32")

                final_np_prediction = mapping(new_pred_cls, {0: 0, 1: 9, 2: 251})

                seq_id, frame_id = meta_list_raw[0][-2][0], meta_list_raw[0][-1][0]

                prediction_folder_path = os.path.join(
                    save_path,
                    "config_TripleMOS",
                    "results",
                    "sequences",
                    seq_id,
                    "predictions",
                )

                if not os.path.exists(prediction_folder_path):
                    os.makedirs(prediction_folder_path)

                prediction_label_path = os.path.join(
                    prediction_folder_path, frame_id + ".label"
                )
                final_np_prediction.tofile(prediction_label_path)

        #######################################################################################################

        metric_cate = criterion_cate.get_metric()
        string = "Epoch {}".format(epoch)
        for key in metric_cate:
            string += "; {}: {:.4f}".format(key, metric_cate[key])
            if writer:
                writer.add_scalar(f"Eval/{key}", metric_cate[key], epoch)
        print(string)
        f.write(string + "\n")
        f.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")
    model_epoch = args.model_epoch

    # define dataloader
    val_dataset = eval("datasets.{}.DataloadVal".format(pDataset.Val.data_src))(
        pDataset.Val
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=pDataset.Val.num_workers,
        pin_memory=True,
    )

    # define model
    model = eval(pModel.prefix)(pModel)
    model.cuda()
    model.eval()

    pretrain_model = os.path.join(model_prefix, "{}-checkpoint.pth".format(model_epoch))
    print("pretrain_model:", pretrain_model)
    model.load_state_dict(
        torch.load(pretrain_model, map_location="cpu")["model_state_dict"]
    )
    val(
        model_epoch,
        model,
        val_loader,
        pGen.category_list,
        save_path,
        None,
        save_label=args.save_label,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lidar segmentation")
    parser.add_argument("--config", help="config file path", type=str)
    parser.add_argument("--model_epoch", type=int, default=0)
    parser.add_argument("--save_label", default=False, action="store_true")

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    main(args, config)
