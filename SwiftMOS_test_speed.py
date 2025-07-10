import argparse
import importlib
import os
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import datasets
from networks import MainNetwork

cudnn.benchmark = True
cudnn.enabled = True


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    # define dataloader
    val_dataset = datasets.data_MOS.DataloadVal(pDataset.Val)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    val_loader = iter(val_loader)

    # define model
    model = MainNetwork.MOSNet(pModel)
    FRAME = 42  # looks good to visualize

    # load pretrained model
    use_pretrained_model = None
    while True:
        print("Using pretrained model is for feature visualization.")
        print("Please check networks/MultiViewNetwork.py if you want to visualize features.")
        use_pretrained_model = input("Use pretrained model? (y/n)").capitalize()
        if use_pretrained_model == "Y":
            break
        elif use_pretrained_model == "N":
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue

    if use_pretrained_model == "Y":
        model_epoch = int(input("Enter the epoch number of the pretrained model: "))
        pretrain_model = os.path.join(model_prefix, "{}-checkpoint.pth".format(model_epoch))
        print("pretrain_model:", pretrain_model)
        model.load_state_dict(torch.load(pretrain_model, map_location="cpu")["model_state_dict"])

    (
        xyzi,
        descartes_coord,
        sphere_coord,
        label_3D,
        label_2D,
        valid_mask_list,
        pad_length_list,
        meta_list_raw,
    ) = val_dataset[FRAME]

    xyzi = xyzi.cuda().unsqueeze(0)
    descartes_coord = descartes_coord.cuda().unsqueeze(0)
    sphere_coord = sphere_coord.cuda().unsqueeze(0)

    arr = label_2D.cpu().numpy().squeeze()
    if not os.path.exists("images/features"):
        os.makedirs("images/features")
    plt.imsave("images/features/label_2D.png", arr, cmap="viridis")

    model.eval()
    model.cuda()
    pred_cls, temporal_res = model.infer(xyzi, descartes_coord, sphere_coord, None)
    time_cost = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(1000)):
            start = time.time()
            pred_cls, temporal_res = model.infer(xyzi, descartes_coord, sphere_coord, temporal_res)
            torch.cuda.synchronize()
            end = time.time()
            print((end - start) * 1000, "ms")
            time_cost.append(end - start)

    print("Time: ", np.array(time_cost[20:]).mean() * 1000, "ms")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="lidar segmentation")
    parser.add_argument("--config", help="config file path", type=str)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    main(args, config)
