import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
import time
from torch.utils.data import DataLoader
import tqdm
import importlib
import torch.backends.cudnn as cudnn
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
    model_epoch = 60
    FRAME = 42
    pretrain_model = os.path.join(model_prefix, "{}-checkpoint.pth".format(model_epoch))
    print("pretrain_model:", pretrain_model)
    model.load_state_dict(torch.load(pretrain_model, map_location="cpu")["model_state_dict"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    model.eval()
    model.cuda()

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

    print(xyzi.shape, descartes_coord.shape, sphere_coord.shape)

    # label_2D : [256, 256, 1] -> viridis 컬러맵으로 저장
    arr = label_2D.cpu().numpy().squeeze()  # shape: [256, 256]
    if not os.path.exists("images/features"):
        os.makedirs("images/features")
    plt.imsave("images/features/label_2D.png", arr, cmap="viridis")

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
