import os
import warnings
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
    model.eval()
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    for _ in range(41):
        (
            xyzi,  # [1, 3, 7, 160000, 1]
            descartes_coord,  # [1, 3, 160000, 3, 1]
            sphere_coord,  # [1, 3, 160000, 2, 1]
            label_3D,  # [1, 160000, 1]
            label_2D,  # [1, H, W, 1]
            valid_mask_list,
            pad_length_list,
            meta_list_raw,
        ) = val_loader.next()

    xyzi = xyzi.cuda()
    descartes_coord = descartes_coord.cuda()
    sphere_coord = sphere_coord.cuda()
    label_3D = label_3D.cuda()
    label_2D = label_2D.cuda()

    pred_cls, deep_128_res = model.infer(xyzi, descartes_coord, sphere_coord, None)
    time_cost = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(1000)):
            start = time.time()
            pred_cls, deep_128_res = model.infer(xyzi, descartes_coord, sphere_coord, deep_128_res)
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
