from collections import defaultdict
import traceback
import torch

import PIL.Image as Im
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import numpy.linalg as lg

import yaml
import random
import json

import deep_point
from utils.pretty_print import shprint
from . import utils, copy_paste
import os


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(
        pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate
    ).to(pcds_feat.dtype)
    return voxel_feat


def generate_both_bev_labels(c_coord, p_coord, label):
    c_coord = torch.clone(c_coord[:1, :, :2, :])  # 1, 160000, 2, 1
    p_coord = torch.clone(p_coord[:1, :, :2, :])
    label = torch.clone(label).unsqueeze(0).unsqueeze(0)  # 1, 1, 160000, 1
    c_label = (
        VoxelMaxPool(
            pcds_feat=label,
            pcds_ind=c_coord,
            output_size=(256, 256),
            scale_rate=(0.5, 0.5),
        )
        .squeeze(0)
        .squeeze(0)
        .unsqueeze(-1)
    )

    p_label = (
        VoxelMaxPool(
            pcds_feat=label,
            pcds_ind=p_coord,
            output_size=(256, 256),
            scale_rate=(0.5, 0.5),
        )
        .squeeze(0)
        .squeeze(0)
        .unsqueeze(-1)
    )

    return c_label, p_label


def make_point_feat(pcds_xyzi, pcds_coord):
    # make point feat
    x = pcds_xyzi[:, 0].copy()
    y = pcds_xyzi[:, 1].copy()
    z = pcds_xyzi[:, 2].copy()
    intensity = pcds_xyzi[:, 3].copy()

    dist = np.sqrt(x**2 + y**2 + z**2) + 1e-12

    # grid diff
    diff_x = pcds_coord[:, 0] - np.floor(pcds_coord[:, 0])
    diff_y = pcds_coord[:, 1] - np.floor(pcds_coord[:, 1])

    point_feat = np.stack((x, y, z, intensity, dist, diff_x, diff_y), axis=-1)
    return point_feat


class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open("datasets/semantic-kitti.yaml", "r") as f:
            self.task_cfg = yaml.load(f)

        self.cp_aug = None
        if config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste.SequenceCutPaste(
                config.CopyPasteAug.ObjBackDir, config.CopyPasteAug.paste_max_obj_num
            )

        self.aug = utils.DataAugment(
            noise_mean=config.AugParam.noise_mean,
            noise_std=config.AugParam.noise_std,
            theta_range=config.AugParam.theta_range,
            shift_range=config.AugParam.shift_range,
            size_range=config.AugParam.size_range,
        )

        self.aug_raw = utils.DataAugment(
            noise_mean=0, noise_std=0, theta_range=(0, 0), shift_range=((0, 0), (0, 0), (0, 0)), size_range=(1, 1)
        )

        seq_num = config.seq_num
        # add training data
        self.seq_split = [str(i).rjust(2, "0") for i in self.task_cfg["split"]["train"]]
        for seq_id in self.seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, "velodyne")
            fpath_label = os.path.join(fpath, "labels")

            fname_calib = os.path.join(fpath, "calib.txt")
            fname_pose = os.path.join(fpath, "poses.txt")

            calib = utils.parse_calibration(fname_calib)
            poses_list = utils.parse_poses(fname_pose, calib)
            for i in range(len(poses_list)):
                meta_list = []
                meta_list_raw = []
                current_pose_inv = np.linalg.inv(poses_list[i])
                if i < (seq_num - 1):
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                else:
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                self.flist.append((meta_list, meta_list_raw))

        print("Before Training Samples: ", len(self.flist))
        self.remove_few_static_frames()
        print("After Training Samples: ", len(self.flist))

    def remove_few_static_frames(self):
        """
        특정 txt 파일에 기록된 (seq_id, file_id) 프레임만 남기고,
        나머지 정적 프레임(불필요한 프레임)을 제거해 학습 속도를 높이는 함수
        """

        remove_mapping_path = "config/train_split_dynamic_pointnumber.txt"

        # 해당 txt 파일이 없으면 그냥 스킵
        if not os.path.exists(remove_mapping_path):
            print(f"⚠️ {remove_mapping_path} 파일이 없어 제거 과정을 건너뜁니다.")
            return

        # (seq, fid, dynamic_num) 파싱해서 keep_dict에 저장
        with open(remove_mapping_path, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        keep_dict = {}
        for line in lines:
            seq_id, file_id, dynamic_num = line.split()
            if seq_id not in keep_dict:
                keep_dict[seq_id] = set()
            keep_dict[seq_id].add(file_id)

        # 제거 전 길이 기록
        before_len = len(self.flist)

        # 새로운 flist 구성
        new_flist = []
        for meta_list, meta_list_raw in self.flist:
            # meta_list[0]에 (fname_pcd, fname_label, pose_diff, seq_id, file_id)가 있다고 가정
            center_seq_id = meta_list[0][3]
            center_file_id = meta_list[0][4]

            # txt 파일에 해당 (seq_id, file_id)가 있으면 살림
            if center_seq_id in keep_dict and center_file_id in keep_dict[center_seq_id]:
                new_flist.append((meta_list, meta_list_raw))

        # 필터링 완료 후 self.flist에 재할당
        self.flist = new_flist
        after_len = len(self.flist)

        print(f"remove_few_static_frames: {before_len} -> {after_len}")

    def form_batch(self, pcds_total):
        # 1) augment
        pcds_total = self.aug(pcds_total)

        N = pcds_total.shape[0] // self.config.seq_num
        pcds_xyzi = pcds_total[:, :4]

        pcds_coord = utils.Quantize(
            pcds_xyzi,
            range_x=self.Voxel.cart_bev_range_x,
            range_y=self.Voxel.cart_bev_range_y,
            range_z=self.Voxel.cart_bev_range_z,
            size=self.Voxel.cart_bev_shape,
        )

        pcds_polar_coord = utils.PolarQuantize(
            pcds_xyzi,
            range_r=self.Voxel.polar_bev_range_r,
            range_theta=self.Voxel.polar_bev_range_theta,
            range_z=self.Voxel.polar_bev_range_z,
            size=self.Voxel.polar_bev_shape,
        )

        # make feature
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_polar_coord = torch.FloatTensor(pcds_polar_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)

        return pcds_xyzi, pcds_coord, pcds_polar_coord

    def form_seq(self, meta_list):
        pc_list = []
        pc_label_list = []
        pc_raw_label_list = []
        pc_road_list = []
        for ht in range(self.config.seq_num):
            fname_pcd, fname_label, pose_diff, _, _ = meta_list[ht]
            # load pcd
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)
            pc_list.append(pcds_ht)

            # load label
            pcds_label = np.fromfile(fname_label, dtype=np.uint32)
            pcds_label = pcds_label.reshape((-1))
            sem_label = pcds_label & 0xFFFF
            inst_label = pcds_label >> 16

            pc_road_list.append(pcds_ht[sem_label == 40])

            pcds_label_use = utils.relabel(sem_label, self.task_cfg["learning_map"])
            pc_label_list.append(pcds_label_use)
            pc_raw_label_list.append(sem_label)

        return pc_list, pc_label_list, pc_road_list, pc_raw_label_list

    def __getitem__(self, index):
        meta_list, meta_list_raw = self.flist[index]

        # load history pcds
        if random.random() > 0.5:
            pc_list, pc_label_list, pc_road_list, pc_raw_label_list = self.form_seq(meta_list)
        else:
            pc_list, pc_label_list, pc_road_list, pc_raw_label_list = self.form_seq(meta_list_raw)

        # copy-paste
        if self.cp_aug is not None:
            pc_list, pc_label_list = self.cp_aug(pc_list, pc_label_list, pc_road_list, pc_raw_label_list)

        # filter
        for ht in range(len(pc_list)):
            valid_mask_ht = utils.filter_pcds_mask(
                pc_list[ht],
                range_x=self.Voxel.cart_bev_range_x,
                range_y=self.Voxel.cart_bev_range_y,
                range_z=self.Voxel.cart_bev_range_z,
            )
            pc_list[ht] = pc_list[ht][valid_mask_ht]
            pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]

        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length > 0
            pc_list[ht] = np.pad(pc_list[ht], ((0, pad_length), (0, 0)), "constant", constant_values=-1000)
            pc_list[ht][-pad_length:, 2] = -4000

            pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), "constant", constant_values=0)

        pc_list = np.concatenate(pc_list, axis=0)
        label = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)

        xyzi, c_coord, p_coord = self.form_batch(pc_list.copy())
        c_label, p_label = generate_both_bev_labels(c_coord, p_coord, label)

        return (
            xyzi,  # 3, 7, 160000, 1
            c_coord,  # 3, 160000, 3, 1
            p_coord,  # 3, 160000, 3, 1
            label,  # 160000, 1
            c_label,  # 256, 256, 1
            p_label,  # 256, 256, 1
            meta_list_raw,
        )

    def __len__(self):
        return len(self.flist)


class DataloadVal(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open("datasets/semantic-kitti.yaml", "r") as f:
            self.task_cfg = yaml.load(f)

        seq_num = config.seq_num
        # add validation data
        seq_split = [str(i).rjust(2, "0") for i in self.task_cfg["split"]["valid"]]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, "velodyne")
            fpath_label = os.path.join(fpath, "labels")

            fname_calib = os.path.join(fpath, "calib.txt")
            fname_pose = os.path.join(fpath, "poses.txt")

            calib = utils.parse_calibration(fname_calib)
            poses_list = utils.parse_poses(fname_pose, calib)
            for i in range(len(poses_list)):
                meta_list = []
                meta_list_raw = []
                current_pose_inv = np.linalg.inv(poses_list[i])
                if i < (seq_num - 1):
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                else:
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        fname_label = os.path.join(fpath_label, "{}.label".format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                self.flist.append((meta_list, meta_list_raw))

    def form_batch(self, pcds_total):
        N = pcds_total.shape[0] // self.config.seq_num
        pcds_xyzi = pcds_total[:, :4]

        pcds_coord = utils.Quantize(
            pcds_xyzi,
            range_x=self.Voxel.cart_bev_range_x,
            range_y=self.Voxel.cart_bev_range_y,
            range_z=self.Voxel.cart_bev_range_z,
            size=self.Voxel.cart_bev_shape,
        )

        pcds_polar_coord = utils.PolarQuantize(
            pcds_xyzi,
            range_r=self.Voxel.polar_bev_range_r,
            range_theta=self.Voxel.polar_bev_range_theta,
            range_z=self.Voxel.polar_bev_range_z,
            size=self.Voxel.polar_bev_shape,
        )

        # make feature
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_polar_coord = torch.FloatTensor(pcds_polar_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)

        return pcds_xyzi, pcds_coord, pcds_polar_coord

    def form_batch_tta(self, pcds_total):
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_polar_list = []

        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pcds_tmp = pcds_total.copy()
                pcds_tmp[:, 0] *= x_sign
                pcds_tmp[:, 1] *= y_sign

                pcds_xyzi, pcds_coord, pcds_polar = self.form_batch(pcds_tmp)

                pcds_xyzi_list.append(pcds_xyzi)
                pcds_coord_list.append(pcds_coord)
                pcds_polar_list.append(pcds_polar)

        # stack
        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_polar = torch.stack(pcds_polar_list, dim=0)

        return pcds_xyzi, pcds_coord, pcds_polar

    def form_seq(self, meta_list):
        pc_list = []
        pc_label_list = []
        for ht in range(self.config.seq_num):
            fname_pcd, fname_label, pose_diff, _, _ = meta_list[ht]
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)
            pc_list.append(pcds_ht)

            # load label
            pcds_label = np.fromfile(fname_label, dtype=np.uint32)
            pcds_label = pcds_label.reshape((-1))
            sem_label = pcds_label & 0xFFFF
            inst_label = pcds_label >> 16

            pcds_label_use = utils.relabel(sem_label, self.task_cfg["learning_map"])
            pc_label_list.append(pcds_label_use)

        return pc_list, pc_label_list

    def __getitem__(self, index):
        meta_list, meta_list_raw = self.flist[index]

        pc_list, pc_label_list = self.form_seq(meta_list_raw)

        valid_mask_list = []
        for ht in range(len(pc_list)):
            valid_mask_ht = utils.filter_pcds_mask(
                pc_list[ht],
                range_x=self.Voxel.cart_bev_range_x,
                range_y=self.Voxel.cart_bev_range_y,
                range_z=self.Voxel.cart_bev_range_z,
            )
            pc_list[ht] = pc_list[ht][valid_mask_ht]
            pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]
            valid_mask_list.append(valid_mask_ht)

        pad_length_list = []
        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length >= 0
            pc_list[ht] = np.pad(pc_list[ht], ((0, pad_length), (0, 0)), "constant", constant_values=-1000)
            pc_list[ht][-pad_length:, 2] = -4000

            pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), "constant", constant_values=0)
            pad_length_list.append(pad_length)

        pc_list = np.concatenate(pc_list, axis=0)
        label = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)

        xyzi, c_coord, p_coord = self.form_batch_tta(pc_list.copy())
        c_label, p_label = generate_both_bev_labels(c_coord[0], p_coord[0], label)

        return (
            xyzi,
            c_coord,
            p_coord,
            label,
            c_label,
            p_label,
            valid_mask_list,
            pad_length_list,
            meta_list_raw,
        )

    def __len__(self):
        return len(self.flist)
