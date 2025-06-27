import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import deep_point
from . import utils, copy_paste
import os
import random


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(
        pcds_feat=pcds_feat.float(),
        pcds_ind=pcds_ind,
        output_size=output_size,
        scale_rate=scale_rate,
    ).to(pcds_feat.dtype)
    return voxel_feat


def generate_img_labels(coord, label, size):
    coord = torch.clone(coord[:1, :, :2, :])  # 1, 160000, 2, 1
    label = torch.clone(label).unsqueeze(0).unsqueeze(0)  # 1, 1, 160000, 1
    img_label = (
        VoxelMaxPool(
            pcds_feat=label,
            pcds_ind=coord,
            output_size=size,
            scale_rate=(0.5, 0.5),
        )
        .squeeze(0)
        .squeeze(0)
        .unsqueeze(-1)
    )

    return img_label  # H, W, 1


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


other_mode = "sphere"


class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open("datasets/semantic-kitti.yaml", "r") as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.cp_aug = None
        if config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste.SequenceCutPaste(config.CopyPasteAug.ObjBackDir, config.CopyPasteAug.paste_max_obj_num)

        self.aug = utils.DataAugment(
            noise_mean=config.AugParam.noise_mean,
            noise_std=config.AugParam.noise_std,
            theta_range=config.AugParam.theta_range,
            shift_range=config.AugParam.shift_range,
            size_range=config.AugParam.size_range,
        )

        self.aug_raw = utils.DataAugment(
            noise_mean=0,
            noise_std=0,
            theta_range=(0, 0),
            shift_range=((0, 0), (0, 0), (0, 0)),
            size_range=(1, 1),
        )

        seq_num = config.seq_num + 2
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

        rank = int(os.environ["LOCAL_RANK"])
        if rank == 0:
            print("[Info] Before Training Samples: ", len(self.flist))
        self.remove_few_static_frames()
        if rank == 0:
            print("[Info] Static-Reduced Training Samples: ", len(self.flist))

        # 데이터 샘플링으로 flist 크기 줄이기
        # self.sample_flist()
        # if rank == 0:
        #     print("[Info] After Sampling Training Samples: ", len(self.flist))

    def remove_few_static_frames(self):
        remove_mapping_path = "config/train_split_dynamic_pointnumber.txt"

        if not os.path.exists(remove_mapping_path):
            print(f"{remove_mapping_path} 파일이 없어 제거 과정을 건너뜁니다.")
            return

        with open(remove_mapping_path, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        keep_dict = {}
        for line in lines:
            seq_id, file_id, dynamic_num = line.split()
            if seq_id not in keep_dict:
                keep_dict[seq_id] = set()
            keep_dict[seq_id].add(file_id)

        new_flist = []
        for meta_list, meta_list_raw in self.flist:
            center_seq_id = meta_list[0][3]
            center_file_id = meta_list[0][4]

            if center_seq_id in keep_dict and center_file_id in keep_dict[center_seq_id]:
                new_flist.append((meta_list, meta_list_raw))

        self.flist = new_flist

    def sample_flist(self):
        """flist를 샘플링해서 크기를 줄이는 메서드"""
        # 설정에서 샘플링 비율 가져오기
        sample_ratio = 0.5

        original_size = len(self.flist)
        target_size = int(original_size * sample_ratio)

        # 랜덤 샘플링
        sampled_indices = random.sample(range(original_size), target_size)

        # 샘플링된 인덱스로 새로운 flist 생성
        new_flist = [self.flist[i] for i in sampled_indices]
        self.flist = new_flist

        rank = int(os.environ["LOCAL_RANK"])
        if rank == 0:
            print(f"데이터 샘플링 완료: {original_size} → {len(self.flist)} (비율: {sample_ratio:.1%})")

    def form_batch(self, pcds_total):
        pcds_total = self.aug(pcds_total)

        N = pcds_total.shape[0] // self.config.seq_num
        pcds_xyzi = pcds_total[:, :4]

        pcds_descartes_coord = utils.Quantize(
            pcds_xyzi,
            range_x=self.Voxel.range_x,
            range_y=self.Voxel.range_y,
            range_z=self.Voxel.range_z,
            size=self.Voxel.descartes_shape,
        )

        pcds_sphere_coord = utils.SphereQuantize(
            pcds_xyzi,
            phi_range=self.Voxel.range_phi,
            theta_range=self.Voxel.range_theta,
            r_range=self.Voxel.range_r,
            size=self.Voxel.sphere_shape,
        )

        # make feature
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_descartes_coord)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_descartes_coord = torch.FloatTensor(pcds_descartes_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)

        return pcds_xyzi, pcds_descartes_coord, pcds_sphere_coord

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
        xyzi_stages = []
        descartes_coord_stages = []
        sphere_coord_stages = []
        label_3D_stages = []
        label_2D_stages = []
        meta_list_raw_stages = []

        for idx in [index, index - 1, index - 2]:
            meta_list, meta_list_raw = self.flist[idx]

            # load history pcds
            pc_list, pc_label_list, pc_road_list, pc_raw_label_list = self.form_seq(meta_list_raw)

            # copy-paste
            if self.cp_aug is not None:
                pc_list, pc_label_list = self.cp_aug(pc_list, pc_label_list, pc_road_list, pc_raw_label_list)

            # filter
            for ht in range(len(pc_list)):
                valid_mask_ht = utils.filter_pcds_mask(
                    pc_list[ht],
                    range_x=self.Voxel.range_x,
                    range_y=self.Voxel.range_y,
                    range_z=self.Voxel.range_z,
                )
                pc_list[ht] = pc_list[ht][valid_mask_ht]
                pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]

            for ht in range(len(pc_list)):
                pad_length = self.frame_point_num - pc_list[ht].shape[0]
                assert pad_length > 0
                pc_list[ht] = np.pad(
                    pc_list[ht],
                    ((0, pad_length), (0, 0)),
                    "constant",
                    constant_values=-1000,
                )
                pc_list[ht][-pad_length:, 2] = -4000

                pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), "constant", constant_values=0)

            pc_list = np.concatenate(pc_list, axis=0)

            # [3, 7, 160000, 1], [3, 160000, 3, 1], [3, 160000, 2, 1]
            xyzi, descartes_coord, sphere_coord = self.form_batch(pc_list.copy())
            label_3D = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)
            label_2D = generate_img_labels(descartes_coord, label_3D, size=(256, 256))

            xyzi_stages.append(xyzi)
            descartes_coord_stages.append(descartes_coord)
            sphere_coord_stages.append(sphere_coord)
            label_3D_stages.append(label_3D)
            label_2D_stages.append(label_2D)
            meta_list_raw_stages.append(meta_list_raw)

        xyzi_stages = torch.stack(xyzi_stages, dim=0)
        descartes_coord_stages = torch.stack(descartes_coord_stages, dim=0)
        sphere_coord_stages = torch.stack(sphere_coord_stages, dim=0)
        label_3D_stages = torch.stack(label_3D_stages, dim=0)
        label_2D_stages = torch.stack(label_2D_stages, dim=0)
        meta_list_raw_stages = meta_list_raw_stages[0]

        return (
            xyzi_stages,  # [Stage, 3, 7, 160000, 1]
            descartes_coord_stages,  # [Stage, 3, 160000, 3, 1]
            sphere_coord_stages,  # [Stage, 3, 160000, 2, 1]
            label_3D_stages,  # [Stage, 160000, 1]
            label_2D_stages,  # [Stage, 32, 1024, 1]
            meta_list_raw_stages,
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
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

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

        pcds_descartes_coord = utils.Quantize(
            pcds_xyzi,
            range_x=self.Voxel.range_x,
            range_y=self.Voxel.range_y,
            range_z=self.Voxel.range_z,
            size=self.Voxel.descartes_shape,
        )

        pcds_sphere_coord = utils.SphereQuantize(
            pcds_xyzi,
            phi_range=self.Voxel.range_phi,
            theta_range=self.Voxel.range_theta,
            r_range=self.Voxel.range_r,
            size=self.Voxel.sphere_shape,
        )

        # make feature
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_descartes_coord)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_descartes_coord = torch.FloatTensor(pcds_descartes_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)

        return pcds_xyzi, pcds_descartes_coord, pcds_sphere_coord

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
                range_x=self.Voxel.range_x,
                range_y=self.Voxel.range_y,
                range_z=self.Voxel.range_z,
            )
            pc_list[ht] = pc_list[ht][valid_mask_ht]
            pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]
            valid_mask_list.append(valid_mask_ht)

        pad_length_list = []
        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length >= 0
            pc_list[ht] = np.pad(
                pc_list[ht],
                ((0, pad_length), (0, 0)),
                "constant",
                constant_values=-1000,
            )
            pc_list[ht][-pad_length:, 2] = -4000

            pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), "constant", constant_values=0)
            pad_length_list.append(pad_length)

        pc_list = np.concatenate(pc_list, axis=0)

        xyzi, descartes_coord, sphere_coord = self.form_batch(pc_list.copy())
        label_3D = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)
        label_2D = generate_img_labels(descartes_coord, label_3D, size=(256, 256))

        return (
            xyzi,  # [3, 7, 160000, 1]
            descartes_coord,  # [3, 160000, 3, 1]
            sphere_coord,  # [3, 160000, 2, 1]
            label_3D,  # [160000, 1]
            label_2D,  # [32, 1024, 1]
            valid_mask_list,
            pad_length_list,
            meta_list_raw,
        )

    def __len__(self):
        return len(self.flist)


class DataloadTest(Dataset):
    def __init__(self, config, seq):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open("datasets/semantic-kitti.yaml", "r") as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        seq_num = config.seq_num
        # add validation data
        seq_split = [seq]

        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, "velodyne")
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
                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list.append((fname_pcd, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, pose_diff, seq_id, file_id))
                else:
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, pose_diff, seq_id, file_id))

                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, "0")
                        fname_pcd = os.path.join(fpath_pcd, "{}.bin".format(file_id))
                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, pose_diff, seq_id, file_id))

                self.flist.append((meta_list, meta_list_raw))

    def form_batch(self, pcds_total):
        N = pcds_total.shape[0] // self.config.seq_num
        pcds_xyzi = pcds_total[:, :4]

        pcds_descartes_coord = utils.Quantize(
            pcds_xyzi,
            range_x=self.Voxel.range_x,
            range_y=self.Voxel.range_y,
            range_z=self.Voxel.range_z,
            size=self.Voxel.descartes_shape,
        )

        pcds_sphere_coord = utils.SphereQuantize(
            pcds_xyzi,
            phi_range=self.Voxel.range_phi,
            theta_range=self.Voxel.range_theta,
            r_range=self.Voxel.range_r,
            size=self.Voxel.sphere_shape,
        )

        # make feature
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_descartes_coord)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_descartes_coord = torch.FloatTensor(pcds_descartes_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)

        return pcds_xyzi, pcds_descartes_coord, pcds_sphere_coord

    def form_seq(self, meta_list):
        pc_list = []
        for ht in range(self.config.seq_num):
            fname_pcd, pose_diff, _, _ = meta_list[ht]
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)
            pc_list.append(pcds_ht)

        return pc_list

    def __getitem__(self, index):
        meta_list, meta_list_raw = self.flist[index]

        pc_list = self.form_seq(meta_list_raw)

        valid_mask_list = []
        for ht in range(len(pc_list)):
            valid_mask_ht = utils.filter_pcds_mask(
                pc_list[ht],
                range_x=self.Voxel.range_x,
                range_y=self.Voxel.range_y,
                range_z=self.Voxel.range_z,
            )
            pc_list[ht] = pc_list[ht][valid_mask_ht]
            valid_mask_list.append(valid_mask_ht)

        pad_length_list = []
        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length >= 0
            pc_list[ht] = np.pad(
                pc_list[ht],
                ((0, pad_length), (0, 0)),
                "constant",
                constant_values=-1000,
            )
            pc_list[ht][-pad_length:, 2] = -4000
            pad_length_list.append(pad_length)

        pc_list = np.concatenate(pc_list, axis=0)

        xyzi, descartes_coord, sphere_coord = self.form_batch(pc_list.copy())

        return (
            xyzi,  # [3, 7, 160000, 1]
            descartes_coord,  # [3, 160000, 3, 1]
            sphere_coord,  # [3, 160000, 2, 1]
            valid_mask_list,
            pad_length_list,
            meta_list_raw,
        )

    def __len__(self):
        return len(self.flist)
