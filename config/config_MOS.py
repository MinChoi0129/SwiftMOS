def get_config():
    class General:
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_size_per_gpu = 3

        SeqDir = "/home/workspace/KITTI/dataset/sequences"
        category_list = ["static", "moving"]

        loss_mode = "ohem"
        K = 2

        class Voxel:
            # 해상도
            descartes_shape = (512, 512, 30)
            sphere_shape = (64, 2048, 30)
            cylinder_shape = (64, 2048, 30)
            polar_shape = (64, 2048, 30)

            # 데이터 범위
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-4.0, 2.0)
            range_r = (2, 50)
            range_phi = (-180, 180)
            range_theta = (-25.0, 3.0)

    class DatasetParam:
        class Train:
            num_workers = 4
            frame_point_num = 160000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            seq_num = General.K + 1

            class CopyPasteAug:
                is_use = True
                ObjBackDir = "/home/workspace/KITTI/object_bank_semkitti"
                paste_max_obj_num = 20

            class AugParam:
                noise_mean = 0
                noise_std = 0.0001
                theta_range = (-180.0, 180.0)
                shift_range = ((-3, 3), (-3, 3), (-0.4, 0.4))
                size_range = (0.95, 1.05)

        class Val:
            num_workers = 3
            frame_point_num = 160000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            seq_num = General.K + 1

        class Test:
            num_workers = 3
            frame_point_num = 160000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            seq_num = General.K + 1

    class ModelParam:
        Voxel = General.Voxel
        category_list = General.category_list
        class_num = len(category_list) + 1
        loss_mode = General.loss_mode
        seq_num = General.K + 1
        fusion_mode = "CatFusion"
        point_feat_out_channels = 64

        class BEVParam:
            base_block = "BasicBlock"
            context_layers = [64, 32, 64, 128]
            layers = [2, 3, 4]
            bev_grid2point = dict(type="BilinearSample", scale_rate=(0.5, 0.5))

        class RVParam:
            base_block = "BasicBlock"
            context_layers = [64, 32, 64, 128]
            layers = [2, 3, 4]
            rv_grid2point = dict(type="BilinearSample", scale_rate=(1.0, 0.5))

        class pretrain:
            pretrain_epoch = 50  # 이 숫자까지 학습했다고 가정함. 즉 +1 한 Epoch을 이어서 시작할 것임.

    class OptimizeParam:
        class optimizer:
            type = "sgd"
            base_lr = 0.02
            momentum = 0.9
            nesterov = True
            wd = 1e-3

        class schedule:
            type = "step"
            begin_epoch = 0
            end_epoch = 80
            pct_start = 0.01
            final_lr = 1e-6
            step = 10
            decay_factor = 0.1

    return General, DatasetParam, ModelParam, OptimizeParam
