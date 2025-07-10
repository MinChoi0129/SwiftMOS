# **SwiftMOS: A Fast and Lightweight Moving Object Segmentation via Feature Flowing Direct View Transformation**

This repository offers official SwiftMOS codes.

[<img src="https://img.shields.io/badge/PDF-Paper-red?style=flat&logo=arxiv&logoColor=white" width="90" height="20">](https://arxiv.org/)

![Overview of SwiftMOS Architecture](images/readme/swiftmos.png)

### 1. Basic Environment

We recommend to use PyTorch-CUDA Docker image.
```bash
$ docker pull pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

$ export DISPLAY=***.***.***.***:0 (* : your IP address for visualization)
$ xhost +
$ docker run -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
    -v (path to KITTI dataset in local):(path to KITTI dataset to set in container)
    --privileged \
    --name SwiftMOS \
    --ipc=host \
    --gpus all \
    pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel \
    /bin/bash

$ conda init
$ source ~/.bashrc

$ apt-get update -y
$ apt install -y git vim unzip wget vim git dpkg build-essential
$ apt install -y libgl1-mesa-glx libglib2.0-0 libxcb-cursor0 x11-apps libglib2.0-0

$ conda create -n swiftmos python=3.8
$ conda activate swiftmos
```

### 2. Clone and Install Python Packages

##### 2.1 Clone Repository
```bash
git clone https://github.com/MinChoi0129/SwiftMOS.git
cd SwiftMOS
```

##### 2.2 Install more packages
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install -r requirements.txt

cd deep_point
python setup.py install
```

### 3. Two Datasets

#### 3.1. SemanticKITTI
Please download the [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#overview) dataset to the folder `SemanticKITTI` and the structure of the folder should look like:

```
ROOT_to_SemanticKITTI
└── dataset/
    ├──sequences
        ├── 00/         
        │   ├── velodyne/
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	        └── ...
```

#### 3.2. Object Bank
And download the [object bank](https://drive.google.com/file/d/1QdSpkMLixvKQL6QPircbDI_0-GlGwsdj/view?usp=sharing) on the SemanticKITTI to the folder `object_bank_semkitti` and the structure of the folder should look like:

```
ROOT_to_Object_Bank
├── bicycle
├── bicyclist
├── car
├── motorcycle
├── motorcyclist
├── other-vehicle
├── person
├── truck
```

### 4. Edit Configuration

In `config/config_MOS.py`
* Fill `batch_size_per_gpu` according to your computing resources.
* The SemanticKITTI's `sequence` path should be filled in `SeqDir`(Recommend Absolute Path)
* The path of `Object Bank` should be filled in `ObjBackDir`(Recommend Absolute Path)

In `scripts/train_multi_gpu.sh`
* Fill `CUDA_VISIBLE_DEVICES` and `NumGPUs` according to your computing resources.
* Note : Number of gpus should be same with the length of exported env variable(CUDA_VISIBLE_DEVICES)

### 5. Training / Evaluating / Inference Speed


##### 5.1 Training Session

```bash
bash scripts/train_multi_gpu.sh
```

After every single epoch in the training session, you can see metrics like Moving IoU for validation sequence (08). But, the evaluation process in training session doesn't save the prediction labels for fast training time.

If you want to save the label, you can just run the `5.2 Evaluation Process` below.

##### 5.2 Evaluate Process
This process saves prediction labels. Just comment '--save-label' and remove backslash(\\) above if you don't want to.

Before you run valdation command below, create folder `experiments/config_MOS/checkpoint` first. We provide pretrained SwiftMOS model file as `./50-checkpoint.pth`. Move the `.pth` file into the folder you just created.

```bash
bash scripts/validate.sh
```


##### 5.3 Measuring Average Model Inference Time

```bash
bash scripts/model_infer_speed.sh
```

### 6. Running on On-Board environment
We provide a Dockerfile due to the complex installation process for the Nvidia AGX ORIN NX hardware. This Dockerfile primarily sets up the environment. You will still need to install SwiftMOS properly within this environment.