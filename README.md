# **TripleMOS: Moving Object Segmentation based on Multiple Coordinate Systems**

Official code for TripleMOS


### 1. Basic Environment

We recommend to use the docker hub's PyTorch-Cuda image.
```bash
$ docker pull pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

$ export DISPLAY=***.***.***.***:0 (* should be your IP address.)
$ xhost +
$ docker run -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
    -v (path to KITTI dataset in local):(path to KITTI dataset to set in container)
    -v /dev:/dev:ro \
    --privileged \
    --name TripleMOS \
    --ipc=host \
    --gpus all \
    pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel \
    /bin/bash

$ conda init
$ source ~/.bashrc

$ apt-get update -y
$ apt install -y git vim unzip wget vim git dpkg build-essential
$ apt-get install -y libgl1-mesa-glx libglib2.0-0 libxcb-cursor0 x11-apps

$ conda create -n triple python=3.8
$ conda activate triple
```

### 2. Clone and Install Python Packages

##### 2.1 Clone Repository
```bash
git clone https://github.com/MinChoi0129/TripleMOS.git
cd TripleMOS
```

##### 2.2 Install more packages
```bash
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

In `config/config_TripleMOS.py`
* Fill `batch_size_per_gpu` according to your computing resources.
* The SemanticKITTI's `sequence` path should be filled in `SeqDir`(Recommend Absolute Path)
* The path of `Object Bank` should be filled in `ObjBackDir`(Recommend Absolute Path)

In `scripts/eval_for_paper.sh`
* Change `DatasetPath` properly to fit your environment.(Should be Absolute Path)

In `scripts/train_multi_gpu.sh`
* Fill `CUDA_VISIBLE_DEVICES` and `NumGPUs` according to your computing resources.
* Warning : Number of gpus should be same with the length of exported env variable(CUDA_VISIBLE_DEVICES)

### 5 Training / Evaluating / Inference Speed

After every single epoch in the training session, you can check the various metrics like Moving IOU about validation sequence(08).

But, the evaluation process in training session, TripleMOS doesn't save the prediction labels for fast training time(No R/W).

If you want to save the label, you can just run the `5.2 Evaluation Process`.

##### 5.1 Training Session

```bash
bash scripts/train_multi_gpu.sh
```

##### 5.2 Evaluate Process(save prediction labels)

```bash
bash scripts/validate_multi_gpu_save_label.sh
bash eval_for_paper.sh
```

##### 5.3 Measuring Average Model Inference Time

```bash
bash scripts/model_infer_speed.sh
```
