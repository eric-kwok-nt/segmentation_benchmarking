#!/bin/bash

setup_mmseg_dependencies(){
    pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

    pip install mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

    cd mmsegmentation
    pip install -e .
    cd ..

    pip install click
}

setup_mmdet_dependencies() {
    cd mmdetection
    pip install -r requirements/build.txt
    pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
    pip install -v -e .
    cd ..
}

setup_detectron_dependencies() {
    pip install git+https://github.com/cocodataset/panopticapi.git
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    pip install seaborn # Optional
}

setup_mmseg_dependencies
setup_mmdet_dependencies
setup_detectron_dependencies
