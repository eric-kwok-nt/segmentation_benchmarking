#!/bin/bash

: '
NOTE, run the following shell script in the following format
. run_benchmaeking.sh -f <directory_of_the_library> [-g]
when -g flag is present, it activates the gpu option.
E.g. `. run_benchmaeking.sh -f mmsegmentation -g`
'

# -f is the folder (library) to use
# if -g is present, GPU will be used
GPU='--no-gpu'
FOLDER=''
while getopts 'f:g' 'flag'
do 
    case "${flag}" in
        'f') 
            FOLDER=${OPTARG}
            ;;
        'g') 
            GPU='--gpu'
            echo "GPU option is activate"
            ;;
        '?')
            echo "INVALID OPTION -- ${OPTARG}" >&2
            exit 1
            ;;
        ':')
            echo "MISSING ARGUMENT for option -- ${OPTARG}" >&2
            exit 1
            ;;
        *)
            echo "UNIMPLEMENTED OPTION -- ${flag}" >&2
            exit 1
            ;;
    esac
done

# To reset getopts since getopts uses OPTIND to keep track of the last option argument
OPTIND=1

declare_var() {
    CONFIG_PARENT_DIR="configs/"
    CKPT_PARENT_DIR="checkpoints/"
    if [ $FOLDER == 'mmsegmentation' ]
    then
        declare -ga config_files=(\
            "segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py" \
            "segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k.py"\
        )
        declare -ga checkpoints=(\
            "seg_tiny_mask/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth" \
            "seg_small_mask/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706-511bb103.pth"\
        )
        declare -ga model_name=("Seg-T-Mask-16" "Seg-S-Mask-16")
    elif [ $FOLDER == 'mmdetection' ]
    then
        declare -ga config_files=(\
            "queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco.py" \
            "queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py" \
            "detr/detr_r50_8x2_150e_coco.py"\
        )
        declare -ga checkpoints=(\
            "queryinst_r50_fpn_mstrain_480-800_3x_coco_20210901_103643-7837af86.pth" \
            "queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth" \
            "detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"\
        )
        declare -ga model_name=("R-50-FPN-3x-100" "R-101-FPN-3x-300" "detr_resnet50_panoptic")
    elif [ $FOLDER == 'detectron' ]
    then 
        # repo and model_name corresponds to the 1st 2 arguments of torch.hub.load method
        declare -ga model_name=("detr_resnet101_panoptic")
        declare -ga repo=("facebookresearch/detr")
    else
        echo ${FOLDER}" not found!" >&2
        exit 2  # No such directory
    fi
}

benchmark() {
    arraylength=${#model_name[@]}
    for (( i=0; i<${arraylength}; i++ ));
    do
        echo "Running "${model_name[$i]}
        if [ $FOLDER != 'detectron' ]
        then
            python -m src.inference_time ${GPU} --config ${CONFIG_PARENT_DIR}${config_files[$i]} --ckpt ${CKPT_PARENT_DIR}${checkpoints[$i]} \
            2>&1 | tee -a "logs/"${model_name[$i]}".txt"
        else
            python -m src.inference_time ${GPU} --gpu --repo ${repo[$i]} --model_name ${model_name[$i]}\
            2>&1 | tee -a "logs/"${model_name[$i]}".txt"
        fi
        echo 
        # echo ${GPU}
        # echo ${CONFIG_PARENT_DIR}${config_files[$i]}
        # echo ${CKPT_PARENT_DIR}${checkpoints[$i]}
    done
}

create_logs_folder(){
    # Create logs folder if it doesn't exist
    if [[ ! -d logs ]]
    then
        mkdir logs
        echo "Logs folder created!"
    fi
}

if [ ! -d $FOLDER ] 
then
    echo ${FOLDER}" not found!" >&2
    exit 2  # No such directory
else 
    cd $FOLDER
    declare_var
    create_logs_folder
    benchmark
    cd ..
fi

