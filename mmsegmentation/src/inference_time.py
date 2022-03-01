from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import click
from time import perf_counter
import pathlib
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Inferencing")


@click.command()
@click.option("--gpu/--no-gpu", default=False)
@click.option("--config", required=True, type=str, help="Path to config file")
@click.option("--ckpt", required=True, type=str, help="Path to checkpoint file")
def main(gpu, config, ckpt):
    # config = "configs/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py"
    # ckpt = "checkpoints/seg_tiny_mask/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth"
    setup_start_time = perf_counter()
    if gpu:
        # build the model from a config file and a checkpoint file
        model = init_segmentor(config, ckpt, device="cuda:0")
    else:
        model = init_segmentor(config, ckpt, device="cpu")
    setup_end_time = perf_counter()
    logger.info(f"Model setup time: {setup_end_time - setup_start_time:.2f}s")
    # test a video and show the results
    video_paths = [
        "../data/videos/single_person.mp4",
        "../data/videos/multiple_people.mp4",
    ]
    for path in video_paths:
        video = mmcv.VideoReader(path)
        filename = pathlib.Path(path).name
        logger.info(f"Recording time for {filename}")
        for i in range(3):
            logger.info(f"Run number {i+1}")
            num_frames = 0
            start_time = perf_counter()
            for frame in video:
                result = inference_segmentor(model, frame)
                num_frames += 1
            end_time = perf_counter()
            fps = num_frames / (end_time - start_time)
            logger.info(f"Average FPS for run {i+1} on {filename} is {fps:.2f}")


if __name__ == "__main__":
    main()
