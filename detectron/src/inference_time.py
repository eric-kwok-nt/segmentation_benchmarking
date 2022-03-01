import cv2
import click
import torch
import torchvision.transforms as T
from time import perf_counter
import logging
from PIL import Image

torch.set_grad_enabled(False)

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Inferencing")


@click.command()
@click.option("--gpu/--no-gpu", default=False)
@click.option(
    "--repo",
    default="facebookresearch/detr",
    show_default=True,
    help="Github repo or a local directory to load the model from",
)
@click.option(
    "--model_name",
    default="detr_resnet101_panoptic",
    show_default=True,
    help="the name of a callable (entrypoint) defined in the repo/dir’s hubconf.py",
)
def main(gpu, repo, model_name):
    setup_start_time = perf_counter()
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # standard PyTorch mean-std input image normalization
    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model, postprocessor = torch.hub.load(
        repo,
        model_name,
        pretrained=True,
        return_postprocessor=True,
        num_classes=250,
    )
    model.eval().to(device)
    setup_end_time = perf_counter()
    logger.info(f"Model setup time: {setup_end_time - setup_start_time:.2f}s")
    # test a video and show the results
    video_paths = [
        "../data/videos/single_person.mp4",
        "../data/videos/multiple_people.mp4",
    ]
    for path in video_paths:
        logger.info(f"Recording time for {model_name}")
        for i in range(3):
            logger.info(f"Run number {i+1}")
            cap = cv2.VideoCapture(path)
            assert cap.isOpened(), "Error opening video stream or file!"
            num_frames = 0
            start_time = perf_counter()
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = Image.fromarray(frame_rgb)
                    img = transform(frame_rgb).unsqueeze(0).to(device)
                    out = model(img)
                    result = postprocessor(
                        out, torch.as_tensor(img.shape[-2:]).unsqueeze(0)
                    )[0]
                    num_frames += 1
                else:
                    break
            end_time = perf_counter()
            fps = num_frames / (end_time - start_time)
            logger.info(f"Average FPS for run {i+1} on {model_name} is {fps:.2f}")
            cap.release()


if __name__ == "__main__":
    main()
