import cv2
import io
import click
import torch
import torchvision.transforms as T
import numpy
from panopticapi.utils import rgb2id
import logging
import itertools
import seaborn as sns
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
def main(gpu):
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
        "facebookresearch/detr",
        "detr_resnet101_panoptic",
        pretrained=True,
        return_postprocessor=True,
        num_classes=250,
    )
    model.eval().to(device)
    # test a video and show the results
    video_paths = [
        "../data/videos/single_person.mp4",
        "../data/videos/multiple_people.mp4",
    ]
    palette = itertools.cycle(sns.color_palette())
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), "Error opening video stream or file!"

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
                # The segmentation is stored in a special-format png
                panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
                panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
                # We retrieve the ids corresponding to each mask
                panoptic_seg_id = rgb2id(panoptic_seg)

                # Finally we color each mask individually
                panoptic_seg[:, :, :] = 0
                for id in range(panoptic_seg_id.max() + 1):
                    panoptic_seg[panoptic_seg_id == id] = (
                        numpy.asarray(next(palette)) * 255
                    )
                # print(panoptic_seg)
                cv2.imshow(
                    "frame",
                    cv2.cvtColor(numpy.asarray(panoptic_seg), cv2.COLOR_RGB2BGR),
                )
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
