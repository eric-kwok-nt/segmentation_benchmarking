from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = "configs/queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco.py"
checkpoint_file = (
    "checkpoints/queryinst_r50_fpn_mstrain_480-800_3x_coco_20210901_103643-7837af86.pth"
)

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # visualize the results in a new window
# model.show_result(img, result, show=True)
# # or save the visualization results to image files
# # you can change the opacity of the painted segmentation map in (0, 1].
# model.show_result(img, result, out_file='result.jpg', opacity=0.5)

# test a video and show the results
video = mmcv.VideoReader("../data/videos/single_person.mp4")
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1, show=True)
