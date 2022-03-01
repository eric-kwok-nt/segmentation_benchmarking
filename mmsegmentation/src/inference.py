from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = "configs/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py"
checkpoint_file = "checkpoints/seg_tiny_mask/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth"

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device="cuda:0")

# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # visualize the results in a new window
# model.show_result(img, result, show=True)
# # or save the visualization results to image files
# # you can change the opacity of the painted segmentation map in (0, 1].
# model.show_result(img, result, out_file='result.jpg', opacity=0.5)

# test a video and show the results
video = mmcv.VideoReader("data/videos/single_person.mp4")
for frame in video:
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, wait_time=1, show=True)
