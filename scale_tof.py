import cv2
import argparse
from test_image import GetImages
from test_image import DATA_TYPE
import os
from file import Walk, MkdirSimple
import numpy as np

def GetArgs():
    parser = argparse.ArgumentParser(description='input depth and tof, to scale depth same scale to tof')
    parser.add_argument('--depth_path', type=str, required=True)
    parser.add_argument('--tof_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="dir saves scaled depth image")
    parser.add_argument('--scale', type=float, default=1, help="scale that depth data need multi to be real depth")
    parser.add_argument("--width",type=str, default=None)
    parser.add_argument("--height",type=str, default=None)
    parser.add_argument("--scale_distance",type=int, default=None)
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")

    args = parser.parse_args()

    return args

def WriteDepth(predict_np, path, name):
    name = os.path.splitext(name)[0] + ".png"
    output_scale = os.path.join(path, "scale_tof", name)

    MkdirSimple(output_scale)

    cv2.imwrite(output_scale, predict_np)
    return

def get_boundary(image, width, height):
    assert image.shape[0] >= int(height), "original image height is small, can't crop"
    assert image.shape[1] >= int(width), "original image width is small, can't crop"

    height_crop = image.shape[0] - int(height)
    width_crop = image.shape[1] - int(width)
    left = round(width_crop // 2)
    right = image.shape[1] - (width_crop - left)
    assert right - left == width

    top = round(height_crop // 2)
    bottom = image.shape[0] - (height_crop - top)
    assert bottom - top == height

    return left, right, top, bottom

def tof_scale_depth(disp_file, tof_file, scale=1.0, width=None, height=None, scale_distance=None, bf = 14.2):
    disp_img = cv2.imread(disp_file, -1)
    disp_img = disp_img/256
    depth_img = bf /disp_img * 100

    tof_img = cv2.imread(tof_file, -1)

    # tof_img = tof_img[:, :, 0] + (tof_img[:, :, 1] > 0) * 255 + tof_img[:, :, 1] + (
    #             tof_img[:, :, 2] > 0) * 511 + tof_img[:, :, 2]

    if height is not None and width is not None:
        left, right, top, bottom = get_boundary(tof_img, width=int(width), height=int(height))
        tof_img = tof_img[top: bottom, left: right]
    tof_img = tof_img * scale

    real_depth = depth_img.copy()

    mask_tof = tof_img > 0
    if scale_distance is not None:
        mask_tof =  (tof_img > 0) & (tof_img < scale_distance * scale)

    mask_depth = real_depth > 0
    mask = mask_tof * mask_depth
    mask_ratio = tof_img[mask] / real_depth[mask]
    median_ratio = np.median(mask_ratio)
    depth_save = depth_img * median_ratio

    disp_save = bf/depth_save *100
    disp_save = disp_save * 256.0
    disp_save[disp_save > 65535] = 65535
    disp_save[disp_save < 0] = 0
    disp_save = disp_save.astype(np.uint16)

    print("median_ratio: {}".format(median_ratio))
    return disp_save

def main():
    args = GetArgs()

    root_len = len(args.depth_path)

    for k in DATA_TYPE:
        depth_files, _, _ = GetImages(args.depth_path, k)
        if len(depth_files) != 0:
            break
    for k in DATA_TYPE:
        tof_files, _, _ = GetImages(args.tof_path, k)
        if len(tof_files) != 0:
            break
    depth_files.sort()
    tof_files.sort()
    for depth_file, tof_file in zip(depth_files, tof_files):
        output_name = depth_file[root_len+1:]
        assert depth_file.split('/')[-1].split('/')[0] == tof_file.split('/')[-1].split('/')[0]\
            , "assert same image depth: {} with tof: {}".format(depth_file, tof_file)

        depth_save = tof_scale_depth(depth_file, tof_file, args.scale, args.width, args.height, args.scale_distance, args.bf)

        WriteDepth(depth_save, args.output, output_name)

if __name__ == '__main__':
    main()
