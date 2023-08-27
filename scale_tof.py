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

    args = parser.parse_args()

    return args

def WriteDepth(predict_np, path, name):
    name = os.path.splitext(name)[0] + ".png"
    output_scale = os.path.join(path, "scale_tof", name)

    MkdirSimple(output_scale)

    cv2.imwrite(output_scale, predict_np)
    return

def tof_scale_depth(depth_file, tof_file, scale=1.0):
    depth_img = cv2.imread(depth_file, -1)
    tof_img = cv2.imread(tof_file, -1)

    tof_img = tof_img[:, :, 0] + (tof_img[:, :, 1] > 0) * 255 + tof_img[:, :, 1] + (
                tof_img[:, :, 2] > 0) * 511 + tof_img[:, :, 2]

    real_depth = depth_img / scale

    mask_tof = tof_img > 0
    mask_depth = real_depth > 0
    mask = mask_tof * mask_depth
    mask_ratio = tof_img[mask] / real_depth[mask]
    median_ratio = np.median(mask_ratio)
    depth_save = depth_img * median_ratio

    depth_save[depth_save > 65535] = 65535
    depth_save[depth_save < 0] = 0
    depth_save = depth_save.astype(np.uint16)
    print("median_ratio: {}".format(median_ratio))
    return depth_save

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

        depth_save = tof_scale_depth(depth_file, tof_file, args.scale)

        WriteDepth(depth_save, args.output, output_name)

if __name__ == '__main__':
    main()
