from matplotlib import cm as CM
import scipy.ndimage
import numpy as np
import os
import torch
from torchvision import transforms
import cv2
from glob import glob
import json

import matplotlib.pyplot as plt
from matplotlib import cm

import argparse
from models.fusion import fusion_model
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    sigma = 16
    density += scipy.ndimage.filters.gaussian_filter(gt, sigma, mode='constant')
    return density


def density_map(img, gt):
    k = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(gt)): 
        if gt[i][0] < img.shape[1] and gt[i][1] < img.shape[0]:
            k[int(gt[i][1])][int(gt[i][0])] += 1
    k = gaussian_filter_density(k)
    return k


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--GTpath', default='/root/datasets/bayes-RGBT-CC/train/',)
parser.add_argument('--processGT', default='/root/datasets/bayes-RGBT-CC/train/xxx_GT.npy',
                    help='the path of the image that needs to be visualized')
parser.add_argument('--dataset', default='RGBTCC',
                        help='Choose the dataset: RGBTCC or ShanghaiTechRGBD')
parser.add_argument('--save-dir', default='.../checkpoints/',
                        help='folder path to save the models')
parser.add_argument('--model', default='best_RGBT.pth'
                    , help='model name')
parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

def image_processing(gt_path):
    RGB_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.407, 0.389, 0.396],
            std=[0.241, 0.246, 0.242]),
    ])
    T_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.492, 0.168, 0.430],
            std=[0.317, 0.174, 0.191]),
    ])
    rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
    t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')

    RGB = cv2.imread(rgb_path)[..., ::-1].copy()
    T = cv2.imread(t_path)[..., ::-1].copy()

    keypoints = np.load(gt_path)
    gt = keypoints

    k = np.zeros((T.shape[0], T.shape[1]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < T.shape[0] and int(gt[i][0]) < T.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    target = k

    RGB = RGB_transform(RGB)
    T = T_transform(T)
    name = os.path.basename(gt_path).split('.')[0]

    input = [RGB, T]
    return input, target, name

if __name__ == '__main__':
    input, target, name = image_processing(args.processGT)

    input[0] = torch.unsqueeze(input[0], 0).cuda()
    input[1] = torch.unsqueeze(input[1], 0).cuda()

    model = fusion_model()
    model.cuda()
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    output = model(input, args.dataset).cpu().detach().numpy()

    pre_count = output.sum()
    target_num = target.sum()

    output = output[0][0]
    H, W = target.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio * ratio)

    plt.imshow(output,cmap=cm.jet)
    plt.axis('off')
    plt.savefig('./xxx.jpg', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    print(pre_count)# output the predicted number


    # ## GT density map
    # json_path = args.processGT.replace('bayes-RGBT-CC', 'RGBT-CC').replace('.npy', '.json')
    # rgb_img_path = json_path.replace('.json', '.jpg').replace('_GT','_RGB')
    #
    # with open(json_path, 'r')as f:
    #     gt = json.load(f)
    # img = cv2.imread(rgb_img_path)
    # groundtruth = density_map(img, gt['points'])
    # print(gt['count'])
    # # plt.figure(2)
    #
    # plt.imshow(groundtruth, cmap=CM.jet)
    # plt.axis('off')
    # plt.savefig('./gtxxx.jpg', bbox_inches='tight', pad_inches=0.0)
    # plt.show()





