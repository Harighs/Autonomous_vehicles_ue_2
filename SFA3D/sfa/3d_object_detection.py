##########################################################
####  Author: Ariharasudhan Muthusami
####  Email: ariharasudhan.muthusami@gmail.com
####  Date: 28/4/2024
####  Modified from: https://github.com/maudzung/SFA3D
##########################################################


import pickle
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from models.model_utils import create_model
from easydict import EasyDict as edict
from utils.torch_utils import _sigmoid
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
import dataset_tools
from plot_tools import plot_tracks
import cv2
import numpy as np
from kalman_filter import KalmanFilter3D
from io import BytesIO
import matplotlib.pyplot as plt



def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--data_path', type=str,
                        default='/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project2/SFA3D/sfa/bev_maps_v1.pkl',
                        help='the path to the pickle file containing the BEV maps and its images')
    parser.add_argument('--image_path', type=str,
                        default='/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project2/waymo/data_2',
                        help='the path to the pickle file containing the BEV maps and its images')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    return configs


# Defining Color Labels for Detections
COLOR_DICT = {
    "confirmed": "green",
    "tentative": "yellow",
    "initialized": "red"
}

LABEL_DICT = {
    "confirmed": "Confirmed track",
    "tentative": "Tentative track",
    "initialized": "Initialized track",
}


# create a dataloader for testing
class BEVMapDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.bev_maps = pickle.load(f)
        self.keys = list(self.bev_maps.keys())

    def __len__(self):
        return len(self.bev_maps)

    # def __getitem__(self, idx):
    #     key = self.keys[idx]
    #     bev_map = self.bev_maps[key]
    #     image = self.bev_maps[key + '_img']
    #     return torch.tensor(bev_map), key, image

    def __getitem__(self, idx):
        key = self.keys[idx]
        # Check if the key already ends with '_img'
        if key.endswith('_img'):
            image = self.bev_maps[key]
            bev_key = key[:-4]  # Properly get the BEV map key
            bev_map = self.bev_maps[bev_key]
            return_key = bev_key  # return the BEV key without '_img'
        else:
            image_key = key + '_img'
            image = self.bev_maps[image_key]
            bev_map = self.bev_maps[key]
            return_key = key

        return torch.tensor(bev_map), return_key, image


def calculate_iou(boxA, boxB):
    # Unpack the box parameters
    xA, yA, zA, wA, hA, dA, angleA = boxA
    xB, yB, zB, wB, hB, dB, angleB = boxB

    # Determine the coordinates of the intersection rectangle
    xA_max = xA + wA / 2
    xA_min = xA - wA / 2
    yA_max = yA + hA / 2
    yA_min = yA - hA / 2
    zA_max = zA + dA / 2
    zA_min = zA - dA / 2

    xB_max = xB + wB / 2
    xB_min = xB - wB / 2
    yB_max = yB + hB / 2
    yB_min = yB - hB / 2
    zB_max = zB + dB / 2
    zB_min = zB - dB / 2

    # Calculate the edges of the intersection box
    x_min = max(xA_min, xB_min)
    x_max = min(xA_max, xB_max)
    y_min = max(yA_min, yB_min)
    y_max = min(yA_max, yB_max)
    z_min = max(zA_min, zB_min)
    z_max = min(zA_max, zB_max)

    # Check if there is an intersection
    if x_min < x_max and y_min < y_max and z_min < z_max:
        intersection_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    else:
        return 0

    # Calculate the volume of each box
    volumeA = wA * hA * dA
    volumeB = wB * hB * dB

    # Calculate union
    union_volume = volumeA + volumeB - intersection_volume

    # Calculate IoU
    iou = intersection_volume / union_volume
    return iou


def evaluate_sequential_detections(ground_truths, predictions, iou_threshold=0.5):
    true_positives = 0
    ious = []

    # Calculate IoU for overlapping detections
    for i in range(min(len(predictions), len(ground_truths))):
        pred = predictions[i]
        gt = ground_truths[i]
        iou = calculate_iou(pred, gt)
        ious.append(iou)
        if iou >= iou_threshold:
            true_positives += 1

    # If predictions list is longer, the extra predictions are considered false positives
    # and should not contribute to the average IoU calculation
    if len(predictions) > len(ground_truths):
        ious.extend([0] * (len(predictions) - len(ground_truths)))

    # Calculate metrics
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - true_positives
    precision = true_positives / len(predictions) if len(predictions) > 0 else 0
    recall = true_positives / len(ground_truths) if len(ground_truths) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate average IoU, only considering matched detections (excluding padding zeros)
    matched_ious = ious[:min(len(predictions), len(ground_truths))]
    average_iou = np.mean(matched_ious) if matched_ious else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'average_iou': average_iou,  # Calculate average IoU
    }


def scaled_lidar_values(x, y, z, h, w, l, BEV_width, BEV_height, x_lim, y_lim, z_lim, yaw):
    """

    Calculate the LIDAR coordinates and dimensions from BEV detections.

    Parameters:
    X_BEV, Y_BEV : float - BEV coordinates (pixels)
    Z_BEV : float - BEV Z coordinate (shifted ground level accounted)
    H_BEV, W_BEV, L_BEV : float - BEV dimensions (pixels)
    BEV_width, BEV_height : int - dimensions of the BEV image (pixels)
    x_lim, y_lim : list - range limits for x and y in LIDAR ([min, max])
    z_lim : list - range limits for z in LIDAR ([min, max])

    Returns:
    tuple : LIDAR coordinates (X, Y, Z) and dimensions (h, w, l)
    """
    X_lidar = ((y / BEV_height) * (x_lim[1] - x_lim[0]) + x_lim[0])
    Y_lidar = ((x / BEV_width) * (y_lim[1] - y_lim[0]) + y_lim[0])
    Z_lidar = z
    width_lidar = (w / BEV_width) * (y_lim[1] - y_lim[0])
    length_lidar = (l / BEV_height) * (x_lim[1] - x_lim[0])
    height_lidar = h
    return (X_lidar, Y_lidar, Z_lidar, height_lidar, width_lidar, length_lidar, yaw)


class Track:
    def __init__(self, detection):
        self.x = np.asarray(detection[0:3])
        self.h = detection[3]
        self.w = detection[4]
        self.l = detection[5]
        self.yaw = detection[6]
        self.state = "initialized"
        # self.x = np.asarray(detection.pos)
        # self.l = detection.scale[0]
        # self.w = detection.scale[1]
        # self.h = detection.scale[2]
        # self.yaw = detection.rot[2]
        # self.state = "initialized"
        self.id = 0


# Dictionary to transform from cameras ids to numeric ids
ids_to_num = {}
i = 0

if __name__ == '__main__':
    # change the path to the pickle file
    configs = parse_test_configs()

    dataset = BEVMapDataset(configs.data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = create_model(configs)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    # video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 1, (640, 480))

    precision_list = []
    recall_list = []
    f1_score_list = []
    ious_list = []
    y_pred_list = []
    y_true_list = []

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_precision = []
    all_recall = []

    for batch_idx, batch_data in enumerate(dataloader):
        bev_map, img_name, img = batch_data
        # input_bev_map = bev_map.to(configs  .device, non_blocking=True).float()
        input_bev_map = bev_map.to(device=configs.device).float()
        outputs = model(input_bev_map)
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
        detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                            outputs['dim'], K=configs.K)
        detections = detections.detach().cpu().numpy().astype(float)
        detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
        detections = detections[0]  # only first batch
        bev_map = (bev_map.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2RGB)
        bev_map = cv2.resize(bev_map, (configs.output_width, configs.output_width))
        bev_map = draw_predictions(bev_map, detections, configs.num_classes)
        img = img[0].numpy()
        img = cv2.resize(img, (608, 608))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_img = merge_rgb_to_bev(img, bev_map, output_width=configs.output_width)
        img = cv2.resize(img, (configs.output_width, configs.output_width))

        # comment out this part to plot the detections
        #     cv2.imshow('Image', out_img)
        #     cv2.imshow('BEV Map', bev_map)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        stored_detections = []
        for i in range(detections[1].shape[0]):
            x, y, z, h, w, l, yaw = detections[1][i][1:]
            # print('unscaled coords', x, y, z, h, w, l, yaw)
            scaled_coords = scaled_lidar_values(x, y, z, h, w, l, 608, 608,
                                                [0, 50], [-25, 25], [-1.5, 3], yaw)
            stored_detections.append(scaled_coords)
        frame_path = os.path.join(configs.image_path, img_name[0])
        frame = dataset_tools.read_frame(frame_path)
        camera = frame.cameras[0]
        img_arr = dataset_tools.decode_img(camera)
        lidar = frame.lidars[0]

        # comment out this part to plot show the detections with
        tracks = [Track(scaled_coords) for scaled_coords in stored_detections if 0 < scaled_coords[0] < 55]
        plot_tracks(img_arr, tracks, [], lidar.detections, camera)
        # to make the plot visible for every frame
        #plt.show()

        # sort the stored detections based on the x coordinate
        stored_detections.sort(key=lambda x: x[0])

        # remove the detections with x coordinate out of the range
        stored_detections = [detection for detection in stored_detections if 0 < detection[0] < 50]

        lidar_detection_list = []
        for i in range(0, len(lidar.detections)):
            x, y, z, h, w, l, yaw = (lidar.detections[i].pos[0], lidar.detections[i].pos[1],
                                     lidar.detections[i].pos[2], lidar.detections[i].scale[2],
                                     lidar.detections[i].scale[1], lidar.detections[i].scale[0],
                                     lidar.detections[i].rot[2])
            lidar_detection_list.append((x, y, z, h, w, l, yaw))

        # sort the list based on the x coordinate
        lidar_detection_list.sort(key=lambda x: x[0])

        # remove the detections with x coordinate out of the range
        lidar_detection_list = [detection for detection in lidar_detection_list if 0 < detection[0] < 50]

        precision_list.append(evaluate_sequential_detections(lidar_detection_list, stored_detections)['precision'])
        recall_list.append(evaluate_sequential_detections(lidar_detection_list, stored_detections)['recall'])
        f1_score_list.append(evaluate_sequential_detections(lidar_detection_list, stored_detections)['f1_score'])
        ious_list.append(evaluate_sequential_detections(lidar_detection_list, stored_detections)['average_iou'])

        kf = KalmanFilter3D()
        # Set initial conditions if known
        kf.x = np.array([[0], [0], [0], [0], [0], [0]])  # [x, y, z, vx, vy, vz]
        kf.P = np.eye(6) * 1000  # High initial uncertainty

        # Simulated measurements (position only)
        measurements = [
            np.array([[x], [y], [z]]).astype(np.float64) for x, y, z, _, _, _, _ in stored_detections
        ]

        # Process each measurement
        for measurement in measurements:
            _, _ = kf.predict()  # You might only want to use predictions if needed for interim processing
            # print('before update', measurement)
            updated_state, _ = kf.update(measurement)
            # print("Updated State:", updated_state)

    print('Precision:', sum(precision_list) / len(precision_list))
    print('Recall:', sum(recall_list) / len(recall_list))
    print('F1 Score:', sum(f1_score_list) / len(f1_score_list))
    print('Average IoU:', sum(ious_list) / len(ious_list))


    #     # Save the plot to a buffer
    #     buf = BytesIO()
    #     plt.savefig(buf, format='png')
    #     buf.seek(0)
    #
    #     # Load this image back as an OpenCV image (numpy array)
    #     img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    #
    #     video_writer.write(img)
    #
    # video_writer.release()
