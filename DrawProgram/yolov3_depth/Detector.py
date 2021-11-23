import os
import argparse
from sys import platform

import cv2

from yolov3_depth.models import *
from yolov3_depth.utils.datasets import *
from yolov3_depth.utils.utils import *
from yolov3_depth.utils.parse_config import *

class Detector(object):
    def __init__(self, opt):
        self.opt = opt
        self.img_size = self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.weights = self.opt.weights

        # Initialize
        self.device = torch_utils.select_device(device=self.opt.device)

        # Initialize model
        self.model = Darknet(self.opt.cfg, self.img_size)

        # Load weights
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        print("Load", self.weights)

        # Eval mode
        self.model.to(self.device).eval()

        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get classes and colors
        self.classes = load_classes(parse_data_cfg(self.opt.data)['names'])
        print(self.classes)
        self.classes[3] = 'start'
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def img_convert(self, img0):
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    #RawImage(NumpyArray) In, TargetInfo Out
    def detect(self, img0, showResultImg=True):
        # Run inference
        t0 = time.time()

        img = self.img_convert(img0)

        with torch.no_grad():
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres)

        result=[]
        result_img = img0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img0
            result_img = img0
            # 如果检测出了目标 #bys
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, _, cls in det:
                    pred_depth = float(det[0][6].cpu())
                    pred_depth = (pred_depth - 0.5) * 1200 + 877.45
                    result.append(int(cls))
                    result.append(conf)
                    result.append(pred_depth)
                    result.append(xyxy)
                    pred_depth /= 1000
                    label = '%s %.2f %.3fm' % (self.classes[int(cls)], conf, pred_depth)
                    label = '%s %.2f ' % (self.classes[int(cls)], conf)
                    result_img = plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

        if showResultImg:
            cv2.imshow("Result", result_img)
            # Save results (image with detections)

        #print('Done. (%.3fs)' % (time.time() - t0))
        return result, result_img

    def parse_result(self, result):
        target_class = result[0]
        target_conf = float(result[1].cpu())
        target_depth = result[2]
        target_box = result[3]
        target_xmid = int((target_box[0] + target_box[2]) / 2)
        target_ymid = int((target_box[1] + target_box[3]) / 2)
        return target_class, target_conf, target_depth, target_xmid, target_ymid

def args_init(model_type="tinyyolo"):
    parser = argparse.ArgumentParser()
    if model_type == "tinyyolo":
        parser.add_argument('--cfg', type=str, default='yolov3_depth/cfg/yolov3-tinygesture.cfg', help='cfg file path')
    elif model_type == "yolo":
        parser.add_argument('--cfg', type=str, default='yolov3_depth/cfg/yolov3-gesture.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='yolov3_depth/data/gesture.data', help='gesture.data file path')
    if model_type=="tinyyolo":
        parser.add_argument('--weights', type=str, default='yolov3_depth/weights/td957421.pt', help='path to weights file')
    elif model_type=="yolo":
        parser.add_argument('--weights', type=str, default='yolov3_depth/weights/d819999.pt',
                            help='path to weights file')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    return parser

if __name__ == '__main__':
    TEST_RGB_DIR = "E:/bishe2/YOLOv3-complete-pruning-master (2)/data/images/test"
    opt = args_init().parse_args()
    detector = Detector(opt)

    test_img_list = os.listdir(TEST_RGB_DIR)
    for i in range(len(test_img_list)):
        test_img_list[i] = os.path.join(TEST_RGB_DIR,test_img_list[i])

    pred_depths = []

    for test_img_name in test_img_list:
        img = cv2.imread(test_img_name)
        label = detector.detect(img)
        if len(label)!=0:
            pred_depths.append(int(label[2]))
        else:
            pred_depths.append(0)

    pred_depths = np.array(pred_depths,dtype=np.int)

    np.savetxt("output/test_pred_depths.txt", pred_depths, fmt='%d')
