import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.multiprocessing import Pool, Pipe
from torch import multiprocessing

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from mot.defaults import _C as cfg
from mot.mot_sde_model import build_model, SDE_ReID
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, cv2, remove_little_box,
                           increment_path, non_max_suppression, print_args, scale_boxes)
from utils.plots import plot_tracking
from utils.torch_utils import select_device, smart_inference_mode


def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

    pass


class PredictCam:
    def __init__(self):
        self.model = None
        self.reid_model = None
        self.dataset = None
        self.recv = None
        self.send = None

    def run_single_cam(self, weights, source, hide_conf, data, fp16, save_dir, recv, send, img_sz, vid_stride,
                       conf_thres, iou_thres, classes, agnostic_nms, max_det, device, crop, i):
        self.recv = recv
        self.send = send
        if self.model is None:
            self.model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=fp16)
        if self.reid_model is None:
            if fp16:
                reid_model = build_model(cfg, device, 8, fp16).half().to(device)
            else:
                reid_model = build_model(cfg, device, 8, fp16).to(device)
            reid_model.load_state_dict(torch.load("./mot/weights/resnet18_model_90.pth"))  # 加载训练模型权重
            self.reid_model = SDE_ReID(reid_model, mtmct=True)

        if self.dataset is None:
            # self.dataset = LoadStreams("rtsp://127.0.0.1:554/video1", img_size=img_sz, stride=64, auto=True,
            #                            vid_stride=vid_stride, crop=crop[i])
            self.dataset = LoadImages(ROOT / source[i], img_size=img_sz, stride=64,
                                      auto=True, vid_stride=vid_stride, crop=crop[i])
        self.predict_cam(save_dir, hide_conf, conf_thres, iou_thres, classes, agnostic_nms, max_det, device, crop, i)

    @smart_inference_mode()
    def predict_cam(self, save_dir, hide_conf, conf_thres, iou_thres, classes, agnostic_nms, max_det, device, crop, i):
        seen = 0
        save_path = "./" + str(save_dir / (str(i + 1) + ".mp4")).replace("\\", "/")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (3072, 1728))
        executor = ThreadPoolExecutor(max_workers=2048)

        t0 = time.time()
        for cam_info in self.dataset:
            t1 = time.time()
            path, im, im0s, vid_cap, s = cam_info
            im = torch.from_numpy(im).to(device)
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            pred = self.model(im)
            # LOGGER.info(f"yolo ::  \t{time.time() - t1}")
            t2 = time.time()
            pred_det = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            # LOGGER.info(f"nms ::  \t{time.time() - t2}")

            seen += 1
            imc = im0s.copy()
            # s += '%gx%g ' % im.shape[2:]  # print string
            if len(pred_det):  # 这里的坐标是xyxy
                if crop:
                    pred_det[:, :4] = scale_boxes(im.shape[2:], pred_det[:, :4],
                                                  [crop[i][1][1] - crop[i][0][1],
                                                   crop[i][1][0] - crop[i][0][0]]).round()
                    pred_det[..., [0, 2]] = pred_det[..., [0, 2]] + crop[i][0][0]  # x1, x2
                    pred_det[..., [1, 3]] = pred_det[..., [1, 3]] + crop[i][0][1]  # y1, y2
                else:
                    pred_det[:, :4] = scale_boxes(im.shape[2:], pred_det[:, :4], imc.shape).round()
            pred_det = remove_little_box(pred_det, 50, 50)
            crops = self.reid_model.get_crops(pred_det[:, :4], imc)
            # LOGGER.info(f"nms and crop img:  \t{time.time() - t2}")
            t3 = time.time()
            tracking_outs = self.reid_model.predict(crops, pred_det, i, device, frame_id=seen, seq_name="1")
            # LOGGER.info(f"get feature and track:  \tn{time.time() - t3}")
            t4 = time.time()
            if self.recv is not None:
                last_predict = self.recv.recv()
            else:
                last_predict = None
            tracking_outs = self.reid_model.cam_match(tracking_outs, i, last_predict)
            if self.send is not None:
                self.send.send(tracking_outs)  # 每个镜头都需要向下一个镜头发送

            executor.submit(
                lambda: self.plot_save_result(tracking_outs, hide_conf, save_path, vid_writer, seen, cam_info, i))
            # LOGGER.info(f"send and recv info submit thread:  \t{time.time() - t4}")

            LOGGER.info(f"deal img total time cam id:%d:  \t{time.time() - t1}" % i)
        print("处理视频总共用时：", time.time() - t0)

    def plot_save_result(self, tracking_outs, hide_conf, save_path, vid_writer, seen, cam_info, i):
        path, im, im0s, vid_cap, s = cam_info
        online_tlwhs = tracking_outs['online_tlwhs']
        online_scores = tracking_outs['online_scores']
        online_ids = tracking_outs['online_ids']
        online_im = plot_tracking(im0s, online_tlwhs, online_ids, online_scores, seen, hide_conf)  # 将跟踪对象画到image中

        if save_path is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            save_path = "./mot/output/" + str(i + 1) + "/" + str(i + 1) + ".mp4"
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
        vid_writer.write(online_im)


def print_error(value):
    print("error:", value)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型存放的地址
        source=None,  # 测试图片存放地址
        data=ROOT / 'data/my_data.yaml',  # dataset.yaml path
        img_sz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cuda:0',
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        project=ROOT / 'runs/mot_infer',  # 存放检测结果的path
        name='exp',  # save results to project/name
        exist_ok=False,  # 是否已经创建好存放地址，为True，则不会重新创建文件夹。
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        vid_stride=1,  # 视频跳几帧进行识别
        crop=None,
):
    # 加载检测模型
    if source is None:
        source = []
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True)
    device = select_device(device)
    multiprocessing.log_to_stderr()
    pool = Pool(3)
    img_sz = check_img_size(img_sz, s=64)  # check image size

    cam_list = [PredictCam(), PredictCam(), PredictCam()]
    pipe0 = Pipe(duplex=False)
    pipe1 = Pipe(duplex=False)
    pipe2 = Pipe(duplex=False)
    pipe_list = [[None, pipe0[1]], [pipe0[0], pipe1[1]], [pipe1[0], None]]
    # cam_list[0].run_single_cam(weights, source, hide_conf, data, half, save_dir, pipe_list[0][0], pipe_list[0][1],
    #                            img_sz,
    #                            vid_stride, conf_thres, iou_thres, classes, agnostic_nms, max_det, device, crop, 0)

    for i in range(3):
        pool.apply_async(LogExceptions(cam_list[i].run_single_cam),
                         args=(weights, source, hide_conf, data, half, save_dir, pipe_list[i][0], pipe_list[i][1],
                               img_sz, vid_stride, conf_thres, iou_thres, classes, agnostic_nms, max_det, device, crop,
                               i,), error_callback=print_error)

    pool.close()
    pool.join()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp4/weights/best.pt')
    parser.add_argument('--source', type=list, default=['data/images/cam0', 'data/images/cam1',
                                                        'data/images/cam2', 'data/images/cam3'],
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/my_data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--img_sz', '--img', '--img-size', nargs='+', type=int, default=[1280],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/mot_infer', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', default=True, help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # 1:[[114, 675], [2349, 1350]] 2:[[314,736],[2131,1443]]   3:[[552,564],[2886,1414]]
    parser.add_argument('--crop', type=list, default=[[[130, 675], [2349, 1350]],
                                                      [[314, 736], [2131, 1443]],
                                                      [[552, 564], [2886, 1414]],
                                                      [[752, 582], [2868, 1477]]
                                                      ], help='crop image to detect,x y x y')

    opt = parser.parse_args()
    opt.img_sz *= 2 if len(opt.img_sz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
