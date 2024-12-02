import numpy as np
import torch
from utils.general import xyxy2xywh, xyxy2tlwh
from .tracker.deepsort_tracker import DeepSORTTracker
from .baseline import Baseline
from .defaults import _C as cfg
from .utils import preprocess_reid


class SDE_ReID(object):
    """
    完成deepsort的跟踪
    """

    def __init__(self, model, max_age=70, max_iou_distance=0.9, mtmct=True):
        self.model = model
        self.tracker = DeepSORTTracker(max_age=max_age, max_iou_distance=max_iou_distance)
        self.mtmct = mtmct

    def get_crops(self, xyxy, ori_img):
        """
        从原图中提取检测区域，并resize到指定大小(特征提取网络的输入尺寸)
        """
        w, h = self.tracker.input_size
        crops = []
        xyxy = xyxy.int()
        ori_img = ori_img.transpose(1, 0, 2)  # [h,w,3]->[w,h,3]
        for i, bbox in enumerate(xyxy):
            crop = ori_img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
            crops.append(crop)
        crops = preprocess_reid(crops, w, h)
        return crops

    def postprocess(self, pred_dets, pred_embs):
        tracker = self.tracker
        tracker.predict()
        online_targets = tracker.update(pred_dets, pred_embs)
        online_tlwhs, online_scores, online_ids = [], [], []
        for t in online_targets:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            tlwh = t.to_tlwh()
            tscore = t.score
            tid = t.track_id
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            if 0 < tracker.vertical_ratio < tlwh[2] / tlwh[3]:
                continue
            online_tlwhs.append(tlwh)
            online_scores.append(tscore)
            online_ids.append(tid)

        tracking_outs = {
            'online_tlwhs': online_tlwhs,
            'online_scores': online_scores,
            'online_ids': online_ids,
        }
        return tracking_outs

    def detection_track_match_update(self, pred_dets, pred_embs):
        tracker = self.tracker
        tracker.predict()  # 对现有的track先进行卡尔曼滤波predict
        online_targets = tracker.update(pred_dets, pred_embs)  # 正在跟踪的目标的的对象
        online_targets = sorted(online_targets)
        return online_targets

    def postprocess_mtmct(self, pred_dets, pred_embs, frame_id, seq_name, cam_id, last_result):
        tracker = self.tracker
        tracker.predict()  # 对现有的track先进行卡尔曼滤波predict
        online_targets = tracker.update(pred_dets, pred_embs)  # 正在跟踪的目标的的对象
        # 先对目标进行排序，方便后续镜头匹配
        online_targets = sorted(online_targets)
        online_tlwhs, online_scores, online_ids, cid_tid_dict, matchs = [], [], [], {}, []
        online_tlbrs, online_feats = [], []
        pass_num = 0
        out_num = 0
        for i, t in enumerate(online_targets):
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            tlwh = t.to_tlwh()
            tscore = t.score
            tid = t.track_id
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            if 0 < tracker.vertical_ratio < tlwh[2] / tlwh[3]:
                continue
            # 坐标匹配
            # match = 0
            #
            # if cam_id == 0:  # 如果是第一个镜头，将超越匹配线(去掉匹配线)的跟踪目标设置为匹配状态
            #     # if tlwh[0] < 1200:
            #     t.match_state = 4
            #     t.show_id = t.track_id
            #     match = 1
            # elif cam_id == 1:  # 第二个镜头，需要结合第一个，进行匹配
            #     if t.match_state != 4:  # 如果还没有匹配上
            #         if 800 < tlwh[0] < 2050:
            #             if out_num != 0:
            #                 tid = last_result["online_ids"][i - out_num]
            #             else:
            #                 tid = last_result["online_ids"][i - pass_num]
            #             t.show_id = tid
            #             t.match_state = 4  # 匹配标志位
            #         else:
            #             pass_num += 1
            #             continue
            #     else:  # 如果已经匹配上了
            #         if 800 < tlwh[0] < 2050:
            #             pass_num += 1
            #         else:
            #             out_num += 1
            # elif cam_id == 2:
            #     if t.match_state != 4:  # 如果没有匹配上
            #         if 2329 < tlwh[0] < 2800:
            #             if out_num != 0:
            #                 tid = last_result["online_ids"][i - out_num]
            #             else:
            #                 tid = last_result["online_ids"][i - pass_num]
            #             t.show_id = tid
            #             t.match_state = 4  # 匹配标志位
            #         else:
            #             pass_num += 1
            #             continue
            #     else:
            #         if 2329 < tlwh[0] < 2800:
            #             pass_num += 1
            #         else:
            #             out_num += 1
            #
            # elif cam_id == 3:
            #     if t.match_state != 4:  # 如果没有匹配上
            #         if 2364 < tlwh[0] < 2810:
            #             if out_num != 0:
            #                 tid = last_result["online_ids"][i - out_num]
            #             else:
            #                 tid = last_result["online_ids"][i - pass_num]
            #             t.show_id = tid
            #             t.match_state = 4  # 匹配标志位
            #         else:
            #             pass_num += 1
            #             continue
            #     else:
            #         if 2364 < tlwh[0] < 2810:
            #             pass_num += 1
            #         else:
            #             out_num += 1

            # if cam_id == 0 and (tid == 1 or tid == 12 or tid == 18 or tid == 15 or tid == 19):
            #     cid_tid_dict[(cam_id, tid)] = {
            #         'cam': cam_id,
            #         "tid": tid,
            #         "mean_feat": t.feat,
            #         "scores": tscore,
            #         "tlwhs": tlwh
            #     }
            # if cam_id == 1 and (tid == 1 or tid == 3 or tid == 6 or tid == 2 or tid == 4):
            #     cid_tid_dict[(cam_id, tid)] = {
            #         'cam': cam_id,
            #         "tid": tid,
            #         "mean_feat": t.feat,
            #         "scores": tscore,
            #         "tlwhs": tlwh
            #     }

            # cid_tid_dict[(cam_id, tid)] = {
            #     'cam': cam_id,
            #     "tid": tid,
            #     "mean_feat": t.feat,
            #     "scores": tscore,
            #     "tlwhs": tlwh
            # }
            online_tlwhs.append(tlwh)
            online_scores.append(tscore)
            online_ids.append(t.show_id)  # 显示到跟踪画面中的ID
            # matchs.append(match)
            online_tlbrs.append(t.to_tlbr())
            online_feats.append(t.feat)
        # 目前已经confirmed的track(连续检测超过3帧并且当前帧中存在该目标)
        tracking_outs = {
            'online_tlwhs': online_tlwhs,
            'online_scores': online_scores,
            'online_ids': online_ids,
            'feat_data': {},
            # 'matchs': matchs
        }
        for _tlbr, _id, _feat in zip(online_tlbrs, online_ids, online_feats):
            feat_data = {}
            feat_data['bbox'] = _tlbr
            feat_data['frame'] = f"{frame_id:06d}"
            feat_data['id'] = _id
            _imgname = f'{seq_name}_{_id}_{frame_id}.jpg'  # 视频名称、跟踪id、第几张图片。
            feat_data['imgname'] = _imgname
            feat_data['feat'] = _feat
            tracking_outs['feat_data'].update({_imgname: feat_data})
        return tracking_outs
        # return cid_tid_dict

    def predict(self, crops, pred_det, cam_id, device="cpu", frame_id=0, seq_name='', last_result=None):
        """
        首先对crop提取特征，接着进行deepsort的一系列操作
        """
        crops = torch.Tensor(crops).to(device)
        pred_emb = self.model(crops)[0].to("cpu")
        pred_det = pred_det.to("cpu")
        pred_det[:, :4] = xyxy2tlwh(pred_det[:, :4])
        # tracking_outs = self.postprocess_mtmct(pred_det, pred_emb, frame_id, seq_name, cam_id, last_result)
        tracking_outs = self.detection_track_match_update(pred_det, pred_emb)
        return tracking_outs

    def cam_match(self, online_targets, cam_id, last_result):
        online_tlwhs, online_scores, online_ids, cid_tid_dict, matchs = [], [], [], {}, []
        online_tlbrs, online_feats = [], []
        pass_num = 0
        out_num = 0
        if cam_id == 2:
            print("last_result",len(last_result["online_tlwhs"]))
            print("online_targets",len(online_targets))
        for i, t in enumerate(online_targets):
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            tlwh = t.to_tlwh()
            tscore = t.score
            tid = t.track_id
            # 坐标匹配
            match = 0
            if cam_id == 0:  # 如果是第一个镜头，将超越匹配线(去掉匹配线)的跟踪目标设置为匹配状态
                # if tlwh[0] < 1200:
                t.match_state = 4
                t.show_id = t.track_id
                match = 1
            elif cam_id == 1:  # 第二个镜头，需要结合第一个，进行匹配
                if t.match_state != 4:  # 如果还没有匹配上
                    if 850 < tlwh[0] < 2100:
                        if out_num != 0:
                            tid = last_result["online_ids"][i - out_num]
                        else:
                            tid = last_result["online_ids"][i - pass_num]
                        t.show_id = tid
                        t.match_state = 4  # 匹配标志位
                    else:
                        pass_num += 1
                        continue
                else:  # 如果已经匹配上了
                    if 800 < tlwh[0] < 2050:
                        pass_num += 1
                    else:
                        out_num += 1
            elif cam_id == 2:
                if t.match_state != 4:  # 如果没有匹配上
                    if 2329 < tlwh[0] < 2800:
                        if out_num != 0:
                            tid = last_result["online_ids"][i - out_num]
                        else:
                            tid = last_result["online_ids"][i - pass_num]
                        t.show_id = tid
                        t.match_state = 4  # 匹配标志位
                    else:
                        pass_num += 1
                        continue
                else:
                    if 2329 < tlwh[0] < 2800:
                        pass_num += 1
                    elif 552 < tlwh[0] < 1371:
                        pass
                    else:
                        out_num += 1

            elif cam_id == 3:
                if t.match_state != 4:  # 如果没有匹配上
                    if 2364 < tlwh[0] < 2810:
                        if out_num != 0:
                            tid = last_result["online_ids"][i - out_num]
                        else:
                            tid = last_result["online_ids"][i - pass_num]
                        t.show_id = tid
                        t.match_state = 4  # 匹配标志位
                    else:
                        pass_num += 1
                        continue
                else:
                    if 2364 < tlwh[0] < 2810:
                        pass_num += 1
                    else:
                        out_num += 1
            online_tlwhs.append(tlwh)
            online_scores.append(tscore)
            online_ids.append(t.show_id)  # 显示到跟踪画面中的ID
            matchs.append(match)
            online_tlbrs.append(t.to_tlbr())
            online_feats.append(t.feat)
        # 目前已经confirmed的track(连续检测超过3帧并且当前帧中存在该目标)
        tracking_outs = {
            'online_tlwhs': online_tlwhs,
            'online_scores': online_scores,
            'online_ids': online_ids,
            'feat_data': {},
            'matchs': matchs
        }
        return tracking_outs


def build_model(cfg, device, num_classes, half):
    """构建模型"""
    model = Baseline(num_classes, half, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                     cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, device)
    return model
