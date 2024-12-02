import os
from functools import reduce

import cv2
import motmetrics as mm
import numpy as np

from .camera_utils import get_labels_with_camera
from .utils import get_labels, getData, gen_new_mot
from .utils import parse_pt_gt, parse_pt, compare_dataframes_mtmc
from .zone import Zone


__all__ = [
    'trajectory_fusion',
    'sub_cluster',
    'gen_res',
    'print_mtmct_result',
    'get_mtmct_matching_results',
    'save_mtmct_crops',
]


def trajectory_fusion(mot_feature, cid, cid_bias, use_zone=False, zone_path=''):
    cur_bias = cid_bias[cid]
    mot_list_break = {}
    if use_zone:
        zones = Zone(zone_path=zone_path)
        zones.set_cam(cid)
        mot_list = parse_pt(mot_feature, zones)  # 以跟踪对象id作为key的一个检测结果的列表
    else:
        mot_list = parse_pt(mot_feature)

    if use_zone:
        mot_list = zones.break_mot(mot_list, cid)
        mot_list = zones.filter_mot(mot_list, cid)  # filter by zone
        mot_list = zones.filter_bbox(mot_list, cid)  # filter bbox
    # 返回key为每帧图片中的某一个实例的名称(视频名称_跟踪id_,frame)
    mot_list_break = gen_new_mot(mot_list)  # 相对于mot_feature只是key换了个顺序

    tid_data = dict()
    for tid in mot_list:
        tracklet = mot_list[tid]
        if len(tracklet) <= 1:
            continue
        frame_list = list(tracklet.keys())
        frame_list.sort()
        # filter area too large
        zone_list = [tracklet[f]['zone'] for f in frame_list]
        feature_list = [tracklet[f]['feat'] for f in frame_list
                        if (tracklet[f]['bbox'][3] - tracklet[f]['bbox'][1]) * (
                                tracklet[f]['bbox'][2] - tracklet[f]['bbox'][0]) > 2000]
        if len(feature_list) < 2:
            feature_list = [tracklet[f]['feat'] for f in frame_list]
        io_time = [cur_bias + frame_list[0] / 10., cur_bias + frame_list[-1] / 10.]
        all_feat = np.array([feat for feat in feature_list])
        mean_feat = np.mean(all_feat, axis=0)  # 求出这个跟踪目标的平均feature
        tid_data[tid] = {
            'cam': cid,
            'tid': tid,
            'mean_feat': mean_feat,
            'zone_list': zone_list,
            'frame_list': frame_list,
            'tracklet': tracklet,
            'io_time': io_time
        }
    return tid_data, mot_list_break


def sub_cluster(cid_tid_dict,
                scene_cluster,
                use_ff=True,
                use_rerank=True,
                use_camera=False,
                use_st_filter=False):
    '''
    cid_tid_dict: all camera_id and track_id
    scene_cluster: like [41, 42, 43, 44, 45, 46] in AIC21 MTMCT S06 test videos
    '''
    assert (len(scene_cluster) != 0), "Error: scene_cluster length equals 0"
    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster])  # 按照相机编号进行排序。
    if use_camera:
        clu = get_labels_with_camera(cid_tid_dict, cid_tids, use_ff=use_ff, use_rerank=use_rerank,
                                     use_st_filter=use_st_filter)
    else:
        # 仅得一个列表，里面元素也为列表，就是匹配上的两个跟踪目标的id放到同一个列表里面
        clu = get_labels(cid_tid_dict, cid_tids, use_ff=use_ff, use_rerank=use_rerank, use_st_filter=use_st_filter)
    new_clu = list()  # 在clu的基础上，加上每个跟踪目标所在的相机id
    for c_list in clu:
        if len(c_list) <= 1:
            continue
        cam_list = [cid_tids[c][0] for c in c_list]  # 取出相机的id
        if len(cam_list) != len(set(cam_list)):  # 这里避免有同一个相机里面的跟踪id匹配上了。
            continue
        new_clu.append([cid_tids[c] for c in c_list])
    all_clu = new_clu
    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    return cid_tid_label  # 构建为[(cid,tid):label]的格式，label相同，则表示二者匹配上了


def gen_res(output_dir_filename, scene_cluster, map_tid, mot_list_breaks, use_roi=False, roi_dir=''):
    f_w = open(output_dir_filename, 'w')
    for idx, mot_feature in enumerate(mot_list_breaks):
        cid = scene_cluster[idx]
        img_rects = parse_pt_gt(mot_feature)
        if use_roi:
            assert (roi_dir != ''), "Error: roi_dir is not empty!"
            roi = cv2.imread(os.path.join(roi_dir, f'c{cid:03d}/roi.jpg'), 0)
            height, width = roi.shape

        for fid in img_rects:
            tid_rects = img_rects[fid]
            fid = int(fid) + 1
            for tid_rect in tid_rects:
                tid = tid_rect[0]
                rect = tid_rect[1:]
                cx = 0.5 * rect[0] + 0.5 * rect[2]
                cy = 0.5 * rect[1] + 0.5 * rect[3]
                w = rect[2] - rect[0]
                w = min(w * 1.2, w + 40)
                h = rect[3] - rect[1]
                h = min(h * 1.2, h + 40)
                rect[2] -= rect[0]
                rect[3] -= rect[1]
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                x1, y1 = max(0, cx - 0.5 * w), max(0, cy - 0.5 * h)
                if use_roi:
                    x2, y2 = min(width, cx + 0.5 * w), min(height, cy + 0.5 * h)
                else:
                    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
                w, h = x2 - x1, y2 - y1
                new_rect = list(map(int, [x1, y1, w, h]))
                rect = list(map(int, rect))
                if (cid, tid) in map_tid:
                    new_tid = map_tid[(cid, tid)]
                    f_w.write(str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(
                        map(str, new_rect)) + ' -1 -1''\n')
    print('gen_res: write file in {}'.format(output_dir_filename))
    f_w.close()


def print_mtmct_result(gt_file, pred_file):
    names = [
        'CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld',
        'Yworld'
    ]
    gt = getData(gt_file, names=names)
    pred = getData(pred_file, names=names)
    summary = compare_dataframes_mtmc(gt, pred)
    print('MTMCT summary: ', summary.columns.tolist())

    formatters = {
        'idf1': '{:2.2f}'.format,
        'idp': '{:2.2f}'.format,
        'idr': '{:2.2f}'.format,
        'mota': '{:2.2f}'.format
    }
    summary = summary[['idf1', 'idp', 'idr', 'mota']]
    summary.loc[:, 'idp'] *= 100
    summary.loc[:, 'idr'] *= 100
    summary.loc[:, 'idf1'] *= 100
    summary.loc[:, 'mota'] *= 100
    print(mm.io.render_summary(summary, formatters=formatters, namemap=mm.io.motchallenge_metric_names))


def get_mtmct_matching_results(pred_mtmct_file, secs_interval=0.5, video_fps=20):
    res = np.loadtxt(pred_mtmct_file)  # 'cid, tid, fid, x1, y1, w, h, -1, -1'
    camera_ids = list(map(int, np.unique(res[:, 0])))

    res = res[:, :7]
    # each line in res: 'cid, tid, fid, x1, y1, w, h'

    camera_tids = []
    camera_results = dict()
    for c_id in camera_ids:
        camera_results[c_id] = res[res[:, 0] == c_id]
        tids = np.unique(camera_results[c_id][:, 1])
        tids = list(map(int, tids))
        camera_tids.append(tids)

    # select common tids throughout each video
    common_tids = reduce(np.intersect1d, camera_tids)
    if len(common_tids) == 0:
        print('No common tracked ids in these videos, please check your MOT result or select new videos.')
        return None, None

    # get mtmct matching results by cid_tid_fid_results[c_id][t_id][f_id]
    cid_tid_fid_results = dict()
    cid_tid_to_fids = dict()
    interval = int(secs_interval * video_fps)  # preferably less than 10
    for c_id in camera_ids:
        cid_tid_fid_results[c_id] = dict()
        cid_tid_to_fids[c_id] = dict()
        for t_id in common_tids:
            tid_mask = camera_results[c_id][:, 1] == t_id
            cid_tid_fid_results[c_id][t_id] = dict()

            camera_trackid_results = camera_results[c_id][tid_mask]
            fids = np.unique(camera_trackid_results[:, 2])
            fids = fids[fids % interval == 0]
            fids = list(map(int, fids))
            cid_tid_to_fids[c_id][t_id] = fids

            for f_id in fids:
                st_frame = f_id
                ed_frame = f_id + interval

                st_mask = camera_trackid_results[:, 2] >= st_frame
                ed_mask = camera_trackid_results[:, 2] < ed_frame
                frame_mask = np.logical_and(st_mask, ed_mask)
                cid_tid_fid_results[c_id][t_id][f_id] = camera_trackid_results[frame_mask]

    return camera_results, cid_tid_fid_results


def save_mtmct_crops(cid_tid_fid_res, images_dir, crops_dir, width=300, height=200):
    camera_ids = cid_tid_fid_res.keys()
    seqs_folder = os.listdir(images_dir)
    seqs = []
    for x in seqs_folder:
        if os.path.isdir(os.path.join(images_dir, x)):
            seqs.append(x)
    assert len(seqs) == len(camera_ids)
    seqs.sort()

    if not os.path.exists(crops_dir):
        os.makedirs(crops_dir)

    common_tids = list(cid_tid_fid_res[list(camera_ids)[0]].keys())

    # get crops by name 'tid_cid_fid.jpg
    for t_id in common_tids:
        for i, c_id in enumerate(camera_ids):
            infer_dir = os.path.join(images_dir, seqs[i])
            if os.path.exists(os.path.join(infer_dir, 'img1')):
                infer_dir = os.path.join(infer_dir, 'img1')
            all_images = os.listdir(infer_dir)
            all_images.sort()

            for f_id in cid_tid_fid_res[c_id][t_id].keys():
                frame_idx = f_id - 1 if f_id > 0 else 0
                im_path = os.path.join(infer_dir, all_images[frame_idx])
                im = cv2.imread(im_path)  # (H, W, 3)
                # only select one track
                track = cid_tid_fid_res[c_id][t_id][f_id][0]
                cid, tid, fid, x1, y1, w, h = [int(v) for v in track]
                clip = im[y1:(y1 + h), x1:(x1 + w)]
                clip = cv2.resize(clip, (width, height))
                cv2.imwrite(os.path.join(crops_dir, 'tid{:06d}_cid{:06d}_fid{:06d}.jpg'.format(tid, cid, fid)), clip)
            print("Finish cropping image of tracked_id {} in camera: {}".format(t_id, c_id))
