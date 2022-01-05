# -*- coding: utf-8 -*-

import numpy as np


def compute_iou(box, boxes, box_area, boxes_area):
    """
    計算兩個 box 之間的 IoU。

    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. 框的區域。
    boxes_area: 長度為 box_count 的陣列。
    Note: 為了提高效能，直接用 area 計算一次以避免重複工作。
    """
    # 計算交疊 (intersection) 區域
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """
    計算兩組 box 之間的 IoU 重疊。

    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    Note: 為了獲得更好的性能，首先傳遞最大的集合，然後傳遞較小的集合。
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 計算重疊區域來產生矩陣 [boxes1 count, boxes2 count]
    # overlaps中的單元格都包含 IoU 值。
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """
    執行非最大抑制 (non-maximum suppression) 並返回保留 box 的索引。

    boxes: [N, (y1, x1, y2, x2)]. 請注意 (y2, x2) 位於框外。
    scores: 1-D array , 每個 box 的分數。
    threshold: Float. 用於過濾的 IoU 閾值。
    """
    if boxes.shape[0] < 1:
        return np.array([], dtype=np.int32)
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # 計算 box 的區域
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # 按分數排序來建立 box 索引（從高到低）
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # 選擇分數最高的框並將其索引添加到 list 中
        i = ixs[0]
        pick.append(i)
        # 計算選中框與其餘框的 IoU
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # 篩選出 IoU 超過閾值的 box。
        # 將索引返回到 ixs[1:]，因此加 1 以將索引返回到 ixs。
        remove_ixs = np.where(iou > threshold)[0] + 1
        # 刪除重疊 box 的索引。
        ixs = np.delete(ixs, remove_ixs)
        # 刪除被選中的 box 索引。
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def trim_zeros(x):
    """
    此函數刪除全為0的行
    *通常張量大於可用數據時，會用填充0
    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids,
                    pred_boxes, pred_class_ids, pred_scores,
                    iou_threshold=0.5, score_threshold=0.0):
    """
    比對預測(prediction, pred)和真實情況(ground truth, gt)實例之間的匹配狀況

    Returns:
        gt_match: 1-D array. 對於每個 GT 框，它都有匹配的預測框的索引
        pred_match: 1-D array. 對於每個預測框，它都有匹配的真實情況框的索引。
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5):
    """
    在設置的 IoU 閾值（默認為 0.5）下計算平均精度(mAP)

    Returns:
    mAP: Mean Average Precision
    precisions: 不同類別分數閾值的精度(precisions)列表
    recalls: 不同類別分數閾值的召回率(recall)列表
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids,
        pred_boxes, pred_class_ids, pred_scores,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """
    計算給定 IoU 閾值的 recall
    (表示偵測預測框中，找到了多少正確框)

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


def euclidean_distance_matrix(A, B, squared=False):
    """
    計算 A 和 B 中向量之間的所有成對距離

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    
    M = A.shape[0]
    N = B.shape[0]
    
    assert A.shape[1] == B.shape[1], "The number of components for vectors in A does not match that of B!"
        
    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)
    
    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)
    
    return D_squared


def err_counter(tr, pr, cls_num=10):
    a = pr - tr
    err_pr = tr[np.where(a!=0)]
    err_tr = pr[np.where(a!=0)]
    
    err_list = {}
    for i in range(0, cls_num):
        temp_list = {}
        total = 0
        tr_pos = np.where(err_tr==i)[0]
        for j in err_pr[tr_pos]:
            j = int(j)
            if j in temp_list.keys():
                temp_list[j] += 1
            else:
                temp_list[j] = 1
            total += 1 
        
        print("------------------------------")
        print("ground true ID: %d   total: %d"%(i, total))
        for k in sorted(temp_list.keys()):
            v = temp_list[k]
            print("%5d - error: %5d"%(k, v))
            err_list[i] = temp_list
    print("\n  total of wrong:", len(err_pr), "\n\n")


def err_counter_one_cls(tr, pr, threshold=0.5):
    pr = np.where(pr>threshold,1,0)
    tr = np.where(tr>threshold,1,0)
    a = pr - tr
    err_pr = pr[np.where(a!=0)]
    err_tr = tr[np.where(a!=0)]
        
    err_list = {}
    for i in range(2):
        temp_list = {}
        total = 0
        tr_pos = np.where(err_tr==i)[0]
        for j in err_pr[tr_pos]:
            j = int(j)
            if j in temp_list.keys():
                temp_list[j] += 1
            else:
                temp_list[j] = 1
            total += 1 
            
        print("------------------------------")
        print("ground true ID: %d   total: %d"%(i, total))
        for k in sorted(temp_list.keys()):
            v = temp_list[k]
            print("%5d - error: %5d"%(k, v))
            err_list[i] = temp_list
    print("\n  total of wrong:", len(err_pr), "\n\n")