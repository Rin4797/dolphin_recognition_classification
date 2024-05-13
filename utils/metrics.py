from collections import Counter
import torchvision
import math
import torch


def get_predicted_boxes_format(preds: dict, fname2index, min_area=0, max_area=1000000):
    predicted_boxes = []
    for f_name in preds:
        for bbox in preds[f_name]:
            x_min, y_min, w, h = bbox['bbox'][:4]
            score = bbox['score']
            area = bbox['area']
            if area >= min_area and area <= max_area:
                predicted_boxes.append([fname2index[f_name], 0, score, x_min, y_min, x_min + w, y_min + h])
    return predicted_boxes


def get_true_boxes_format(true_boxes, fname2index):
    true_boxes_format = []
    for f_name in true_boxes:
        for bbox in true_boxes[f_name]['coco']:
            x_min, y_min, w, h = bbox[:4]
            true_boxes_format.append([fname2index[f_name], 0, 1, x_min, y_min, x_min + w, y_min + h])
    return true_boxes_format

def calc_tp_fp_fn(ground_truths, detections, iou_threshold):
  
  '''
  ground_truths: (list): [[train_index, class_prediction, prob_score, x1, y1, x2, y2],[],...[]]
  detections: (list): [[train_index, class_prediction, prob_score, x1, y1, x2, y2],[],...[]]
  '''

  global_tp = []
  global_fp = []
  global_gt = []

  global_gt = ground_truths

  amount_bboxes = Counter([gt[0] for gt in ground_truths])
  for key, value in amount_bboxes.items():
      amount_bboxes[key] = torch.zeros(value)

  detections.sort(key = lambda x: x[2], reverse = True)
    
  tp = torch.zeros(len(detections))
  fp = torch.zeros(len(detections))
  total_gt_bboxes = len(ground_truths)

  fp_frame = []
  tp_frame = []


  for detection_index, detection in enumerate(detections):
      ground_truth_image = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

      num_gt_boxes = len(ground_truth_image)
      best_iou = 0
      best_gt_index = 0


      for index, gt in enumerate(ground_truth_image):
        
        iou = torchvision.ops.box_iou(torch.tensor(detection[3:]).unsqueeze(0), 
                                      torch.tensor(gt[3:]).unsqueeze(0))
        
        if iou > best_iou:
          best_iou = iou
          best_gt_index = index

      if best_iou > iou_threshold:
        if amount_bboxes[detection[0]][best_gt_index] == 0:
          tp[detection_index] = 1
          amount_bboxes[detection[0]][best_gt_index] == 1
          tp_frame.append(detection)
          global_tp.append(detection)

        else:
          fp[detection_index] = 1
          fp_frame.append(detection)
          global_fp.append(detection)
      else:
          fp[detection_index] = 1
          fp_frame.append(detection)
          global_fp.append(detection)


  global_gt_updated = []
  for gt in global_gt:
    if math.isnan(gt[3]) == False:
      global_gt_updated.append(gt)


  global_fn = len(global_gt_updated) - len(global_tp)
  
  return  global_tp, global_fp, global_fn


def print_main_metrics(tp, fp, fn):
    precision = len(tp)/ (len(tp)+ len(fp))
    recall = len(tp)/ (len(tp) + fn)

    f1_score =  2* (precision * recall)/ (precision + recall)

    print('TP', len(tp))
    print('recall', round(recall, 3))
    print('precision', round(precision, 3))
    print('f_score', round(f1_score, 3))