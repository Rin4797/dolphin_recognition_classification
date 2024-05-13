import cv2
import matplotlib.pyplot as plt


def from_coco_to_yolo(x_min, y_min, w, h, img_shape):
    x_max, y_max = x_min + w, y_min + h
    return [((x_min + x_max)/2)/img_shape[1], ((y_min + y_max)/2)/img_shape[0], w/img_shape[1], h/img_shape[0]]


def from_yolo_to_coco(x_center, y_center, w, h, img_shape):
    x_center_pix, y_center_pix = int(x_center*img_shape[1]), int(y_center*img_shape[0])
    w_pix, h_pix = int(w*img_shape[1]), int(h*img_shape[0])
    
    x_min, x_max = x_center_pix - w_pix//2, x_center_pix + w_pix//2
    y_min, y_max = y_center_pix - h_pix//2, y_center_pix + h_pix//2
    return [x_min, y_min, w, h]

def visualize_bbox(img, bbox, class_name, thickness=3):
    """Visualizes a single bounding box on the image"""
    
    if class_name == 'True':
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.75, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = str(category_id)
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(img)
    
    
def get_img_with_bboxes(image, bboxes, category_ids):
        
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        img = visualize_bbox(img, bbox, category_id)
    return img