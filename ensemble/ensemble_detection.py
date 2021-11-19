
from mmocr.models.textdet.postprocess.wrapper import poly_nms
import cv2
import os
import glob
import numpy as np

## change function poly_nms polygons = np.array(sorted(polygons, key=lambda boxs: Polygon(([int(i),int(j)]) for i,j in zip(boxs[0:8:2],boxs[1:9:2])).area))



root = 'outputs/text_detection'

def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list


def draw_texts(img, boxes=None, draw_box=True, on_ori_img=True,color= (0,0,255)):
    """Draw boxes and texts on empty img.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized image.
    """
   #  color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]

    if on_ori_img:
        out_img = img
    else:
        out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, box in enumerate(boxes):
        if draw_box:
            new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
            Pts = np.array([new_box], np.int32)
            cv2.polylines(
                out_img, [Pts.reshape((-1, 1, 2))],
                True,
                color,
                thickness=2)
    return out_img

def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split(',')
         box.append(1.0)
         boxs.append(box)
   return boxs

craft_paths = sorted(glob.glob('/text_detection/craft/*'))
db50_paths = sorted(glob.glob('/text_detection/dbr50/*'))
assert len(db50_paths) == len(db50_paths), 'file not match'

for (p1,p2) in zip(craft_paths,db50_paths):
   name = os.path.basename(p1).split('.')[0] + '.jpg'
   image_path = os.path.join('TestA',name)
   if not os.path.exists(image_path):
      name = os.path.basename(p1).split('.')[0] + '.jpeg'
      image_path = os.path.join('TestA',name)
   name_save = os.path.basename(image_path)
   # with open(os.path.join('outputs/text_detection/ensemble',name_save+'.txt'),'w') as file:
   image = cv2.imread(image_path)
   boxs1 = get_box(p1)
   boxs2 = get_box(p2)
   box_ensemble = poly_nms(boxs1+boxs2,threshold=0.3)
      # for box in box_ensemble:
      #    file.write(','.join([i.strip() for i in box[:-1]]) + '\n')

   img1 = draw_texts(image.copy(),boxs1,color=(0,255,0))
   img2 = draw_texts(image.copy(),boxs2)
   img3 = draw_texts(image.copy(),box_ensemble,color=(255,0,5))
   cv2.imshow('craft',cv2.resize(img1,(600,400)))
   cv2.imshow('db50',cv2.resize(img2,(600,400)))
   cv2.imshow('ensemble',cv2.resize(img3,(600,400)))
   cv2.waitKey(0)
cv2.destroyAllWindows()