import json
import os
import numpy as np
import cv2
from pycocotools import mask as maskUtils

base_dir = "DentAI.v2i.coco-segmentation"

datasets = ["train","test","valid"]

for d in datasets:

    ann_path = os.path.join(base_dir,d,"_annotations.coco.json")

    with open(ann_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    for img_id,img in images.items():

        h = img["height"]
        w = img["width"]

        mask = np.zeros((h,w),dtype=np.uint8)

        for ann in coco["annotations"]:

            if ann["image_id"] != img_id:
                continue

            seg = ann["segmentation"]

            # polygon
            if isinstance(seg,list):

                for poly in seg:
                    pts = np.array(poly).reshape(-1,2).astype(np.int32)
                    cv2.fillPoly(mask,[pts],255)

            # RLE
            elif isinstance(seg,dict):

                rle = seg
                m = maskUtils.decode(rle)
                mask = np.maximum(mask,m*255)

        out_dir = os.path.join(base_dir,d,"masks")
        os.makedirs(out_dir,exist_ok=True)

        cv2.imwrite(os.path.join(out_dir,img["file_name"]),mask)