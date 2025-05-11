import os
import torch
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)


from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="test/images/", split="train")

os.makedirs("eval_output", exist_ok=True)
os.makedirs("eval_output/mask", exist_ok=True)
os.makedirs("eval_output/combined", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
from PIL import Image

for idx in range(len(dataset)):
    # load image
    image = dataset[idx]["image"]
    IMAGE_ORIGINAL_W, IMAGE_ORIGINAL_H = image.size
    image


    image = image.resize((256,256), Image.BICUBIC)
    SCALE_X, SCALE_Y = 256 / IMAGE_ORIGINAL_W, 256 / IMAGE_ORIGINAL_H
    input_boxes = dataset[idx]["objects"]["bbox"]
    if len(input_boxes) == 0:
        continue

    for i in range(len(input_boxes)):
        input_boxes[i][0] *= SCALE_X
        input_boxes[i][1] *= SCALE_Y
        input_boxes[i][2] *= SCALE_X
        input_boxes[i][3] *= SCALE_Y


    from PIL import ImageDraw
    test = image.copy()
    imgd = ImageDraw.Draw(test) 

    for i in range(len(input_boxes)):
        x1, y1, x2, y2 = input_boxes[i]
        # print(x1, y1, x2, y2)
        imgd.rectangle([x1, y1, x2, y2])





    # prepare image + box prompt for the model
    inputs = processor(image, input_boxes=[input_boxes], return_tensors="pt").to(device)

    model.to(device)

    # forward pass
    # note that the authors use `multimask_output=False` when performing inference
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)


        outputs.pred_masks.shape


        import matplotlib.pyplot as plt

        pred_maskses = outputs.pred_masks
        pred_maskses = torch.reshape(pred_maskses, (-1, 256, 256))
        pred_maskses = torch.sigmoid(pred_maskses)
        pred_maskses = (pred_maskses > 0.5)
        pred_maskses = pred_maskses.cpu().numpy()

        import cv2 
        import imutils
        from PIL import ImageDraw


        fig, axes = plt.subplots()

        image_arr = np.array(image)
        image_PIL_square = Image.fromarray(image_arr)
        image_PIL_square = image_PIL_square.convert("RGBA")
        image_PIL_square = image_PIL_square.resize((256,256), Image.BICUBIC)

        bbox_centers = []
        contour_centers = []

        center_to_cropimg = dict()

        for i in range(len(pred_maskses)):
            mask_image = Image.fromarray(pred_maskses[i])
            mask_image = mask_image.convert("RGBA")
            mask_image = mask_image.resize((256, 256), Image.BICUBIC)
            
            ys, xs = np.nonzero(pred_maskses[i])
            if len(xs) > 0:
                cX = int(np.mean(xs))
                cY = int(np.mean(ys))
                contour_centers.append([cX, cY])

            # print(pred_maskses[i])
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), -1, -1
            for ycoord in range(len(pred_maskses[i])):
                for xcoord in range(len(pred_maskses[i][ycoord])):
                    if pred_maskses[i][ycoord][xcoord]:
                        min_x = min(min_x, xcoord)
                        max_x = max(max_x, xcoord)
                        min_y = min(min_y, ycoord)
                        max_y = max(max_y, ycoord)
            
            bbox_centers.append([(min_x + max_x) / 2, (min_y + max_y) / 2])

            imgd = ImageDraw.Draw(image_PIL_square)
            imgd.rectangle([min_x, min_y, max_x, max_y])
            image_PIL_square = Image.blend(image_PIL_square, mask_image, alpha=0.10)
            
            center_to_cropimg[((min_x + max_x) / 2, (min_y + max_y) / 2)] = np.array(image_PIL_square)[min_x:max_x, min_y:max_y, :]

        imgd = ImageDraw.Draw(image_PIL_square)
        for i in range(len(bbox_centers)):
            # imgd.ellipse([bbox_centers[i][0]-2, bbox_centers[i][1]-2, bbox_centers[i][0]+2, bbox_centers[i][1]+2], fill="#ffff33")
            imgd.ellipse([contour_centers[i][0]-2, contour_centers[i][1]-2, contour_centers[i][0]+2, contour_centers[i][1]+2], fill="red")

        image_PIL_square = image_PIL_square.resize((256, 256), Image.BICUBIC)

        image_PIL_square = image_PIL_square.convert("RGB")
        image_PIL_square.save(os.path.join("eval_output", "combined", f"combined_{idx}.jpg"))