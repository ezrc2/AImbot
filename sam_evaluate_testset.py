import os
import matplotlib.pyplot as plt
import torch
from transformers import SamModel, SamProcessor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="demo_set/images", split="train")

import numpy as np
from PIL import Image

idx = 0

os.makedirs("eval_output", exist_ok=True)
os.makedirs("eval_output/mask", exist_ok=True)
os.makedirs("eval_output/combined", exist_ok=True)

for idx in tqdm(range(len(dataset))):
    # load image
    image = dataset[idx]["image"]
    IMAGE_ORIGINAL_W, IMAGE_ORIGINAL_H = image.size

    image = image.resize((256,256), Image.BICUBIC)
    SCALE_X, SCALE_Y = 256 / IMAGE_ORIGINAL_W, 256 / IMAGE_ORIGINAL_H
    input_boxes = dataset[idx]["objects"]["bbox"]

    for i in range(len(input_boxes)):
        input_boxes[i][0] *= SCALE_X
        input_boxes[i][1] *= SCALE_Y
        input_boxes[i][2] *= SCALE_X
        input_boxes[i][3] *= SCALE_Y
    
    # prepare image + box prompt for the model
    inputs = processor(image, input_boxes=[input_boxes], return_tensors="pt").to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    pred_maskses = outputs.pred_masks
    pred_maskses = torch.reshape(pred_maskses, (-1, 256, 256))
    pred_maskses = torch.sigmoid(pred_maskses)
    pred_maskses = (pred_maskses > 0.5)
    pred_maskses = pred_maskses.cpu().numpy()
    
    fig, axes = plt.subplots()

    image_arr = np.array(image)
    image_PIL_square = Image.fromarray(image_arr)
    image_PIL_square = image_PIL_square.convert("RGBA")
    image_PIL_square = image_PIL_square.resize((256,256), Image.BICUBIC)

    for i in range(len(pred_maskses)):
        mask_image = Image.fromarray(pred_maskses[i])
        mask_image = mask_image.convert("RGBA")
        mask_image = mask_image.resize((256, 256), Image.BICUBIC)
        
        image_PIL_square = Image.blend(image_PIL_square, mask_image, alpha=0.44)

        mask_image = mask_image.convert("RGB")
        mask_image.save(os.path.join("eval_output", "mask", f"mask_{idx}_maskno{i}.jpg"))

    image_PIL_square = image_PIL_square.resize((IMAGE_ORIGINAL_W, IMAGE_ORIGINAL_H), Image.BICUBIC)
    
    image_PIL_square = image_PIL_square.convert("RGB")
    image_PIL_square.save(os.path.join("eval_output", "combined", f"combined_{idx}.jpg"))