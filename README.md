# AImbot

AImbot: Generic AI Aimbot for FPS Games

Steve Xing (stevex2), Eric Chen (ericzc2)

### Dataset

Download the compressed 'datasets' directory from [Google Drive link](https://drive.google.com/file/d/1QUZ0OfJUlFfdEdf3pQ7ILQWJ4S1RwPRb/view?usp=drive_link)

Move `datasets.zip` it into the AImbot directory, and unzip:
```
unzip datasets.zip
```

(unzipping might create a '__MACOSX' folder, but you can delete it)

This dataset is a combination of the [combat vehicles warthunder Computer Vision Project](https://universe.roboflow.com/warthunder-aehie/combat-vehicles-warthunder) dataset and [CS2v2](https://universe.roboflow.com/miktory/cs2v2) dataset from Roboflow.

The `train.yaml` file for training the YOLOv11 model is inside the datasets folder.

### Setup

Conda environment:
```
conda env create -f environment.yml
conda activate fp
```

### Usage

* `train_yolo.ipynb`: the YOLO model (stored at 'runs/detect/finetuned_yolov11_combined_640_augmented_2cls/weights/last.pt') is already trained, so you only need to run the "Test" section. The images with predicted bounding boxes (trained_results directory) and predicted bounding box coordinates (trained_boxes directory) are in the `datasets/test` directory.
* `format_dataset.ipynb`: run after train_yolo.ipynb notebook to format images and predicted bounding boxes

Then to run inference with our SAM method: 
```
python sam_eval_2.py
```
Make sure the dataset directory paths are correct for each `.ipynb` and `.py` file

### Code Description

* `train_yolo.ipynb`: jupyter notebook for training and testing the YOLOv11 model
* `format_dataset.ipynb`: transforms test images and trained_boxes pairs into HuggingFace transformers dataset loader format
* `sam_eval_2.py`: run inference with SAM; uses datasets dataloader, draws mask, bounding box, and centroid
* `SAM_TANK_inference.ipynb`: same as sam eval_eval_2.py, but for better visualization of single images
