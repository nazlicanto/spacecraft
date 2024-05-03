# Spacecraft Detection with YOLOv8

This project automates spacecraft detection in images with both speed and precision. It utilizes YOLOv8, achieving an 0.85 (J) accuracy  while averaging just 1.2 seconds per image ideal for real-time applications.

![Image description](https://github.com/nazlicanto/spacecraft/blob/main/imggit/img11.png)

## Project Structure

```
spacecraft/
│
├── 3space.yaml          # Configuration for YOLOv8 model
├── augment.ipynb        # Notebook for data augmentation (horizontal and vertical flipping, modifying annotations)
├── data/                # Directory for training and inference data
│   ├── train/
│   └── val/
├── imggit/              # Images used in README
├── main.py              # Script for model inference
├── main.sh              # Shell script for simplified process control
├── prep.ipynb           # Notebook for data preparation (resizing, padding, splitting)
├── requirements.txt     # Project dependencies
├── results/             # Outputs from training and inference
│   ├── jaccard_index.py # Score script for Jaccard index
│   └── filter.ipynb     # Notebook for filtering labels
├── train.py             # Script for model training
└── assets/
    └── final_weights.pt # Final model weights

```

## Setup Instructions

Install the necessary Python packages:

``` 
pip install -r requirements.txt
```

### Data Preparation
Since the training data is restricted due to licensing, the initial state of the "data/" directory is empty.  You'll need to place your own images in this folder for both training and inference.

*** Prepare Data: *** Utilize the prep.ipynb notebook to format your data for compatibility with YOLOv8.

*** Augment Data: *** Run augment.ipynb to enhance your data using augmentation techniques. This helps the model better generalize from the training data.


## Training
Follow these steps to train the model:

```
### Run the preparation notebook 
jupyter notebook prep.ipynb
````

```
### Perform data augmentation (optional)
jupyter notebook augment.ipynb
````

```
### Begin training
python train.py
````

![Image description](https://github.com/nazlicanto/spacecraft/blob/main/imggit/img22.png)

## Inference
To use the trained model to detect spacecraft in new images, simply place your images in the "data/" directory and execute:

```
python main.py ${DATA_DIR} ${OUTPUT_PATH}
```


## Scoring
This project employs the Jaccard Index, also known as Intersection over Union (IoU), to assess the model's performance in detecting spacecraft. The Jaccard Index measures the area of overlap between the bounding box predicted by the model and the actual location of the spacecraft in the image.

To calculate the Jaccard Index score, navigate to the "results/" directory and run the provided script (jaccard_index.py). This script will compare the model's predictions with the ground truth data (actual locations of spacecraft) and provide a quantitative measure of the model's accuracy.

```
python score.py ${PREDICTED_LABEL} ${ACTUAL_LABEL}
````


This will evaluate the model's predictions stored during inference and provide a quantitative measure of its accuracy.


![Image description](https://github.com/nazlicanto/spacecraft/blob/main/imggit/img33.png)
