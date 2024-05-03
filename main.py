import os
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import click
from loguru import logger

def centered_box(img, scale=0.1):
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2
    box_width, box_height = int(width * scale), int(height * scale)
    x1 = center_x - box_width // 2
    y1 = center_y - box_height // 2
    x2 = center_x + box_width // 2
    y2 = center_y + box_height // 2
    return [x1, y1, x2, y2]

@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False),
)
def main(data_dir, output_path):
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    submission_file_path = output_path / "final_weights_preds.csv"  
    images_dir = data_dir / "images"

    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
    assert output_path.exists(), f"Expected output directory {output_path} does not exist"
    assert images_dir.exists(), f"Expected images dir {images_dir} does not exist"
    logger.info(f"Using data dir: {data_dir}")

    model = YOLO('assets/final_weights.pt')  

    results = []

    # process each image
    for image_path in tqdm(list(images_dir.glob('*.png'))):
        img = cv2.imread(str(image_path))
        result = model(img, verbose=False)[0]
        
        bbox = result.boxes.xyxy[0].tolist() if len(result.boxes) > 0 else centered_box(img)
        bbox = [int(x) for x in bbox]
        image_name_without_ext = image_path.stem
        results.append([image_name_without_ext, *bbox])
    
    # save 
    df = pd.DataFrame(results, columns=['image_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv(submission_file_path, index=False)  
    print(f"Results saved to {submission_file_path}")

if __name__ == "__main__":
    main()

# RUN: python main.py ${DATA_DIR} ${OUTPUT_PATH}