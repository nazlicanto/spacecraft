import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Union

def jaccard_index(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """ Calculate Jaccard Index (IoU) for each pair of bounding boxes. """
    xmin = np.maximum(actual[:, 0], predicted[:, 0])
    ymin = np.maximum(actual[:, 1], predicted[:, 1])
    xmax = np.minimum(actual[:, 2], predicted[:, 2])
    ymax = np.minimum(actual[:, 3], predicted[:, 3])

    inter_area = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
    pred_area = (predicted[:, 2] - predicted[:, 0]) * (predicted[:, 3] - predicted[:, 1])
    actual_area = (actual[:, 2] - actual[:, 0]) * (actual[:, 3] - actual[:, 1])
    union_area = pred_area + actual_area - inter_area
    return inter_area / union_area

def score_rows(predicted_df: pd.DataFrame, actual_df: pd.DataFrame) -> Dict[str, float]:
    """ Score the set of predicted bounding boxes against the actual ground truth. """
    iou_scores = jaccard_index(predicted_df.values, actual_df.values)
    return {"score": np.mean(iou_scores)}

def main(predicted_path: Union[str, Path], actual_path: Union[str, Path]) -> Dict[str, float]:
    """ Calculate the Jaccard Index score for bounding box predictions. """
    try:
        predicted_df = pd.read_csv(predicted_path).set_index('image_id')
        actual_df = pd.read_csv(actual_path).set_index('image_id')
    except Exception as e:
        return {"error": f"Error loading CSV files: {e}"}

    predicted_df.sort_index(inplace=True)
    actual_df.sort_index(inplace=True)

    if not predicted_df.index.equals(actual_df.index):
        return {"error": "Indices of predicted and actual data do not match."}

    try:
        return score_rows(predicted_df, actual_df)
    except Exception as e:
        return {"error": f"Error calculating IoU scores: {e}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU for bounding box predictions.")
    parser.add_argument("predicted_path", type=str, help="Path to the CSV file with predictions.")
    parser.add_argument("actual_path", type=str, help="Path to the CSV file with ground truth labels.")
    args = parser.parse_args()
    result = main(predicted_path=args.predicted_path, actual_path=args.actual_path)
    print(json.dumps(result, indent=2))


"""
# python score.py /home/ubuntu/aicr/nasa/subspace/sub/pred_full_640_1_.csv /home/ubuntu/aicr/nasa/subspace/sub/actual_partial_labels.csv
 "score": 0.24093268173285587

# python score.py /home/ubuntu/aicr/nasa/subspace/sub/pred_part_640_0_.csv /home/ubuntu/aicr/nasa/subspace/sub/actual_partial_labels.csv
"score": 0.24093268173285587

# python score.py /home/ubuntu/aicr/nasa/subspace/sub/pred_full_1024_0_.csv /home/ubuntu/aicr/nasa/subspace/sub/actual_partial_labels.csv
pred_full_1024_0_


python score.py /home/ubuntu/aicr/nasa/subspace/sub/pred_part_best_640_0_.csv /home/ubuntu/aicr/nasa/subspace/sub/actual_partial_labels.csv
pred_part_best_640_0_

python score.py /home/ubuntu/aicr/nasa/subspace/sub/pred_full_best_640_1_.csv /home/ubuntu/aicr/nasa/subspace/sub/actual_partial_labels.csv
pred_part_best_640_0_

python score.py /home/ubuntu/aicr/nasa/subspace/sub/weights0.csv /home/ubuntu/aicr/nasa/subspace/sub/actual_partial_labels.csv 0.67
  weights 0: "score": 0.6693413078503463
  weights 1: "score": 0.6966813081656328
  weights 5: "score": 0.7578559549713647
  weights 6: "score": 0.7539718132502523
  weights 6: "score": 0.7578110097426968
  weights 24: "score": 0.7820786940916825
"""