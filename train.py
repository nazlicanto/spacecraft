from ultralytics import YOLO
import os
import logging


logging.basicConfig(level=logging.INFO)

model = YOLO('yolov8n.pt')

data_path = '/home/ubuntu/aicr/nasa/yolov8n/3space.yaml'
img_size = 1024
num_epochs = 300
batch = 1
device = '0'

results_dir = '/home/ubuntu/aicr/nasa/yolov8n/results'
os.makedirs(results_dir, exist_ok=True)


for epoch in range(num_epochs):
    results = model.train(data=data_path, epochs=1, imgsz=img_size, batch=batch, device=device)
    logging.info(f"Finished epoch {epoch}: {results}")
    if epoch % 20 == 0:
        checkpoint_path = os.path.join(results_dir, f'weights_epoch_{epoch}.pt')
        model.save(checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")


final_model_path = os.path.join(results_dir, 'final_model_weights.pt')
model.save(final_model_path)
logging.info(f"Final model weights saved at {final_model_path}")


