from ultralytics import YOLO

def train_model():
    # Load YOLOv8-OBB nano model
    model = YOLO('yolov8n-obb.pt')

    # Model training configuration
    model.train(
        data='./ship_data.yaml',      # Dataset configuration file
        epochs=50,                    # Number of training epochs
        imgsz=1024,                   # Resolution for satellite imagery
        batch=2,                      # Batch size for CPU environment
        device='cpu',                 # Force CPU training
        workers=4,                    # Number of worker threads
        project='./runs',             # Project save directory
        name='ship_detection_v1',     # Experiment name
        exist_ok=True                 # Overwrite existing project folder
    )

if __name__ == '__main__':
    train_model()
