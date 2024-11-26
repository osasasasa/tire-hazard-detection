from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('./best.pt')

    # Predict the model
    model.predict('./check/check1.jpg', save=True, conf=0.1)    