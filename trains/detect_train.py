import multiprocessing
import torch

from ultralytics import YOLO

def main():
    try:
        # Load a model
        model = YOLO("yolo11n.pt")

        # Train the model with memory-efficient settings
        train_results = model.train(
            data="goboard.yaml",  # path to dataset YAML
            epochs=400,  # number of training epochs
            imgsz=320,  # reduced image size
            device="cuda",  # device to run on
            batch=-1,  # auto batch size
            workers=min(multiprocessing.cpu_count(), 4),  # limit workers
            amp=True,  # automatic mixed precision
            patience=50,  # early stopping patience
            optimizer='Adam',  # try different optimizer
            close_mosaic=10,  # disable mosaic augmentation in last 10 epochs
        )

        # Additional memory management
        torch.cuda.empty_cache()

        # Rest of your code remains the same
        metrics = model.val()
        results = model("C:\\Users\\13375\\Desktop\\GO.jpg")
        results[0].show()

        path = model.export(format="onnx")
        print(f"Model exported to: {path}")

    except torch.cuda.OutOfMemoryError:
        print("GPU out of memory. Try further reducing batch size or image size.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()