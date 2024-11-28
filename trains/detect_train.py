import multiprocessing
import torch

from ultralytics import YOLO

def main():
    try:
        # Load a model
        model = YOLO("yolo11n.pt")

        # Train the model with memory-efficient settings
        model.train(
            data="goboard.yaml",  # path to dataset YAML
            epochs=400,  # number of training epochs
            imgsz=640,  # reduced image size
            device="cuda",  # device to run on
            workers=min(multiprocessing.cpu_count(), 4),  # limit workers
        )

        # Additional memory management
        torch.cuda.empty_cache()

        # Rest of your code remains the same
        model.val()
        results = model("F:\\Develop\\projects\\go-project\\Step4_processed_img.jpg")
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