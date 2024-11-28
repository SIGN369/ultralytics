import os
from ultralytics import YOLO
import cv2
import torch
import platform
import numpy as np


def crop_objects_from_image(image_path, output_dir_crops, output_dir_visuals, output_dir_transformed, model,
                            conf_threshold=0.73, visualize=False):
    """
    使用 YOLO 模型检测图像中的物体，并将每个检测到的物体剪切成单独的图片。
    对剪切后的图像进行透视变换，并保存结果。

    :param image_path: 原始图像的路径
    :param output_dir_crops: 保存剪切后物体图像的目录
    :param output_dir_visuals: 保存带有检测边界框的原始图像的目录
    :param output_dir_transformed: 保存透视变换后图像的目录
    :param model: 已加载的 YOLO 模型
    :param conf_threshold: 置信度阈值，低于此值的检测将被忽略
    :param visualize: 是否保存带有检测边界框的原始图像
    """
    # 确保输出目录存在
    os.makedirs(output_dir_crops, exist_ok=True)
    os.makedirs(output_dir_visuals, exist_ok=True)
    os.makedirs(output_dir_transformed, exist_ok=True)

    # 获取图像的基名
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # 进行物体检测
    results = model(gray_img, conf=conf_threshold)
    # 解析检测结果
    detections = results[0].boxes  # 获取第一个图像的检测结果
    # 在检测代码中，打印检测结果数量
    print(f"检测到 {len(detections)} 个物体")
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 遍历每个检测到的物体
    for idx, box in enumerate(detections):
        cls_id = int(box.cls[0])  # 获取类别ID
        conf = box.conf[0]  # 获取置信度

        try:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        except Exception as e:
            print(f"解包边界框坐标时出错: {e}")
            continue

        # 将坐标转换为整数，并确保在图像范围内
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), width)
        y2 = min(int(y2), height)

        # 剪切物体
        cropped_image = image[y1:y2, x1:x2]

        # 获取类别名称（假设模型已经定义了类别名称）
        class_name = model.names[cls_id] if model.names else str(cls_id)

        # 构建保存路径
        crop_output_path = os.path.join(output_dir_crops, f"{base_name}_object{idx + 1}_{class_name}.jpg")

        # 保存剪切后的图像
        cv2.imwrite(crop_output_path, cropped_image)
        print(f"保存剪切图像到: {crop_output_path}")

        # 对剪切后的图像进行透视变换
        transformed_image = perspective_transform(cropped_image)

        # 保存透视变换后的图像
        transformed_output_path = os.path.join(output_dir_transformed,
                                               f"{base_name}_object{idx + 1}_{class_name}_transformed.jpg")
        cv2.imwrite(transformed_output_path, transformed_image)
        print(f"保存透视变换后的图像到: {transformed_output_path}")

        # 可视化检测结果（可选）
        if visualize:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2
            )

    # 保存带有边界框的图像（可选）
    if visualize:
        visualized_output_path = os.path.join(output_dir_visuals, f"{base_name}_visualized.jpg")
        cv2.imwrite(visualized_output_path, image)
        print(f"保存可视化图像到: {visualized_output_path}")


def perspective_transform(image):
    """
    对输入图像进行透视变换。

    :param image: 输入图像
    :return: 透视变换后的图像
    """
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 定义源点和目标点，示例为将图像缩放并倾斜
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst_pts = np.float32([[w * 0.1, h * 0.33], [w * 0.9, h * 0.25], [w * 0.2, h * 0.7], [w * 0.8, h * 0.9]])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 进行透视变换
    transformed_image = cv2.warpPerspective(image, matrix, (w, h))

    return transformed_image


def main():
    # 配置路径
    image_dir = './GoDataset'  # 替换为你的图像目录
    output_dir_crops = './cropped_objects/crops'  # 保存剪切后的物体图像
    output_dir_visuals = './cropped_objects/visuals'  # 保存带有边界框的原始图像
    output_dir_transformed = './cropped_objects/transformed'  # 保存透视变换后的图像
    model_path = './weights/best.pt'  # 替换为你的 YOLO 模型路径
    visualize = True  # 是否保存带有检测边界框的原始图像

    # 加载 YOLO 模型（使用 GPU 加速，如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    print("模型类别名称：", model.names)
    # 遍历图像目录中的所有图像文件
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(image_dir, image_file)
            crop_objects_from_image(image_path, output_dir_crops, output_dir_visuals, output_dir_transformed, model,
                                    visualize=visualize)


if __name__ == "__main__":
    print(platform.system())
    main()
