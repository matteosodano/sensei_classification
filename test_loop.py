import os
import json
import torch
from PIL import Image
from torchvision import transforms

def gondola_evaluation(model, train_dataset, device):
    model.eval()

    # Same mapping used during training
    class_to_model_idx = train_dataset.class_to_idx

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    # ------------------------------------------------------------
    # Load annotations
    # ------------------------------------------------------------

    with open("/content/drive/MyDrive/Colab Notebooks/gondola_eval/boxes.json", "r") as f:
        boxes = json.load(f)

    print(f"Found {len(boxes)} annotations")
    correct = 0
    total = 0

    camera_correct = {}
    camera_total = {}

    with torch.no_grad():

        for i, annotation in enumerate(boxes):
            frame = annotation["frame"]

            # Assuming format: cameraXXX/filename.jpg
            camera_id = annotation["camera"]
            
            if camera_id not in camera_total:
                camera_total[camera_id] = 0
                camera_correct[camera_id] = 0

            image_path = os.path.join(
                "/content/drive/MyDrive/Colab Notebooks/gondola_eval/",
                frame
            )

            image = Image.open(image_path).convert("RGB")

            x1, y1, x2, y2 = annotation["bbox_xyxy"]

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            gt_class = annotation["product_id"]

            if str(gt_class) not in class_to_model_idx:
                continue

            gt_label = class_to_model_idx[str(gt_class)]


            input_tensor = transform(crop)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            outputs = model(input_tensor)

            prediction = outputs.argmax(dim=1).item()

            total += 1
            camera_total[camera_id] += 1

            if prediction == gt_label:
                correct += 1
                camera_correct[camera_id] += 1

    overall_accuracy = correct / total

    print("\n==============================")
    print("Overall evaluation")
    print("==============================")

    print(f"Total samples:    {total}")
    print(f"Correct:          {correct}")
    print(f"Accuracy:         {overall_accuracy:.4f}")


    print("\n==============================")
    print("Accuracy per camera")
    print("==============================")

    for camera_id in sorted(camera_total):

        acc = (
            camera_correct[camera_id]
            / camera_total[camera_id]
        )

        print(
            f"{camera_id}: "
            f"{acc:.4f} "
            f"({camera_correct[camera_id]}/{camera_total[camera_id]})"
        )