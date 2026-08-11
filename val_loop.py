import torch
import torch.nn.functional as F


def validate(model, val_loader, train_classes, device, threshold):
    model.eval()

    # Classes seen during training
    train_classes_tensor = torch.tensor(
        train_classes,
        device=device
    )

    # Per-class statistics
    class_correct = {
        class_id: 0
        for class_id in train_classes
    }

    class_total = {
        class_id: 0
        for class_id in train_classes
    }

    # Overall known statistics
    known_correct = 0
    known_total = 0

    # Unknown statistics
    unknown_correct = 0
    unknown_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = F.softmax(outputs, dim=1)

            max_probs, pred_indices = torch.max(
                probs,
                dim=1
            )

            # Convert model indices -> actual class IDs
            pred_classes = torch.tensor(
                [
                    train_classes[i]
                    for i in pred_indices.cpu().tolist()
                ],
                device=device
            )

            predictions = pred_classes.clone()

            reject_mask = max_probs < threshold

            # -1 = unknown prediction
            predictions[reject_mask] = -1

            # Separate known and unknown GT samples
            known_mask = torch.isin(
                labels,
                train_classes_tensor
            )

            unknown_mask = ~known_mask

            # Known samples
            if known_mask.any():

                known_predictions = predictions[known_mask]
                known_labels = labels[known_mask]

                # Overall known statistics
                known_correct += (
                    known_predictions == known_labels
                ).sum().item()

                known_total += known_labels.size(0)

                # Per-class statistics
                for class_id in train_classes:

                    class_mask = known_labels == class_id

                    if class_mask.any():

                        class_total[class_id] += (
                            class_mask.sum().item()
                        )

                        class_correct[class_id] += (
                            (
                                known_predictions[class_mask]
                                == class_id
                            )
                            .sum()
                            .item()
                        )

            # Unknown samples
            if unknown_mask.any():

                unknown_predictions = predictions[unknown_mask]

                unknown_total += unknown_mask.sum().item()

                # Correct if rejected
                unknown_correct += (
                    unknown_predictions == -1
                ).sum().item()

    # Per-class accuracy
    per_class_accuracy = {}

    for class_id in train_classes:

        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = (
                class_correct[class_id]
                / class_total[class_id]
            )
        else:
            per_class_accuracy[class_id] = 0.0


    # Sample-weighted accuracy across ALL known samples
    overall_known_accuracy = (
        known_correct / known_total
        if known_total > 0
        else 0.0
    )

    # Macro-average over known classes
    mean_known_accuracy = (
        sum(per_class_accuracy.values())
        / len(per_class_accuracy)
    )

    # Unknown accuracy = unknown rejection rate
    unknown_accuracy = (
        unknown_correct / unknown_total
        if unknown_total > 0
        else 0.0
    )

    # Print results
    print("Validation results")
    print("------------------")

    print(f"Known samples:   {known_total}")
    print(f"Unknown samples: {unknown_total}")

    print("\nPer-class accuracy:")

    for class_id, accuracy in per_class_accuracy.items():

        print(
            f"  Class {class_id}: {accuracy:.4f} "
            f"({class_correct[class_id]}/{class_total[class_id]})"
        )

    print(
        f"\nOverall known accuracy "
        f"(sample-weighted): {overall_known_accuracy:.4f} "
        f"({known_correct}/{known_total})"
    )

    print(
        f"Mean known-class accuracy "
        f"(macro): {mean_known_accuracy:.4f}"
    )

    print(
        f"Unknown accuracy: "
        f"{unknown_accuracy:.4f} "
        f"({unknown_correct}/{unknown_total})"
    )