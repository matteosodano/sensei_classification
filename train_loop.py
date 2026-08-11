import torch

def train(model, train_loader, optimizer, criterion, train_classes, device, model_path):
    class_to_model_idx = {
        class_id: idx
        for idx, class_id in enumerate(train_classes)
    }

    best_accuracy = 0.0
    best_model_path = model_path

    for epoch in range(15):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
    
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Convert actual class IDs to model indices
            labels = torch.tensor(
                [class_to_model_idx[label.item()] for label in labels],
                device=device
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        print(
            f"Epoch [{epoch+1}/{15}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f}"
        )

        if train_accuracy >= best_accuracy:

            best_accuracy = train_accuracy

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"Saved new best model "
                f"(accuracy={best_accuracy:.4f})"
            )

        else:

            print(
                f"Accuracy dropped "
                f"({train_accuracy:.4f} < {best_accuracy:.4f}). "
                f"Stopping training."
            )

            break
