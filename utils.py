def check_mappings(train_dataset, val_dataset):

    train_mapping = train_dataset.class_to_idx
    val_mapping = val_dataset.class_to_idx

    # 1. Every training class must exist in validation
    missing_from_val = set(train_mapping) - set(val_mapping)

    # 2. Every shared class must have exactly the same ID
    mismatches = {}

    for class_name, train_id in train_mapping.items():
        val_id = val_mapping[class_name]

        if train_id != val_id:
            mismatches[class_name] = (train_id, val_id)

    if mismatches:
        print("ERROR: Class ID mismatches:")
        for class_name, (train_id, val_id) in mismatches.items():
            print(
                f"  {class_name}: "
                f"train={train_id}, validation={val_id}"
            )
        return False

    # 3. Find classes that exist only in validation
    extra_classes = set(val_mapping) - set(train_mapping)

    print("Mapping is consistent.")

    if extra_classes:
        print("Additional validation classes:")
        for class_name in sorted(extra_classes):
            print(
                f"  {class_name} -> {val_mapping[class_name]}"
            )
    else:
        print("No additional validation classes.")

    return True

