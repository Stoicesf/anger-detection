"""Train the DS-CNN on the extracted features (8:1:1 split)."""

import json
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from anger_detection.common import (
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TIME_FRAMES,
    ensure_dirs,
)
from anger_detection.models.model_tf import create_dscnn_model


def augment(x, y):
    """Fast augmentation: batch-wise time shift, spec-augment freq mask, noise."""
    x = tf.cast(x, tf.float32)
    pad = 16
    x = tf.pad(x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
    offset = tf.random.uniform([], 0, 2 * pad + 1, dtype=tf.int32)
    x = x[:, offset : offset + TIME_FRAMES, :, :]
    # spec-augment: zero a random band of 4 consecutive mel bins
    f0 = tf.random.uniform([], 0, 40 - 4, dtype=tf.int32)
    mask = tf.concat(
        [tf.ones([1, 1, f0, 1]), tf.zeros([1, 1, 4, 1]), tf.ones([1, 1, 40 - f0 - 4, 1])],
        axis=2,
    )
    x = x * mask
    noise = tf.random.normal(tf.shape(x), stddev=0.002)
    return x + noise, y


def main() -> None:
    ensure_dirs()
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    file_ids = np.load(PROCESSED_DIR / "file_ids.npy")
    print(f"Loaded X={X.shape} y={y.shape}")

    # file-level stratified split (8:1:1) so crops of one file stay in one split
    unique_ids = np.unique(file_ids)
    file_label = np.array([y[np.where(file_ids == fid)[0][0]] for fid in unique_ids])
    ids_train, ids_tmp, _, _ = train_test_split(
        unique_ids, file_label, test_size=0.2, stratify=file_label, random_state=42
    )
    label_map = dict(zip(unique_ids, file_label))
    y_tmp = np.array([label_map[fid] for fid in ids_tmp])
    ids_val, ids_test, _, _ = train_test_split(
        ids_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    train_mask = np.isin(file_ids, ids_train)
    val_mask = np.isin(file_ids, ids_val)
    test_mask = np.isin(file_ids, ids_test)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print(f"files: train={len(ids_train)} val={len(ids_val)} test={len(ids_test)}")
    print(f"crops: train={len(X_train)} val={len(X_val)} test={len(X_test)} (8:1:1)")

    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights = np.clip(weights, None, 10.0)
    class_weight = {int(c): float(w) for c, w in enumerate(weights)}
    print("class weights:", class_weight)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(2048, seed=42)
        .batch(32)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    model = create_dscnn_model(
        input_shape=(TIME_FRAMES, 40, 1), num_classes=NUM_CLASSES, width=24, dropout=0.15
    )
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / "dscnn_best.keras", save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=6, min_lr=1e-5, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR / "history.csv"),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")

    model.save(MODELS_DIR / "dscnn_final.keras")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  loss: {test_loss:.4f}")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
                "epochs_run": len(history.history["loss"]),
                "class_names": CLASS_NAMES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
