from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT = Path("/content/DIT5411_project")
DATA = Path("/content/handwriting_dataset/cleaned_data")
SIZE = 128
CLASSES = 13065
SEED = 2026

def load_metadata():
    frame = pd.read_csv(
        PROJECT / "development_manifest.csv",
        encoding="utf-8-sig",
        keep_default_na=False
    ).sort_values("sample_index").reset_index(drop=True)

    expected = np.arange(len(frame))
    if not np.array_equal(frame["sample_index"].to_numpy(), expected):
        raise RuntimeError("Manifest sample indices are not contiguous.")

    paths = tf.constant([
        str(DATA / path) for path in frame["relative_path"]
    ])

    with np.load(PROJECT / "development_augmentation_plan.npz") as plan:
        plan = {key: plan[key].copy() for key in plan.files}

    return frame, paths, plan


def affine_augment(image, seed):
    # Stateless randomness gives each planned item a reproducible transform.
    random_values = tf.random.stateless_uniform(
        [4], seed=tf.stack([tf.constant(SEED, tf.int32), seed])
    )

    angle = (random_values[0] * 2.0 - 1.0) * (9.0 * np.pi / 180.0)
    scale = 0.92 + random_values[1] * 0.16
    tx = (random_values[2] * 2.0 - 1.0) * 0.05 * SIZE
    ty = (random_values[3] * 2.0 - 1.0) * 0.05 * SIZE

    # ImageProjectiveTransform uses an output-to-input mapping.
    a = tf.cos(angle) / scale
    b = tf.sin(angle) / scale
    center = (SIZE - 1.0) / 2.0

    transform = tf.stack([
        a, b, center - a * (center + tx) - b * (center + ty),
        -b, a, center + b * (center + tx) - a * (center + ty),
        0.0, 0.0
    ])

    result = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([SIZE, SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=tf.constant(1.0, tf.float32)
    )[0]

    return tf.ensure_shape(result, [SIZE, SIZE, 1])


def make_dataset(frame, paths, plan, split, batch_size=32):
    if split == "train":
        indices = plan["source_index"]
        labels = plan["class_id"]
        flags = plan["augment"]
        seeds = plan["seed"]

        # The stored plan is grouped by class. Globally permute it first
        # so a small shuffle buffer does not keep adjacent classes together.
        order = np.random.default_rng(SEED).permutation(len(indices))
        arrays = (
            indices[order], labels[order],
            flags[order], seeds[order]
        )
    else:
        subset = frame[frame["split"] == split]
        if subset.empty:
            raise ValueError(f"Empty split: {split}")

        arrays = (
            subset["sample_index"].to_numpy(dtype=np.int32),
            subset["class_id"].to_numpy(dtype=np.int32),
            np.zeros(len(subset), dtype=bool),
            np.zeros(len(subset), dtype=np.int32)
        )

    dataset = tf.data.Dataset.from_tensor_slices(arrays)

    if split == "train":
        dataset = dataset.shuffle(
            32768, seed=SEED, reshuffle_each_iteration=True
        )

    def decode(index, label, augment, seed):
        contents = tf.io.read_file(tf.gather(paths, index))
        image = tf.io.decode_png(contents, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(
            image, [SIZE, SIZE], antialias=True
        )
        image = tf.cond(
            augment,
            lambda: affine_augment(image, seed),
            lambda: image
        )
        return tf.ensure_shape(image, [SIZE, SIZE, 1]), label

    dataset = dataset.map(
        decode,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(name):
    layers = tf.keras.layers
    inputs = tf.keras.Input((SIZE, SIZE, 1))
    x = inputs

    if name == "baseline":
        for filters in [16, 32, 64, 128]:
            x = layers.Conv2D(
                filters, 3, padding="same", activation="relu"
            )(x)
            x = layers.MaxPooling2D()(x)

    elif name == "deep":
        for filters in [16, 32, 64, 128]:
            for _ in range(2):
                x = layers.Conv2D(
                    filters, 3, padding="same", use_bias=False
                )(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)

    elif name == "residual":
        for filters in [16, 32, 64, 128]:
            shortcut = layers.Conv2D(
                filters, 1, padding="same", use_bias=False
            )(x)

            y = layers.Conv2D(
                filters, 3, padding="same", use_bias=False
            )(x)
            y = layers.BatchNormalization()(y)
            y = layers.Activation("relu")(y)
            y = layers.Conv2D(
                filters, 3, padding="same", use_bias=False
            )(y)
            y = layers.BatchNormalization()(y)

            x = layers.Add()([shortcut, y])
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
    else:
        raise ValueError("Choose baseline, deep, or residual.")

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(
        CLASSES, activation="softmax", dtype="float32"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name="top5_accuracy"
            )
        ]
    )
    return model


def require_gpu():
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "No GPU detected. Full training was not started."
        )


def train_model(name, train_ds, validation_ds,
                epochs, initial_epoch=0, resume=False):
    require_gpu()
    run_dir = PROJECT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.keras"
    last_path = run_dir / "last.keras"

    if not resume and (best_path.exists() or last_path.exists()):
        raise RuntimeError(
            "Existing checkpoints found. Use resume=True or a new run folder."
        )

    if resume:
        if not last_path.exists():
            raise FileNotFoundError("No last.keras checkpoint to resume.")
        model = tf.keras.models.load_model(last_path)
    else:
        tf.keras.utils.set_random_seed(SEED)
        model = build_model(name)

    # Preserve the previous best validation accuracy after a restart.
    previous_best = None
    log_path = run_dir / "history.csv"
    if resume and log_path.exists() and best_path.exists():
        history = pd.read_csv(log_path)
        previous_best = float(history["val_accuracy"].max())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            initial_value_threshold=previous_best
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(last_path), save_best_only=False
        ),
        tf.keras.callbacks.CSVLogger(
            str(log_path), append=resume
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )
    return model


def evaluate_checkpoint(checkpoint, dataset, output_path):
    model = tf.keras.models.load_model(checkpoint)
    results = model.evaluate(dataset, return_dict=True)
    Path(output_path).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
