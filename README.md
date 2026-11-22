# DIT5411-LAM-Kai-Chun

Chinese handwritten character recognition using Python and TensorFlow.

- **Author:** LAM Kai Chun
- **Course:** THEi DIT5411 Machine Learning
- **Assignment:** Character Recognition, 2026

## 1. Project overview

This project implements convolutional neural networks (CNNs) to classify
traditional Chinese handwritten characters into **13,065 output classes**.

The workflow includes dataset inspection, deterministic sample splitting,
training-only image augmentation, model development, checkpoint saving,
and validation evaluation.

**Current result:** the saved LeakyReLU CNN achieved **66.09% Top-1**
and **88.82% Top-5 accuracy on a fixed 4,096-image validation subset**.

These are **validation results, not test-set results**. Full test evaluation
was not completed, and no best model based on test accuracy is claimed.

## 2. Dataset and preprocessing

The assignment identifies the following dataset:

[AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

The archive used in this project consisted of 12 parts named
`all_data.zip.001` through `all_data.zip.012`.

The saved dataset audit reported:

| Property | Observed value |
|---|---:|
| Chinese character classes | 13,065 |
| Original PNG images | 684,677 |
| Image dimensions | 300 × 300 |
| Original colour mode | RGB |
| Minimum samples per class | 16 |
| Maximum samples per class | 61 |
| Images successfully decoded | 684,677 |
| Image decoding failures | 0 |

These counts describe the archive actually inspected, rather than assuming
exactly 50 samples per class.

### Filename encoding

The Colab preparation workflow used ZIP filename encoding `cp950`.
The initial Kaggle extraction produced incorrectly decoded Chinese filenames.
A subsequent repair matched filenames against the manifest, checked for
ambiguity and collisions, and reported zero missing manifest paths afterward.

This historical repair applied to extracted working files. It should not
be rerun unnecessarily on correctly named images.

### Model input

Training and validation use:

1. PNG decoding to one grayscale channel.
2. Conversion to floating-point pixel values, nominally in `[0, 1]`.
3. Bilinear resizing to **128 × 128**, with antialiasing.
4. Input shape **128 × 128 × 1**.

White background is represented by `1.0` and black strokes by `0.0`.

Class labels are zero-based integers from `0` to `13064`.
`class_mapping.json` maps each class ID to its dataset folder and character.
Folder numbers must not be assumed to equal model output indices.

## 3. Training, validation and test splits

Samples within each class are sorted by the **numeric filename suffix**.
For example, `_2.png` precedes `_10.png`.

For the **13,022 classes with more than 40 images**:

- The first 40 available samples form the training pool.
- Four samples per class are selected from that pool for development validation.
- The remaining 36 samples are used for development training.
- All samples after the first 40 are assigned to the test set.

Validation selection uses NumPy's random generator with seed `2026`.

### Classes with fewer than 40 samples

The audit identified **43 classes containing 1,247 images in total**,
each with fewer than 40 samples.

For these classes, all available images form the training pool.
Four images per class are reserved for validation; the rest are used for
development training. No images from these classes are assigned to test.

This exception was documented but **has not been approved by the instructor**.

### Actual development split

| Split | Original images | Classes represented |
|---|---:|---:|
| Training | 469,867 | 13,065 |
| Validation | 52,260 | 13,065 |
| Test | 162,550 | 13,022 |

The original test assignment was preserved when creating development
validation. Test images were not used to build the augmentation plan.

`original_samples_manifest.csv` and `split_draft_summary.json` preserve
the earlier draft, where the 43 short classes were marked `pending`.
`development_manifest.csv` records the later development allocation.

## 4. Image augmentation

### Per-class training plan

A fixed augmentation plan was created with **200 training items per class**:

| Item type | Count |
|---|---:|
| Original development-training items, without affine augmentation | 469,867 |
| Items requiring affine augmentation | 2,143,133 |
| Total training items | 2,613,000 |

Each original development-training image appears once without affine
augmentation. Additional source images are selected by cycling through
a shuffled list of that class's training images until the class reaches
200 items.

The preparation code checks that:

- All augmentation sources belong to the training split.
- Every class has exactly 200 planned items.
- The test assignment remains unchanged.

The plan is saved in `development_augmentation_plan.npz`.

### Transformations used during training

`training_core.py` applies the following transformations on demand:

| Transformation | Range or setting |
|---|---|
| Rotation | −9° to +9° |
| Uniform scaling | 0.92 to 1.08 |
| Horizontal translation | Up to ±5% of image width |
| Vertical translation | Up to ±5% of image height |
| Interpolation | Bilinear |
| Boundary fill | Constant white, `1.0` |
| Horizontal/vertical flips | Not used |
| Shearing | Not used |

The affine transform uses
`tf.raw_ops.ImageProjectiveTransformV3`.
Stateless random seeds make each planned item's transformation reproducible.
Shuffling changes item order, not the stored transform assigned to each item.

Augmented images are generated in the data pipeline rather than all being
saved as separate PNG files. The ReLU baseline completed one full traversal
of this training dataset. The LeakyReLU runs used partial training segments;
their update counts do not establish complete per-class coverage.

### Preview files

- `augmentation_preview.png`: an early demonstration using Keras random
  rotation, translation and zoom layers.
- `training_batch_preview.png`: sample images from the training pipeline.

The early Keras preview is illustrative; its implementation is not identical
to the final combined affine transform.

`experiment_config.json` retains the preparation-stage text
“Plan only; transformed images not generated yet”.
Later notebook outputs document pipeline execution and Kaggle training.

## 5. Model architectures

All models produce **13,065 Softmax outputs**.

### A. Baseline CNN — ReLU

Four convolution/pooling blocks:

| Block | Convolution | Pooling output |
|---|---|---|
| 1 | 3 × 3, 16 filters, ReLU | 64 × 64 × 16 |
| 2 | 3 × 3, 32 filters, ReLU | 32 × 32 × 32 |
| 3 | 3 × 3, 64 filters, ReLU | 16 × 16 × 64 |
| 4 | 3 × 3, 128 filters, ReLU | 8 × 8 × 128 |

All convolutions use `padding="same"`.
Each block uses 2 × 2 max pooling.

Classification head:

`Flatten(8192) → Dense(256, ReLU) → Dropout(0.4) → Dense(13065, Softmax)`

Total model parameters: **5,552,265**.

One complete development epoch was recorded.
Validation performance remained close to chance.
A diagnostic check on eight training images found zero activations
in the last convolution and hidden dense layer, consistent with
collapsed ReLU features on those probe images.

### B. Baseline CNN — LeakyReLU

The same convolution, pooling and classification-head dimensions are used,
but convolutional and hidden dense ReLU activations are replaced with
**LeakyReLU, negative slope `0.1`**.

The output activation remains Softmax, and Dropout remains `0.4`.
Total model parameters remain **5,552,265**.

The model was created separately, rather than replacing the activation
inside the saved ReLU checkpoint.

Training was performed in three 15,000-update segments, reaching
**45,000 optimizer updates**.

### C. Deep CNN

Defined in `training_core.py` as `deep`.

Each of four blocks uses:

`Conv2D → BatchNormalization → ReLU → Conv2D → BatchNormalization → ReLU → MaxPooling`

Block filters are 16, 32, 64 and 128.

The head is:

`Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(13065, Softmax)`

The architecture is implemented, but no completed training or evaluation
results were found in the supplied records.

### D. Residual CNN

Defined in `training_core.py` as `residual`.

Each of four blocks uses:

- A 1 × 1 convolution to project the shortcut to the required channel count.
- Two 3 × 3 convolutions with BatchNormalization.
- ReLU after the first convolution/normalization and after residual addition.
- Max pooling after the residual block.

Block filters are 16, 32, 64 and 128.
The classification head matches the Deep CNN head.

The architecture is implemented, but no completed training or evaluation
results were found in the supplied records.

## 6. Training setup

| Setting | ReLU baseline run | LeakyReLU runs |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.0001 |
| Batch size | 32 | 32 |
| Loss | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy |
| Recorded training | 1 full development epoch | 45,000 cumulative updates |
| Training data | Augmented development dataset | Partial segments of the same pipeline |

Kaggle outputs report TensorFlow **2.21.0** and two NVIDIA Tesla T4 GPUs.
The code does not implement distributed multi-GPU training; the model
creation for the recorded main runs explicitly selected GPU 0.

Continuation restored the saved model and optimizer state.
The dataset iterator state was not restored.

**45,000 updates must not be described as 45,000 epochs.**
They also do not imply that every planned training item was visited.

## 7. Recognition results

### Validation results

| Model/checkpoint | Evaluation split | Images | Top-1 accuracy | Top-5 accuracy |
|---|---|---:|---:|---:|
| ReLU baseline, 1 epoch | Full development validation | 52,260 | ≈0.00765% | ≈0.03827% |
| LeakyReLU, 15,000 updates | Fixed validation subset | 4,096 | 2.6611% | 7.9346% |
| LeakyReLU, 30,000 updates | Fixed validation subset | 4,096 | 31.4697% | 59.5459% |
| LeakyReLU, 45,000 updates | Fixed validation subset | 4,096 | 66.0889% | 88.8184% |

The three LeakyReLU rows are training stages of the same architecture,
not three independent architectures.

The notebooks use the same validation sampling procedure and seed for
these stages. The final subset is saved as `validation_subset.csv`.

The ReLU and LeakyReLU rows use different validation sizes and training
budgets, so they are not a controlled model comparison on a common test set.

Final LeakyReLU validation loss: **1.4340**.

Top-1 measures whether the highest-scoring class is correct.
Top-5 measures whether the correct class appears among the five
highest-scoring predictions.

The 100% result in the small-batch diagnostic is memorization of
32 training images, not validation or test accuracy.

### Test-set accuracy

| Model | Test Top-1 accuracy | Status |
|---|---|---|
| ReLU baseline | N/A | Test evaluation not completed |
| LeakyReLU baseline | N/A | Test evaluation not completed |
| Deep CNN | N/A | No completed training/test results |
| Residual CNN | N/A | No completed training/test results |

The final run record explicitly states `test_evaluated: false`.
A later complete-test attempt was interrupted during image-directory
discovery, before model inference.

**The assignment requirement to report each model's test accuracy
and select the best model on the test set remains incomplete.**

## 8. Project files

The submission is being assembled. Files listed below are the intended
layout; upload and download availability must be checked before submission.

```text
DIT5411-LAM-Kai-Chun/
├── README.md
├── .gitignore
├── training_core.py
├── class_mapping.json
├── experiment_config.json
├── original_samples_manifest.csv
├── development_manifest.csv
├── split_draft_summary.json
├── development_augmentation_plan.npz
├── baseline_cnn_architecture.json
├── baseline_cnn_summary.txt
├── augmentation_preview.png
├── training_batch_preview.png
├── notebooks/
│   ├── google .ipynb
│   └── kaggle.ipynb
├── results/
│   └── baseline_leaky_continuation/
│       ├── run_config.json
│       ├── continuation_results.json
│       └── validation_subset.csv
└── models/
    └── baseline_leaky_continuation/
        └── last.keras
```

The large model and manifest files may be distributed through GitHub
Releases rather than ordinary browser uploads.
Their download links are pending publication.

`last.keras` is the final saved LeakyReLU checkpoint.
It is not labelled as the best checkpoint on the test set.

## 9. Environment and execution notes

### Dependencies

The recorded workflow uses:

- Python
- TensorFlow / Keras — recorded TensorFlow version: 2.21.0
- NumPy
- pandas
- Pillow
- Matplotlib
- tqdm
- IPython / Jupyter or a hosted notebook environment

The saved environment outputs include Python 3.13.x.
An independently verified, pinned dependency file has not yet been provided.

### Notebook roles

- `google .ipynb`: archive inspection, extraction, dataset audit,
  class mapping, split creation, augmentation-plan generation,
  model smoke test, and training-pipeline preparation.
- `kaggle.ipynb`: dataset restoration, filename repair, GPU checks,
  baseline training, diagnostics, LeakyReLU training,
  continuation and validation evaluation.

These notebooks preserve the development history.
They are not yet a cleaned, one-click “Run All” pipeline.

Before execution:

1. Obtain the dataset from an authorized source and prepare the archive files.
2. Configure paths for your own environment.
3. Use CP950-aware extraction where required.
4. Follow preparation dependencies in order.
5. Do not rerun extraction, filename repair or training cells unnecessarily.
6. Use a new output directory when starting a new experiment.

The historical GitHub download attempt in the Colab notebook failed.
The successful preparation used archive parts uploaded through Google Drive.
Private Drive folders and Kaggle Input paths are not public download links.

`training_core.py` defines functions; running it by itself does not start
training or evaluate a checkpoint.

For the Kaggle workflow, the imported module's `PROJECT` and `DATA`
settings are overridden with Kaggle working paths.

The LeakyReLU construction is in `kaggle.ipynb`.
`training_core.build_model("baseline")` constructs the ReLU version,
not the saved LeakyReLU model.

## 10. Limitations and unfinished work

- Complete test-set evaluation has not been performed.
- Deep and Residual CNN architectures are defined but have no completed
  training/evaluation records.
- The 43 short classes have no test examples under the adopted policy.
- The development models reserve four images per class for validation.
  The proposed final refit using train plus validation has not been completed.
- LeakyReLU training used partial dataset segments with restarted iterators.
- Development notebooks contain diagnostics, historical paths and manual
  steps that require cleanup for straightforward reproduction.
- Public artifact uploads and download links are still being finalized.

## 11. References

### Dataset

- [Traditional Chinese Handwriting Dataset — AI-FREE-Team](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

### Framework documentation

- [TensorFlow documentation](https://www.tensorflow.org/guide)
- [TensorFlow tf.data API](https://www.tensorflow.org/api_docs/python/tf/data)
- [Keras Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
- [Keras LeakyReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU)
- [ImageProjectiveTransformV3](https://www.tensorflow.org/api_docs/python/tf/raw_ops/ImageProjectiveTransformV3)
- [Python zipfile documentation](https://docs.python.org/3/library/zipfile.html)

### References supplied in the assignment

- [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV)
- [Handwritten Digit Recognition using Keras and TensorFlow](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow)

The assignment examples provide background references.
This project's final training augmentation is implemented in TensorFlow,
not OpenCV.

Dataset and third-party materials retain their original licensing terms.
Public repository visibility does not itself grant an open-source license.
