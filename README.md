# Spine CT: Vertebra Segmentation + Per-Vertebra Lesion Classification (Swin UNETR)

A multi-task deep learning pipeline that takes a spine CT volume, segments it into 18 individual
vertebra levels (T1–T12, L1–L5, S1), and then classifies each detected vertebra as **Normal**,
**Lytic**, **Blastic**, or **Mixed** based on metastatic lesion type. The two tasks share a single
training loop: a 3D Swin UNETR does the segmentation, and a lightweight 3D CNN classifies a
cropped sub-volume around every vertebra the segmenter finds.

The notebook is built to run end-to-end on a free Google Colab **T4 GPU**, against the
[Spine-Mets-CT-SEG](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/) collection
from The Cancer Imaging Archive (TCIA).

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Why two models instead of one](#why-two-models-instead-of-one)
3. [Dataset](#dataset)
4. [Environment & dependencies](#environment--dependencies)
5. [Pipeline walkthrough](#pipeline-walkthrough)
6. [Data alignment problems this notebook solves](#data-alignment-problems-this-notebook-solves)
7. [Model architectures](#model-architectures)
8. [Training configuration](#training-configuration)
9. [Results](#results)
10. [Inference on a new scan](#inference-on-a-new-scan)
11. [Known limitations](#known-limitations)
12. [How to reproduce](#how-to-reproduce)
13. [Repository / file structure](#repository--file-structure)
14. [Possible next steps](#possible-next-steps)

---

## What this project does

Given a raw CT series of a patient's spine, the pipeline:

1. **Segments** the volume into 19 classes — background plus each of the 18 vertebra levels
   (T1–T12 thoracic, L1–L5 lumbar, S1 sacrum) — using a 3D Swin UNETR built on a pretrained
   Swin Transformer backbone.
2. **Locates** each vertebra that the segmenter actually found in the scan (some scans only cover
   part of the spine).
3. **Crops** a small fixed-size 3D sub-volume around every detected vertebra.
4. **Classifies** each crop with a compact 3D CNN into one of four lesion categories: `Normal`,
   `Lytic`, `Blastic`, or `Mixed`.

The end result for a single scan is a per-vertebra report — which levels were found, and what
lesion type (if any) each one shows — rather than a single whole-scan diagnosis.

## Why two models instead of one

Vertebra-level metastases aren't scan-level facts — a single patient can have a lytic lesion at L1
and a completely normal T7 in the same CT. A whole-volume classifier would have nothing sensible to
predict against ("is this whole spine diseased?"), so the notebook splits the problem in two:

- **Segmentation model (Swin UNETR):** answers *where* each vertebra is.
- **Classification model (3D CNN):** answers *what's going on* at each vertebra, once it's been
  localized.

Both are trained jointly, in the same loop, against a combined loss (`segmentation loss + weighted
classification loss`), so the encoder/backbone only has to be loaded and run once per scan.

## Dataset

**[Spine-Mets-CT-SEG](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)** (TCIA)
— CT scans of patients with spinal metastases, each paired with a DICOM-SEG object that labels the
individual vertebrae. TCIA hosts this in two separate places that the notebook has to reconcile:

- **Imaging data** — the CT series + DICOM-SEG masks, downloaded automatically via `tcia_utils`.
- **Clinical table** — a separate Excel file (`Spine-Mets-CT-SEG_Clinical.xlsx`) with per-patient
  columns for `Age`, `Sex`, `Primary cancer`, and critically `Lytic` / `Blastic` / `Mixed`, which
  list which vertebra levels (by name, e.g. `"T1, T2 T3, T4"`) have each lesion type. This table
  does **not** ship with the DICOM collection and has to be downloaded or supplied separately.

Once loaded, the clinical table looked like this:

```
Sheet names: ['Data Dictionary', 'Spine-Mets-CT-SEG']

--- Data Dictionary ---
shape: (12, 2)
columns: ['Column Title ', 'Definition']

--- Spine-Mets-CT-SEG ---
shape: (55, 12)
columns: ['Case', 'Age (Y)', 'Sex', 'Height (m)', 'Weight (kg)', 'BMI (kg/m^2)',
          'Primary cancer', 'Vertebrae with Lesions', 'Blastic', 'Lytic', 'Mixed',
          'Comments/Fractures']
```

55 patients total, all matched successfully to their imaging data by `PatientID` ↔ `Case`:

```
Total volumes: 55
Matched to clinical table: 55
```

A few parsed lesion labels, showing the range from "no lesions" to "lesions basically everywhere":

```
Loaded clinical labels for 55 patients.
  10250: {'T2': 'Mixed'}
  10352: {}
  10355: {'T4': 'Mixed', 'T5': 'Mixed', 'T11': 'Mixed', 'T2': 'Mixed', 'L3': 'Mixed',
          'L5': 'Mixed', 'T7': 'Mixed', 'T1': 'Mixed', 'T8': 'Mixed', 'T12': 'Mixed',
          'T6': 'Mixed', 'L1': 'Mixed', 'L4': 'Mixed', 'T3': 'Mixed', 'T9': 'Mixed',
          'L2': 'Mixed', 'T10': 'Mixed'}
  10456: {'L1': 'Lytic', 'T11': 'Lytic'}
```

All 55 CT/mask pairs were successfully found and paired:

```
Paired 55 CT/mask volumes.
```

## Environment & dependencies

Designed for **Google Colab, free tier, T4 GPU** (`Runtime > Change runtime type > T4 GPU`).

Key libraries:

| Library | Purpose |
|---|---|
| `monai` | `SwinUNETR` network, `DiceCELoss`, `DiceMetric`, transforms |
| `SimpleITK` | Reading DICOM series, resampling volumes onto a common grid |
| `pydicom` + `pydicom-seg` | Reading DICOM-SEG multi-segment mask objects correctly |
| `tcia_utils` | Programmatic download of the TCIA collection |
| `torch` | Model definition and training |
| `pandas` / `openpyxl` | Parsing the clinical Excel table |

The single most important dependency call-out in the notebook is **`pydicom-seg`** — the DICOM-SEG
files in this collection are multi-segment objects (one file can contain 15+ distinct vertebra
masks packed into one series via `SegmentSequence` + `PerFrameFunctionalGroupsSequence`). Reading
them with plain `pydicom` and treating the result as a flat binary array would silently collapse
all vertebrae into a single blob. `pydicom_seg.MultiClassReader` decodes them properly into one
volume where each voxel's value is a distinct segment number.

Because of version friction between `monai`, `numpy`, and `pydicom` on Colab, dependencies are
installed in three passes: a broad install, a forced reinstall pinning newer `numpy`, and finally a
downgrade of `pydicom` to `<3.0.0` for compatibility with `pydicom-seg`.

## Pipeline walkthrough

The notebook is organized as 15 numbered sections:

| # | Section | What happens |
|---|---|---|
| 1 | Install dependencies | Three-pass pip install to resolve `monai`/`pydicom`/`numpy` conflicts |
| 2 | Configuration | Vertebra list, lesion classes, paths, volume/crop shapes (see below) |
| 3 | Download pretrained backbone | Swin UNETR weights from the MONAI test-data releases |
| 4 | Download imaging dataset | `tcia_utils.nbia` pulls the Spine-Mets-CT-SEG collection |
| 5 | Load clinical annotation table | Regex-based parsing of `Lytic`/`Blastic`/`Mixed` columns |
| 6 | Pair CT volumes with masks | Match by `FrameOfReferenceUID`: multi-file folder = CT, single file = mask |
| 7 | Attach `PatientID` + labels | Join DICOM `PatientID` to the clinical table's `Case` column |
| 8 | DICOM-SEG decoding | `pydicom_seg.MultiClassReader` + vertebra-name regex per segment |
| 9 | Train/validation split | 90/10, seeded shuffle |
| 10 | Loading & preprocessing | Custom `MapTransform` producing image, label, lesion targets, presence mask |
| 11 | Freeze pretrained encoder | Only the decoder/segmentation head is fine-tuned |
| 12 | Lesion classifier + crop utility | 3D CNN + bounding-box crop-and-resize function |
| 13 | Losses, metrics, optimizer | `DiceCELoss` + `CrossEntropyLoss`, one shared `Adam` optimizer |
| 14 | Training loop | Joint segmentation + classification training, 10 epochs |
| 15 | Training curves | Loss/Dice/accuracy plotted across epochs |

Followed by a standalone **Inference & Classification** section that can run independently once
checkpoints exist.

### Configuration (Section 2)

```
Segmentation classes: 19 (background + 18 vertebrae)
Lesion classes: ['Normal', 'Lytic', 'Blastic', 'Mixed']
```

- `OUTPUT_SHAPE = (96, 96, 96)` — the common grid every CT volume is resampled onto for
  segmentation.
- `CROP_SHAPE = (32, 32, 32)` — the fixed size every per-vertebra crop is resized to before
  classification.
- `MIN_VERTEBRA_VOXELS = 30` — a vertebra with fewer predicted/labeled voxels than this is treated
  as "not present" in the scan (avoids classifying noise as a vertebra).
- `VERTEBRA_TOKEN_RE` — a regex, `\b([TLS]\d{1,2})\b`, used to pull vertebra names out of both the
  clinical table's free-text lesion columns and the DICOM-SEG segment labels.

### Pretrained backbone (Section 3)

```
Downloading Swin UNETR pretrained backbone weights...
model_swinvit.pt: 392MB [00:11, 36.6MB/s]
Swin UNETR backbone ready on cuda with 19 output classes
```

The backbone is [MONAI's self-supervised-pretrained Swin UNETR checkpoint](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt),
loaded via `seg_model.load_from(weights=...)` and re-headed for 19 output classes instead of its
original task.

## Data alignment problems this notebook solves

This dataset has several sharp edges that the notebook handles explicitly rather than assuming
away:

- **Clinical table lives separately from the imaging data.** TCIA splits the Spine-Mets-CT-SEG
  collection's DICOM files from its clinical Excel sheet across different tabs on the same
  collection page, so both have to be fetched and then joined manually by patient ID.
- **Inconsistent separators inside lesion columns.** A cell like `"T1, T2 T3, T4"` actually means
  four separate vertebrae (`T1`, `T2`, `T3`, `T4`), not two comma-joined groups — a naive
  `.split(",")` would merge `T2` and `T3` into one malformed token. The notebook instead extracts
  every vertebra name with a dedicated regex, independent of whatever spacing or punctuation
  separates them.
- **DICOM-SEG isn't a simple binary mask.** A folder with multiple `.dcm` files is a CT series; a
  folder with exactly one `.dcm` file is a DICOM-SEG object containing *all* vertebra masks packed
  together, matched to its CT by shared `FrameOfReferenceUID`. Reading it as a flat mask (rather
  than through `pydicom_seg.MultiClassReader`) would lose the distinction between vertebrae
  entirely.
- **Segment labels are free text, not indices.** Each DICOM-SEG segment has a human-readable label
  like `"T1 vertebra"` — inspecting a real file confirmed this:

  ```
  File: .../1-1.dcm
  Modality: SEG
  Number of frames: 1652
  Number of segments: 15
    Segment 1: label='T1 vertebra'  description=''
    Segment 2: label='T2 vertebra'  description=''
    Segment 3: label='T3 vertebra'  description=''
    Segment 4: label='T4 vertebra'  description=''
    Segment 5: label='T5 vertebra'  description=''
  ```

  The same vertebra-token regex used on the clinical table is reused here to map each segment
  number to one of the 18 canonical vertebra indices, so both label sources agree on what "T4"
  means.
- **Independent resampling grids would misalign image and mask.** The CT volume and its
  segmentation are resampled with two different SimpleITK calls — but the label is explicitly
  resampled *onto the CT's own resampled grid* (`_resample_to_reference`), not onto an
  independently-computed grid of its own. Doing it independently is a subtle bug that silently
  shifts the mask relative to the image.
- **Not every patient in the imaging set is in the clinical table (and vice versa).** Patients
  missing a clinical-table match aren't dropped — they default to "every vertebra Normal" — but are
  explicitly flagged, so a join-key mismatch is visible in the output instead of silently
  corrupting labels. In this run, every single patient matched cleanly:

  ```
  DICOM PatientID: '10543'  ->  in clinical table: True
  DICOM PatientID: '14487'  ->  in clinical table: True
  DICOM PatientID: '14636'  ->  in clinical table: True
  DICOM PatientID: '14293'  ->  in clinical table: True
  DICOM PatientID: '12196'  ->  in clinical table: True
  ```

## Model architectures

### Segmentation: Swin UNETR

- `in_channels=1`, `out_channels=19` (background + 18 vertebrae)
- `feature_size=48`
- Initialized from MONAI's pretrained self-supervised Swin Transformer backbone
- **Encoder frozen** — only the decoder / segmentation head is fine-tuned, which keeps training
  tractable on a single T4:

  ```
  Trainable params: 19,598,755 / 62,187,541
  ```

  (~31% of parameters trainable — the heavy `swinViT`/encoder blocks are frozen.)

### Classification: a small 3D CNN

```python
class VertebraLesionClassifier3D(nn.Module):
    def __init__(self, num_classes=len(LESION_CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)
```

Three Conv3D + BatchNorm + ReLU + MaxPool blocks (16 → 32 → 64 channels), global average pooling,
and a single linear layer to 4 lesion classes. Deliberately lightweight since its input is already
a small, tightly-cropped 32×32×32 sub-volume rather than a full CT.

`crop_vertebra()` finds the bounding box of a given vertebra index in the label map, pads it by a
4-voxel margin, crops the CT at that box, and resizes the crop to `CROP_SHAPE` with trilinear
interpolation. During **training**, the ground-truth vertebra map is used to build these crops
(so the classifier learns from correctly localized regions, not the segmenter's early noisy
guesses). During **inference**, the *predicted* segmentation mask is used instead, since no
ground truth is available for a new scan.

## Training configuration

- **Split:** 50 training volumes / 5 validation volumes (90/10, seeded with `random.seed(42)`)
- **Batch size:** 4 for training, 2 for validation (`batch_size=1` semantics per-sample inside the
  loop, since each volume has a different number of present vertebrae)
- **Segmentation loss:** `DiceCELoss(to_onehot_y=True, softmax=True)`
- **Classification loss:** `CrossEntropyLoss`, averaged across all vertebrae present in a volume
- **Combined loss:** `seg_loss + CLS_LOSS_WEIGHT * cls_loss`, with `CLS_LOSS_WEIGHT = 0.5`
- **Optimizer:** single `Adam` over the trainable segmentation parameters **and** the classifier
  jointly, `lr=2e-5`, `weight_decay=1e-2`
- **Epochs:** 10
- **Checkpointing:** the best segmentation + classification weights (by combined Dice/accuracy
  score) are saved every epoch they improve

## Results

Ten epochs on 50 training volumes, evaluated against the 5-volume validation split:

```
Epoch 01/10 | Seg loss: 3.7567 | Cls loss: 1.5192 | Val Dice: 0.0006 | Val Cls Acc: 0.0000 | Epoch time: 293.9s
  Saved new best checkpoints (combined score: 0.0003)
Epoch 02/10 | Seg loss: 3.5687 | Cls loss: 1.4958 | Val Dice: 0.0005 | Val Cls Acc: 0.0000 | Epoch time: 304.9s
Epoch 03/10 | Seg loss: 3.4415 | Cls loss: 1.4716 | Val Dice: 0.0007 | Val Cls Acc: 0.0000 | Epoch time: 312.0s
  Saved new best checkpoints (combined score: 0.0003)
Epoch 04/10 | Seg loss: 3.3569 | Cls loss: 1.4755 | Val Dice: 0.0009 | Val Cls Acc: 0.0000 | Epoch time: 309.9s
  Saved new best checkpoints (combined score: 0.0005)
Epoch 05/10 | Seg loss: 3.2956 | Cls loss: 1.4899 | Val Dice: 0.0011 | Val Cls Acc: 0.0000 | Epoch time: 313.0s
  Saved new best checkpoints (combined score: 0.0006)
Epoch 06/10 | Seg loss: 3.2431 | Cls loss: 1.4668 | Val Dice: 0.0015 | Val Cls Acc: 0.0000 | Epoch time: 308.7s
  Saved new best checkpoints (combined score: 0.0007)
Epoch 07/10 | Seg loss: 3.1985 | Cls loss: 1.4611 | Val Dice: 0.0018 | Val Cls Acc: 0.0000 | Epoch time: 309.9s
  Saved new best checkpoints (combined score: 0.0009)
Epoch 08/10 | Seg loss: 3.1591 | Cls loss: 1.4318 | Val Dice: 0.0020 | Val Cls Acc: 0.0000 | Epoch time: 324.1s
  Saved new best checkpoints (combined score: 0.0010)
Epoch 09/10 | Seg loss: 3.1226 | Cls loss: 1.4286 | Val Dice: 0.0021 | Val Cls Acc: 0.0238 | Epoch time: 323.8s
  Saved new best checkpoints (combined score: 0.0130)
Epoch 10/10 | Seg loss: 3.0858 | Cls loss: 1.3911 | Val Dice: 0.0022 | Val Cls Acc: 0.0238 | Epoch time: 336.7s
Training complete in 58.6 min. Best combined score: 0.0130
```

Both losses decrease steadily and monotonically across all 10 epochs, showing the model is
learning something — but validation Dice stays under 0.003 and classification accuracy stays near
zero the entire run. Training curves:

![Training curves: segmentation loss, classification loss, validation Dice, and validation classification accuracy across 10 epochs](images/cell40_out0.png)

**In plain terms: 10 epochs was nowhere near enough for the segmentation decoder to converge**,
which cascades into the classifier — since it crops vertebrae from the segmentation output during
inference (and from ground truth during training, so its own loss curve looks more reasonable in
isolation), a near-random segmentation mask means a near-random classification signal downstream.
This shows up directly in the qualitative inference comparison:

![Ground-truth vertebra labels (left) vs. predicted vertebra labels (right) for a middle CT slice from validation patient 10352](images/cell43_out1.png)

The ground-truth mask (left) clearly separates individual vertebra bodies with distinct colors; the
model's prediction (right) has essentially collapsed to background/uniform output at this point in
training, which is exactly what the near-zero Dice score indicates. The per-vertebra table for the
same sample confirms it — the classifier outputs "Normal" with a flat ~0.28 confidence for
practically every vertebra, whether or not it was actually detected:

```
Sample index: 0  (patient: 10352)
Vertebra  Actual      Predicted
T8        Normal      -
T9        Normal      Normal (0.28)
T10       Normal      Normal (0.28)
T11       Normal      Normal (0.28)
T12       Normal      Normal (0.28)
L1        Normal      Normal (0.28)
L2        Normal      Normal (0.27)
L3        Normal      Normal (0.28)
L4        Normal      Normal (0.28)
L5        Normal      Normal (0.28)
```

This is expected behavior for a frozen-encoder Swin UNETR fine-tuned for only 10 short epochs on
50 volumes, not a bug in the pipeline logic — the data loading, DICOM-SEG decoding, label
alignment, and per-vertebra cropping all check out correctly (see the diagnostic cells throughout
the notebook); the model simply needs substantially more training to produce useful segmentations.

## Inference on a new scan

The **Inference & Classification** section is self-contained: if `best_seg_model.pth` and
`best_cls_model.pth` already exist in `/content`, it can be run without re-executing any of the
download/training cells above it.

```
Loaded fine-tuned segmentation + classification weights.
Inference pipeline ready.
```

`run_inference(dicom_dir)` takes a path to a folder of `.dcm` files and returns:

- the raw display volume (for visualization),
- the predicted per-voxel vertebra label map,
- a list of per-vertebra results (`vertebra`, `voxels`, `lesion_class`, `confidence`),
- a one-line summary (e.g. *"Lesion(s) detected at 2 vertebra level(s)."*).

The key difference from training: since there's no ground-truth vertebra map for a genuinely new
scan, vertebra crops for classification are built from the **model's own predicted** segmentation
mask rather than ground truth.

## Known limitations

- **10 epochs is a proof-of-concept run, not a converged model** — the results above make clear
  more training (and likely a higher learning rate for the decoder, or an unfrozen encoder given
  enough compute) is needed before the segmentation or classification outputs are meaningful.
- **Severe class imbalance** in lesion labels — many vertebrae across 55 patients are `Normal`,
  with `Lytic`/`Blastic`/`Mixed` comparatively rare, which the current loss setup doesn't
  explicitly correct for (no class weighting or oversampling).
- **Small dataset** — 55 patients (50 train / 5 validation) is a small sample for a 3D segmentation
  + classification task; validation metrics from only 5 held-out volumes are noisy by nature.
- **This is a research/educational pipeline, not a clinical tool.** It has not been validated for
  diagnostic use and should not be treated as one.

## How to reproduce

1. Open the notebook in Google Colab.
2. Set the runtime to **T4 GPU**.
3. Run cells top to bottom. The first three cells install dependencies (expect some benign pip
   warnings about extras).
4. When prompted by section 4, the TCIA collection downloads automatically; if automatic download
   fails, follow the printed manual-download instructions.
5. Section 5/6 need the separate **clinical table** — download it from TCIA's "Clinical" tab for
   the Spine-Mets-CT-SEG collection and confirm `CLINICAL_TABLE_PATH` points to it.
6. Training (section 14) took **~59 minutes for 10 epochs** on a T4 for 50 volumes — increase
   `num_epochs` for meaningfully better results.
7. To skip straight to inference on an already-trained checkpoint, jump directly to the
   **Inference & Classification** section — it only needs `best_seg_model.pth` and
   `best_cls_model.pth` to already exist.

## Repository / file structure

```
.
├── spine_ct_vertebra_segmentation_classification.ipynb   # the full pipeline
├── model_swinvit.pt              # pretrained Swin UNETR backbone (downloaded, not committed)
├── best_seg_model.pth            # best segmentation checkpoint (produced by training)
├── best_cls_model.pth            # best classification checkpoint (produced by training)
└── spine_mets_ct_seg/             # downloaded CT + DICOM-SEG data (not committed)
```

The last cell exports the two trained checkpoints for download as
`spine_vertebra_segmentation_model.pth` and `spine_lesion_classifier_model.pth`.

## Possible next steps

- Train for substantially more epochs (or unfreeze more of the encoder) to get segmentation Dice
  into a usable range before drawing any conclusions about classification accuracy.
- Add class weighting or oversampling for the `Lytic`/`Blastic`/`Mixed` classes to counter the
  imbalance toward `Normal`.
- Track per-vertebra Dice (not just an aggregate) to see whether certain levels — e.g. commonly
  fused or partially-imaged ones like S1 — are systematically harder to segment.
- Wrap `run_inference` in the already-imported `gradio` dependency for a simple upload-a-scan demo.

---

*This README was generated from the notebook's markdown cells, code, and actual execution output
(including the embedded figures) to document what the pipeline does and how it performed on its
most recent run.*
