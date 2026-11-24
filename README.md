# Indian Traffic Sign Recognition — Small CNN

Updated for your dataset: Images/0, Images/1, ... and traffic_sign.csv.
The actual dataset and a trained model are NOT included. Use the dataset you already downloaded.

## Windows quick start (Python 3.11 recommended)
Open a terminal inside this extracted project folder:

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

If PowerShell blocks activation, open Command Prompt instead, or use
`.venv\Scripts\python.exe` instead of `python` and `.venv\Scripts\python.exe -m streamlit run app.py` for the app.

### Train directly using your existing dataset (no copying required)

    python train.py --data-dir "D:\Downloads\Indian-Traffic Sign-Dataset"

Use your actual folder name if it differs. The selected folder must contain Images/ and traffic_sign.csv.

### Alternatively copy the data

    data/
      Images/
        0/
        1/
        ...
      traffic_sign.csv

Then run:

    python train.py

### Launch the app after training

    python -m streamlit run app.py

Open the local address printed by Streamlit. Select an image from any class folder.
To make a command-line prediction:

    python predict.py "D:\Downloads\Indian-Traffic Sign-Dataset\Images\0\your-image.png"

Replace your-image.png with a real filename.

## What changed from the GTSRB version
- Reads Images/<numeric class ID> rather than Train/ and Test.csv.
- Discovers classes automatically; no fixed 43-class output or German label dictionary.
- Maps numeric class folders to consecutive model output indices and saves the mapping.
- Tries common CSV class-ID and sign-name column headings. If not recognized, safely displays Class N rather than guessing names. Inspect models/labels.json after training; edit only the name fields if needed, then restart Streamlit.
- Per-class train/validation/test splitting, with approximately 64/16/20 percent proportions (rounding differs for small classes).
- Compact CNN with input scaling saved inside the model; no double normalization in prediction.
- Loads images as uint8 to reduce preprocessing memory.
- Saves model, label mapping, split manifest, actual metrics, report, curves, and confusion matrix.
- Requires at least seven readable images per class; reports problems instead of duplicating examples.

## Lightweight run

    python train.py --data-dir "D:\Downloads\Indian-Traffic Sign-Dataset" --epochs 10 --batch-size 16

Fewer epochs may reduce accuracy. Smaller batches do not eliminate the memory used by loading the whole dataset.
No measured accuracy or runtime is promised: this archive has not been trained against your dataset.

## Evaluation limitations
Image-level random splitting can put near-duplicate photographs of the same physical sign in different partitions. If your data includes video frames, augmented duplicates, or scene groups, use group-based splitting before reporting strong generalization results. This project does not provide a physical-sign-independent benchmark.
The app always selects one of the trained classes; it has no reliable unknown-sign rejection and is not an autonomous-driving safety system.

## Troubleshooting
- Images not found: --data-dir must point to the parent of Images, not Images itself.
- Model missing: finish training before starting the app.
- CSV names unavailable: send the CSV column headings or edit models/labels.json names; do not use the old GTSRB class dictionary.
- TensorFlow installation failure: create a fresh Python 3.11 virtual environment.
- A class has too few images: add real images or explicitly remove that class before training. Do not make copies just to pass the split check.

Only Python syntax and archive contents were checked during packaging. Dependencies, training, UI execution, and accuracy were not tested here.
