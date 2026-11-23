from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps
ROOT = Path(__file__).resolve().parent
SIZE = 32
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.webp'}
def image_array(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert('RGB').resize((SIZE, SIZE))
        return np.asarray(img, dtype=np.uint8)
def load_labels(csv_path, folder_names):
    labels = {name: f'Class {name}' for name in folder_names}
    if not csv_path.exists():
        return labels, 'No CSV found; using class IDs.'
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
        normalized = {str(c).lower().replace(' ', '').replace('_', ''): c for c in df.columns}
        id_col = next((normalized[k] for k in ['classid', 'class', 'id', 'labelid', 'target'] if k in normalized), None)
        name_col = next((normalized[k] for k in ['name', 'signname', 'classname', 'labelname', 'description', 'meaning', 'sign', 'label'] if k in normalized and normalized[k] != id_col), None)
        if id_col is None or name_col is None:
            return labels, f'Could not identify ID/name columns in {list(df.columns)}; using class IDs. Edit models/labels.json after training to add names.'
        for _, row in df.iterrows():
            raw = str(row[id_col]).strip()
            try:
                key = str(int(float(raw)))
            except ValueError:
                key = raw
            if key in labels and pd.notna(row[name_col]):
                labels[key] = str(row[name_col]).strip()
        return labels, f'Loaded label mapping from {csv_path.name}.'
    except Exception as exc:
        return labels, f'CSV mapping unavailable ({exc}); using class IDs.'
