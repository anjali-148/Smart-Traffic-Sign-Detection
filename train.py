import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from common import ROOT, SIZE, EXTENSIONS, image_array, load_labels

def main():
    parser = argparse.ArgumentParser(description='Train on Images/<class ID>/<image>')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    data = args.data_dir.resolve()
    image_dir = data / 'Images'
    if not image_dir.is_dir():
        raise SystemExit(f'Missing {image_dir}. Use --data-dir pointing to the folder containing Images and traffic_sign.csv.')
    folders = sorted([p for p in image_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if len(folders) < 2:
        raise SystemExit('Need at least two numeric class folders inside Images.')
    paths, targets = [], []
    for idx, folder in enumerate(folders):
        candidates = sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS)
        valid = []
        for path in candidates:
            try:
                image_array(path)
                valid.append(str(path))
            except Exception as exc:
                print(f'Skipping unreadable image {path}: {exc}')
        print(f'Class {folder.name}: {len(valid)} images')
        if len(valid) < 7:
            raise SystemExit(f'Class {folder.name} needs at least 7 valid images for the train/validation/test split. No images were duplicated.')
        paths.extend(valid)
        targets.extend([idx] * len(valid))
    paths, targets = np.array(paths), np.array(targets)
    # Split each class separately to preserve every class in each partition.
    train_idx, val_idx, test_idx = [], [], []
    for idx in range(len(folders)):
        ids = np.flatnonzero(targets == idx)
        a, b = train_test_split(ids, test_size=0.2, random_state=42)
        a, c = train_test_split(a, test_size=0.2, random_state=42)
        train_idx.extend(a); val_idx.extend(c); test_idx.extend(b)
    def arrays(indices):
        return np.stack([image_array(paths[i]) for i in indices]), targets[indices]
    xtr, ytr = arrays(train_idx)
    xv, yv = arrays(val_idx)
    xt, yt = arrays(test_idx)
    output = ROOT / 'models'
    output.mkdir(exist_ok=True)
    mapping, message = load_labels(data / 'traffic_sign.csv', [p.name for p in folders])
    print(message)
    labels = [{'index': i, 'class_id': f.name, 'name': mapping[f.name]} for i, f in enumerate(folders)]
    (output / 'labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    split_rows = [{'path': paths[i], 'class_id': folders[targets[i]].name, 'split': split} for split, ids in [('train',train_idx), ('validation',val_idx), ('test',test_idx)] for i in ids]
    pd.DataFrame(split_rows).to_csv(output / 'split_manifest.csv', index=False)
    model = tf.keras.Sequential([
        layers.Input((SIZE, SIZE, 3)), layers.Rescaling(1./255),
        layers.RandomRotation(0.04), layers.RandomZoom(0.08),
        layers.Conv2D(24, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(48, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(96, activation='relu'), layers.Dropout(0.35),
        layers.Dense(len(folders), activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(xtr, ytr, validation_data=(xv,yv), epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)])
    model.save(output / 'traffic_sign_cnn.keras')
    loss, accuracy = model.evaluate(xt, yt, verbose=0)
    preds = model.predict(xt, verbose=0).argmax(axis=1)
    ids = list(range(len(folders)))
    report = classification_report(yt, preds, labels=ids, target_names=[f"{v['class_id']}: {v['name']}" for v in labels], zero_division=0)
    (output / 'classification_report.txt').write_text(report, encoding='utf-8')
    metrics = {'test_accuracy': float(accuracy), 'test_loss': float(loss), 'train_images': len(ytr), 'validation_images': len(yv), 'test_images': len(yt), 'classes': len(folders), 'parameters': model.count_params()}
    (output / 'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(history.history).to_csv(output / 'training_history.csv',index=False)
    cm = confusion_matrix(yt,preds,labels=ids)
    pd.DataFrame(cm,index=[f.name for f in folders],columns=[f.name for f in folders]).to_csv(output / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set(xlabel='Predicted class index',ylabel='True class index', title='Held-out test confusion matrix')
    fig.tight_layout(); fig.savefig(output / 'confusion_matrix.png'); plt.close(fig)
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    for ax, metric in zip(axs,['accuracy','loss']):
        ax.plot(history.history[metric],label='Train'); ax.plot(history.history['val_'+metric],label='Validation'); ax.set_title(metric); ax.legend()
    fig.tight_layout(); fig.savefig(output / 'training_curves.png'); plt.close(fig)
    print(report)
    print(json.dumps(metrics,indent=2))
    print('Saved model and evaluation results in', output)

if __name__ == '__main__':
    main()
