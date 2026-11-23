import argparse
import json
import numpy as np
from PIL import Image, ImageOps
from common import ROOT, SIZE

def load_artifacts():
    import tensorflow as tf
    path = ROOT / 'models' / 'traffic_sign_cnn.keras'
    if not path.exists():
        raise FileNotFoundError('Model not found. Run python train.py first.')
    model = tf.keras.models.load_model(path)
    labels = json.loads((ROOT / 'models' / 'labels.json').read_text(encoding='utf-8'))
    return model, labels

def predict(img, model, labels):
    img = ImageOps.exif_transpose(img).convert('RGB').resize((SIZE,SIZE))
    x = np.asarray(img,dtype=np.uint8)[None]
    scores = model.predict(x,verbose=0)[0]
    return [{**labels[int(i)], 'score': float(scores[i])} for i in np.argsort(scores)[::-1][:3]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    model, labels = load_artifacts()
    with Image.open(args.image) as img:
        print(json.dumps(predict(img,model,labels),indent=2))
