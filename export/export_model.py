"""Convert verified CNN weights and compare every heldout prediction without training."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torch import nn


def network() -> nn.Sequential:
    layers = []
    channels = 3
    for width in (16, 32, 64, 128):
        layers.extend((nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)))
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected_sha = '14b9421228a5e08e6f473eae962ab0f2423c8666e17dbf3f9c1db1d871555687'
    assert hashlib.sha256(args.weights.read_bytes()).hexdigest() == expected_sha
    saved = torch.load(args.weights, map_location='cpu', weights_only=True)
    assert saved['model'] == 'cat-cnn-v1' and saved['image_size'] == 128
    assert saved['labels'] == {'cat': 1, 'not_cat': 0} and saved['epochs'] == 15
    torch.set_num_threads(1)
    model = network().eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'cat.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, 128, 128), target, input_names=['image'], output_names=['logit'], opset_version=18, dynamo=False, external_data=False, dynamic_axes={'image': {0: 'batch'}, 'logit': {0: 'batch'}})
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    names, labels, inputs = [], [], []
    with zipfile.ZipFile(args.dataset) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith('test/') and name.endswith('.png'):
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    rgb = ImageOps.exif_transpose(image).convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
                    inputs.append(np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1)
                names.append(name)
                labels.append(int(name.split('/')[1] == 'cat'))
        for domain in ['cat', 'dog']:
            name = next(name for name in names if name.startswith(f'test/{domain}/'))
            (output/f'{domain}.jpg').write_bytes(archive.read('previews/'+name.removesuffix('.png')+'.jpg'))
    inputs = np.stack(inputs)
    assert len(inputs) == 1467
    with torch.no_grad():
        reference = torch.cat([model(batch).flatten() for batch in torch.from_numpy(inputs).split(64)]).numpy()
    actual = np.concatenate([session.run(['logit'], {'image': inputs[i:i+64]})[0].flatten() for i in range(0,len(inputs),64)])
    assert np.isfinite(actual).all()
    assert np.array_equal(actual >= 0, reference >= 0)
    max_difference = float(np.max(np.abs(actual-reference)))
    assert max_difference < 0.0001, max_difference
    report = json.loads(args.report.read_text())
    assert int(((actual>=0) == np.array(labels)).sum()) == 1455
    evidence = dict(source_run='9dc855da-9c8f-4df1-8291-e5425caa8663',source_weights_sha256=expected_sha,onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),samples=1467,matching_cpu_decisions=1467,correct=1455,max_onnx_cpu_logit_difference=max_difference,trained_parameters=97809,epochs=15,training_samples=14336,normalization='RGB, Pillow bilinear resize to 128x128, CHW, pixel/127.5-1',metrics={key:report[key] for key in ['accuracy','balanced_accuracy','cat_precision','cat_recall','nll','brier']},dataset='AFHQ v2',attribution='NAVER Corporation; Choi et al., StarGAN v2 (2020)',dataset_license='CC BY-NC 4.0')
    (output/'model.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
