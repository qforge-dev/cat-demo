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


def network(width_multiplier: int) -> nn.Sequential:
    layers = []
    channels = 3
    for width in (16, 32, 64, 128):
        width *= width_multiplier
        layers.extend((nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)))
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected_sha = args.weights_sha256
    assert hashlib.sha256(args.weights.read_bytes()).hexdigest() == expected_sha
    saved = torch.load(args.weights, map_location='cpu', weights_only=True)
    assert saved['model'] == 'cat-cnn-v1' and saved['image_size'] == 128
    assert saved['labels'] == {'cat': 1, 'not_cat': 0} and saved['epochs'] == 15
    torch.set_num_threads(1)
    width_multiplier = saved.get('width_multiplier', 1)
    assert width_multiplier in (1, 2)
    model = network(width_multiplier).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    report = json.loads(args.report.read_text())
    assert sum(parameter.numel() for parameter in model.parameters()) == report['model_parameters']
    assert width_multiplier == report.get('width_multiplier', 1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'cat.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, 128, 128), target, input_names=['image'], output_names=['logit'], opset_version=18, dynamo=False, external_data=False, dynamic_axes={'image': {0: 'batch'}, 'logit': {0: 'batch'}})
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    predictions = {row['path']: row for row in json.loads(args.predictions.read_text())}
    correct, max_difference, max_saved_difference = 0, 0.0, 0.0
    with zipfile.ZipFile(args.dataset) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith('test/') and name.endswith('.png'))
        assert len(names) == report['test_samples'] == len(predictions)
        for start in range(0, len(names), 64):
            batch_names = names[start:start + 64]
            inputs = []
            for name in batch_names:
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    rgb = ImageOps.exif_transpose(image).convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
                    inputs.append(np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1)
            inputs = np.stack(inputs)
            with torch.no_grad():
                reference = model(torch.from_numpy(inputs)).flatten().numpy()
            actual = session.run(['logit'], {'image': inputs})[0].flatten()
            assert np.isfinite(actual).all()
            assert np.array_equal(actual >= 0, reference >= 0)
            assert np.array_equal(actual >= 0, [predictions[name]['prediction'] for name in batch_names])
            max_difference = max(max_difference, float(np.max(np.abs(actual-reference))))
            max_saved_difference = max(max_saved_difference, float(np.max(np.abs(actual-np.array([predictions[name]['logit'] for name in batch_names])))))
            correct += int(((actual >= 0) == [predictions[name]['target'] for name in batch_names]).sum())
    assert max_difference < 0.0001, max_difference
    assert max_saved_difference < 0.005, max_saved_difference
    assert correct == report['confusion']['true_cat'] + report['confusion']['true_other']
    evidence = dict(source_run=args.source_run,source_weights_sha256=expected_sha,onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),samples=len(names),matching_cpu_decisions=len(names),matching_saved_gpu_decisions=len(names),correct=correct,max_onnx_cpu_logit_difference=max_difference,max_saved_gpu_logit_difference=max_saved_difference,trained_parameters=report['model_parameters'],width_multiplier=width_multiplier,epochs=report['epochs'],training_samples=report['training_samples'],normalization='RGB, Pillow bilinear resize to 128x128, CHW, pixel/127.5-1',metrics={key:report[key] for key in ['accuracy','balanced_accuracy','cat_precision','cat_recall','nll','brier']},per_source_metrics=report['per_source_metrics'],dataset=report['dataset']['dataset'],dataset_revision=report['dataset_revision'],sources=report['dataset']['sources'],attribution='AFHQ: NAVER Corporation; Choi et al., StarGAN v2 (2020). Cats and Dogs: Microsoft. VOC: PASCAL VOC 2007 contributors. Synthetic backgrounds: Labqoat.',dataset_license='; '.join(dict.fromkeys(source['license'] for source in report['dataset']['sources'])))
    (output/'model.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
