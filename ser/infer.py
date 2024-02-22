import torch
import json
import glob
import os
from pathlib import Path
import pprint

from ser.constants import RESULTS_DIR

def run_infer(name, label, dataloader):
    
    # select image to run inference for
    images, labels = next(iter(dataloader))

    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model_path = _get_path(name, 'model')
    model = torch.load(model_path)

    _print_summary(name)

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))
    print(f'Predicted label: {pred}')
    print(f'Confidence: {confidence}')

    pixels = images[0][0]
    print(_generate_ascii_art(pixels))


def _generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(_pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def _pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "


def _get_path(name, file):

    run_dir = RESULTS_DIR / name 

    if file == 'model':
        return list(Path(run_dir).glob('*.pth'))[0]
    elif file == 'params':
        return list(Path(run_dir).glob('*.json'))[0]
    else:
        print('Incorrect file specified.')
        return None


def _print_summary(name):

    params_path = _get_path(name, 'params')

    content = params_path.read_text()
    params = json.loads(content)
    
    print(f'Running inference using model from {name}')
    print('Hyperparams used:')
    pprint.pprint(params)
    