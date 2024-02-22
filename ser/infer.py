import torch

def run_infer(model_path, label, dataloader):
    
    # select image to run inference for
    images, labels = next(iter(dataloader))

    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(model_path)

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(_generate_ascii_art(pixels))
    print(f"This is a {pred}")


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
