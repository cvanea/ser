from torchvision import transforms
import torchvision.transforms.functional as TF

'''
class SetChannels():
    def __init__(self, num_output_channels=1):
        assert num_output_channels in [1, 3], "Channels must be either 1 or 3"
        self.num_output_channels = num_output_channels

    def __call__(self, tensor):
        # if supplied in rgb but you want grayscale

        if self.num_output_channels == 1 and tensor.size(0) == 3:
            return TF.to_grayscale(tensor, num_output_channels=1)
        # if supplied in grayscale but you want rgb - repeat
        if self.num_output_channels == 3 and tensor.size(0) == 1:
            #tensor =  tensor.expand(3, -1, -1)
            return tensor.repeat(3, 1, 1)
            #print(tensor.size(0))
        else:
            return tensor
        # else: leave as is
'''

class MyTransforms:
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(), 
            #SetChannels(num_output_channels=3),
            transforms.Normalize((0.5,), (0.5,))
            ]
        )