from ser.transforms import flip
import  numpy as np
from PIL import Image
from torch import Tensor, equal

def test_flipper():
    a = [[(1,1,1,255),(2,2,2,255)],
         [(1,2,1,255),(2,1,2,255)]]
    x = np.asarray(a, dtype=np.uint8)
    img = Image.fromarray(x)

    b = [[(2,1,2,255), (1,2,1,255)],
         [(2,2,2,255), (1,1,1,255)]]
    y = np.asarray(b, dtype=np.uint8)
    expect = Image.fromarray(y)

    assert flip()(img) == expect

def test_flipper2():
    img = Tensor([[1, 2], [3, 4]])
    expect = Tensor([[4, 3], [2, 1]])

    assert equal(flip()(img), expect) == 1