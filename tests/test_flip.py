from ser.transforms import flip
import  numpy as np
from PIL import Image

def test_flipper():
    a = [[(1,1,1,255),(2,2,2,255),(1,3,3,255)],
    [(1,2,1,255),(2,1,2,255),(3,3,3,255)],
    [(1,1,3,255),(2,2,3,255),(3,3,1,255)]]
    x = np.asarray(a, dtype=np.uint8)
    img = Image.fromarray(x)

    b = [[(3,3,1,255),(2,2,3,255),(1,1,3,255)],
         [(3,3,3,255), (2,1,2,255), (1,2,1,255)],
         [(1,3,3,255), (2,2,2,255), (1,1,1,255)]]
    y = np.asarray(b, dtype=np.uint8)
    expect = Image.fromarray(y)

    assert flip()(img) == expect