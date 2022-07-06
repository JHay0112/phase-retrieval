'''
    Test Module for Phase Retrieval Code
'''

from main import *
import numpy as np



def test_pad():
    '''
        Tests the pad function correctly pads images.
        For example with a scale of 1 the size of the image should double.
        Image should also be preserved in the top right corner
    '''
    shape = (5, 10)
    image = np.ones(shape)
    image = pad(image)
    assert image.shape == (10, 20)
    assert image[0, 0] == 1
    assert image[9, 19] == 0