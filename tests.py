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

def test_fourier_projection():
    '''
        Tests that a minimal modification on an image is correctly performed.
        In practice this is tested by ensuring the returned image's fourier transform
        matches the modulus passed to the function.
    '''
    image = np.ones((100, 100))
    modulus = np.zeros((100, 100))
    transformed_image = fourier_projection(image, modulus)
    assert np.all(np.abs(fourier_modulus(transformed_image) - modulus) < 1e-16)
    assert not np.array_equal(image, transformed_image)