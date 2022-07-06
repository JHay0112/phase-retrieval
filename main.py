'''
    Phase Retrieval from Fourier Modulus of Image

    This is used to determine what an object looks like from the measurement of
    diffraction of a planar wave around the object. This shows up in experiments
    where X-rays are diffracted around crystals of interest and the diffracted rays
    measured with CCDs. The measured diffraction corresponds to the Fourier Modulus
    of the object.

    Based on the following

    [1] V. Elser, I. Rankenburg and P. Thibault, "Searching with Iterated Maps," 
        Proceedings of the National Academy of Sciences - PNAS, vol. 104, (2), pp. 418-423, 2007.

    [2] V. Elser, "The Mermin Fixed Point," Foundations of Physics, vol. 33, (11), pp. 1691, 2003.

    Thank you Joe for teaching me about this :D
'''

import numpy as np
from numpy.fft import fft2

from PIL import Image, ImageOps



BETA = 0.5 # Difference Map Parameter, interpolates between constraints [1]
PADDING = 2 # Zero padding on guess, ensures that solution converges properly

IMG_PATH = "img/"


def image_as_array(path: str) -> np.ndarray:
    '''
        Opens an image as a numpy array.
        Converts coloured images to grayscale.

        Parameters
        ----------

        path: str
            The path to the image to open.

        Returns
        -------

        np.ndarray
            2D array representing grayscale image.
    '''
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    return img

def pad(image: np.ndarray, scale: int = 1) -> np.ndarray:
    '''
        Zero-pads the image with the selected ratio of padding.
        Zeros will extend out from the image, with the image pixel indices preserved.
        This only pads in the first two dimensions.
    
        Parameters
        ----------

        image: np.ndarray
            The image to pad.
        scale: int = 1
            The amount to pad the image by in each dimension scaled by the size of the image.
            The default of one will double the size of the image in each dimension.

        Returns
        -------

        np.ndarray
            The image with additional zero padding.
    '''
    return np.pad(image, ((0, scale * image.shape[0]), (0, scale * image.shape[1])), 'constant')

def fourier_modulus(image: np.ndarray) -> np.ndarray:
    '''
        Computes the Fourier Modulus of a given image (2D array).

        Parameters
        ----------

        image: np.ndarray
            The image to compute the fourier modulus of.

        Returns
        -------

        np.ndarray
            Fourier Modulus of the passed image.
    '''
    return np.abs(fft2(image))



if __name__ == "__main__":

    # Get the expected result and convert it to a fourier modulus
    # This loses ALL PHASE INFORMATION
    # So we can't convert the modulus back to the image without sneakiness
    # This also pads the image as discussed in [2]
    true_image = pad(image_as_array(IMG_PATH), scale = 1)
    modulus = fourier_modulus(true_image)
