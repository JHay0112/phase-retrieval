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



BETA = 0.5 # Difference Map Parameter, interpolates between constraints
PADDING = 2 # Zero padding on guess, ensures that solution converges properly

IMG_PATH = "img/"



def fourier_modulus(image: np.array) -> np.array:
    '''
        Computes the Fourier Modulus of a given image (2D array).

        Parameters
        ----------

        image: np.array
            The image to compute the fourier modulus of.

        Returns
        -------

        np.array
            Fourier Modulus of the passed image.
    '''
    return np.abs(fft2(image))



if __name__ == "__main__":

    true_img = Image.open(IMG_PATH)
    true_img = ImageOps.grayscale(true_img)

