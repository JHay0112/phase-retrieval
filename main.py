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

    discussed in 

    [3] J. Hay, "Phase Retreival", https://jordanhay.com/blog/2022/07/phase-retrieval, 2022.

    Thank you Joe for teaching me about this :D
'''

from difflib import Differ
import numpy as np
from numpy.fft import fftn, ifftn, fftshift

import matplotlib.pyplot as plot

from PIL import Image, ImageOps
from skimage import transform

from typing import Tuple

from dm import DifferenceMap


BETA = 1.15

IMG_PATH = "img/logo.png"
NOISE = 1
TARGET_ERROR = 0.5
MAX_ITERATIONS = 1000



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
    img = np.array(img).astype(float)
    img /= np.max(img)
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
    return np.abs(fftn(image))

def fourier_projection(image: np.ndarray, target_modulus: np.ndarray) -> np.ndarray:
    '''
        Performs a minimal modification of the passed image to match the expected Fourier modulus.
        
        Parameters
        ----------

        image: np.ndarray
            Image to perform the minimal modification on.
        target_modulus: np.ndarray
            The expected fourier modulus. These are the magnitudes that the
            pixels are scaled to match.

        Returns
        -------

        np.ndarray
            The image with minimal modification, 
            passing it to fourier_modulus should match the target modulus.
    '''
    fimage = fftn(image.copy())
    fimage_modulus = np.abs(fimage)
    fimage = (fimage/fimage_modulus) * target_modulus
    return ifftn(fimage)

def support_projection(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    '''
        Restricts the image to the domain of its support.

        Parameters
        ----------

        image: np.ndarray
            The image to constrict the support of.
        support: np.ndarray
            Boolean array describing areas of support.

        Returns
        -------

        np.ndarray
            The image with only supported areas.
    '''
    return image * support


if __name__ == "__main__":

    # Get and pad image
    true_image = image_as_array(IMG_PATH)
    true_image = transform.resize(true_image, (64, 64))
    true_image = true_image/np.max(true_image)
    padded_image = pad(true_image)

    modulus = fourier_modulus(padded_image)
    support = pad(np.ones(true_image.shape))

    # Make a noisy guess of the image
    # No point in making unsupported areas noisy though
    image = np.abs(NOISE*np.random.normal(0, 1, padded_image.shape))
    image = support_projection(image, support)
    guess = image.copy()

    errors = []
    error = float('inf')

    dmap = DifferenceMap(image, lambda i: fourier_projection(i, modulus), lambda i: support_projection(i, support))

    for image, error in dmap(BETA):

        errors.append(error)
        
        if dmap.iterations > MAX_ITERATIONS:
            break
        if error <= TARGET_ERROR:
            break

    image = fourier_projection(dmap.iterand, modulus)

    f, ax = plot.subplot_mosaic("ABXX;CDXX;EFXX")

    # Images
    ax["A"].imshow(padded_image, cmap='gray')
    ax["A"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["A"].set_title("Actual Data")
    ax["C"].imshow(guess, cmap='gray')
    ax["C"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["C"].set_title("Initial Guess")
    ax["E"].imshow(np.abs(image), cmap='gray')
    ax["E"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["E"].set_title("Retrieved Data")

    # Fourier Modulus of said images
    # fftshift is used here to centre the origin of the fourier transform for viewing purposes
    # log10 is used to reduce the 'direct current' --- high valued centre pixels --- also for view purposes
    ax["B"].imshow(np.log10(fftshift(modulus)), cmap='gray')
    ax["B"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["D"].imshow(np.log10(fftshift(fourier_modulus(guess))), cmap='gray')
    ax["D"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["F"].imshow(np.log10(fftshift(fourier_modulus(image))), cmap='gray')
    ax["F"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)

    # Error
    ax["X"].plot(range(dmap.iterations + 1), errors)
    ax["X"].set_title("Approximate Error")

    plot.show()