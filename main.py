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

from more_itertools import padded
import numpy as np
from numpy.fft import fftn, ifftn

import matplotlib.pyplot as plot

from PIL import Image, ImageOps



B = 0.9 # Difference Map Parameter, interpolates between constraints [1]
IMG_PATH = "img/hill.jpg"
NOISE = 100


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
    img = np.array(img)
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
            The image to perform the minimal modification on.
        target_modulus: np.ndarray
            The expected fourier modulus. These are the magnitudes that the
            pixels are scaled to match.

        Returns
        -------

        np.ndarray
            The image with minimal modification, 
            passing it to fourier_modulus should match the target modulus.
    '''
    fimage = fftn(image)
    fimage_modulus = np.abs(fimage)
    fimage /= fimage_modulus
    fimage *= target_modulus
    return np.log(np.abs(ifftn(fimage)))

def support_projection(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    '''
        Restricts the image to the domain of its support.
        Also restricts image to have positive values.

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
    return np.abs(image * support)

def difference_map(image: np.ndarray, modulus: np.ndarray, support: np.ndarray) -> np.ndarray:
    '''
        Executes the difference map described in [1] and [2] upon the image once.

        Parameters
        ----------

        image: np.ndarray
            The image to iterate upon.
        modulus: np.ndarray
            The fourier modulus the image should be coerced to match.
        support: np.ndarray
            The support of the image.

        Returns
        -------

        np.ndarray
            The image transformed with one iteration of the difference map.
    '''
    p_F = fourier_projection(image, modulus)
    p_S = support_projection(image, support)
    y_F = 1/B
    y_S = -1/B
    f_F = (1 + y_F) * p_F - y_F
    f_S = (1 + y_S) * p_S - y_S
    return image + B*(p_S * f_F - p_F * f_S)



if __name__ == "__main__":

    # Get the expected result and convert it to a fourier modulus
    # This loses ALL PHASE INFORMATION
    # So we can't convert the modulus back to the image without sneakiness
    # This also pads the image as discussed in [2]
    true_image = image_as_array(IMG_PATH)
    padded_image = pad(true_image)
    modulus = fourier_modulus(padded_image)

    image = padded_image #+ NOISE*np.random.normal(0, 1, modulus.shape)
    support = np.pad(np.ones(true_image.shape), ((0, padded_image.shape[0] - true_image.shape[0]), (0, padded_image.shape[1] - true_image.shape[1])), 'constant')

    for _ in range(10):
        image = difference_map(image, modulus, support)

    f, axarr = plot.subplots(1, 2)
    axarr[0].imshow(np.log10(image))
    axarr[1].imshow(np.log10(modulus))
    plot.show()