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
from numpy.fft import fftn, ifftn, fftshift, ifftshift

import matplotlib.pyplot as plot

from PIL import Image, ImageOps
from skimage import transform

from typing import Tuple


B = 0.5
Y_F = 1/B
Y_S = -1/B

IMG_PATH = "img/logo.png"
SIDE = 100



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
    image = image/np.abs(image)
    image *= support
    return image

def difference_map(image: np.ndarray, modulus: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, float]:
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
        float 
            A measure of the relative error 
    '''
    f_F = (1 + Y_F)*fourier_projection(image, modulus) - Y_F*image
    f_S = (1 + Y_S)*support_projection(image, support) - Y_S*image
    f_diff = support_projection(f_F, support) - fourier_projection(f_S, modulus)
    error = np.linalg.norm(f_diff)
    image = image + B*(f_diff)
    return (image, error)


if __name__ == "__main__":

    modulus = fftshift(np.abs(image_as_array(IMG_PATH)))
    modulus = transform.resize(modulus, (2*SIDE, 2*SIDE))
    support = pad(np.ones((SIDE, SIDE)))
    image = np.abs(ifftn(modulus))
    #image = np.abs(np.random.normal(0, 1, modulus.shape))
    guess = image.copy()
    errors = []

    for i in range(100):
        image, error = difference_map(image, modulus, support)
        errors.append(error)
    image = support_projection(image, support)

    f, ax = plot.subplot_mosaic("ABXX;CDXX")

    phase = np.angle(image)
    phase[np.abs(phase) < 1e-10] = 0

    output = np.angle(phase)
    output /= np.pi
    output *= 255
    output = output.astype(np.uint8)
    output = Image.fromarray(output[:SIDE, :SIDE])
    output.save('cropped.bmp')

    # Images
    ax["A"].imshow(guess, cmap='gray')
    ax["A"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["A"].set_title("Starting Guess")
    phase_plot = ax["C"].imshow(np.angle(phase[:SIDE, :SIDE]), cmap='gray')
    ax["C"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["C"].set_title("Final Estimate (Phase)")
    # Modulus
    ax["B"].imshow(fftshift(modulus), cmap='gray')
    ax["B"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["B"].set_title("Target Modulus")
    ax["D"].imshow(np.abs(fftshift(fftn(image))), cmap='gray')
    ax["D"].tick_params(bottom = False, labelbottom = False, left = False, labelleft = False)
    ax["D"].set_title("Final Modulus")

    f.colorbar(phase_plot, ax = ax["C"])

    # Error
    ax["X"].plot(range(i + 1), errors)
    ax["X"].set_title("Estimated Error")

    plot.show()