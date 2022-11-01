'''
    Phase retriever.
'''

import numpy as np
from numpy.fft import fftn, ifftn

from dm import DifferenceMap, Array

from typing import Callable


class PhaseRetriever(DifferenceMap):
    '''
        Performs phase retrieval with the difference map algorithm.

        Parameters
        ----------

        target_modulus
            The Fourier modulus to be targeted by the iterative process.
            The produced iterand will have the same or near the same modulus.
        support_projection
            The support projection to be used in the iterative process.
            Externally supplied as implementations may vary (e.g. sparsity or zero-padded).
    '''
    def __init__(self, target_modulus: Array, support_projection: Callable[[Array], Array]) -> None:

        self.target_modulus = target_modulus
        self.support_projection = support_projection

        super().__init__(self.fourier_projection, self.support_projection)

        self.iterand = self.support_projection(np.abs(ifftn(target_modulus)))

    def fourier_projection(self, iterand: Array) -> Array:
        '''
            Performs a minimal modification of the iterand to match the target Fourier modulus.
            Note that this does NOT modify the iterand in place, but returns the minimal modification.
            This is primarily intended for internal use by the PhaseRetriever class, but may have interesting uses if exposed.

            Parameters
            ----------

            iterand
                Image to perform the minimal modification on.

            Returns
            -------

            Array
                The image with minimal modification, 
                passing it to fourier_modulus should match the target modulus.
        '''
        fimage = fftn(iterand)
        fimage_modulus = np.abs(fimage)
        fimage = (fimage/fimage_modulus) * self.target_modulus
        return ifftn(fimage)

    def __call__(self, beta):
        return super().__call__(self.iterand, beta)