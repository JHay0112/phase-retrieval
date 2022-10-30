'''
    Prototype difference map object.
'''

import numpy as np
from typing import Tuple, Callable

Array = np.ndarray


class DifferenceMap:
    '''
        General purpose helper for applying the difference map to n-dimensional arrays.

        Parameters
        ----------

        proj_A: Callable[[Array], Array]
            A minimal constraint projection for constraint A.
        proj_B: Callable[[Array], Array]
            A minimal constraint project for constraint B.
    '''
    def __init__(self, proj_A: Callable[[Array], Array], proj_B: Callable[[Array], Array]) -> None:

        self.iterand = None
        self.iterations = 0
        self.proj_A = proj_A
        self.proj_B = proj_B

    def __call__(self, iterand: Array, beta: float) -> Tuple[Array, float]:
        '''
            Generator style difference-map

            Parameters
            ----------
            
            iterand: Array
                The item that iteration will be performed upon.
            beta: float
                Difference map beta value.

            Example
            -------

            ```
            dmap = DifferenceMap(lambda i: fourier_projection(i, modulus), lambda i: support_projection(i, support))

            for image, error in dmap(image, BETA):

                errors.append(error)
                
                if dmap.iterations > MAX_ITERATIONS:
                    break
                if error <= TARGET_ERROR:
                    break
            ```
        '''

        # Reset iterations
        self.iterand = iterand
        self.iterations = 0

        # Ideal gamma values identified in [2]
        y_A = 1/beta
        y_B = -1/beta

        while True: # Step infinitely

            # Perform difference map operation
            p_A = (1 + y_A)*self.proj_A(self.iterand) - y_A*self.iterand
            p_B = (1 + y_B)*self.proj_B(self.iterand) - y_B*self.iterand
            p_diff = self.proj_B(p_A) - self.proj_A(p_B)
            error = np.linalg.norm(p_diff)
            self.iterand = self.iterand + beta*(p_diff)

            self.iterations += 1

            yield self.iterand, error