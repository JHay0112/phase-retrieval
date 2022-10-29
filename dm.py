'''
    Prototype difference map object.
'''

import numpy as np
from typing import Tuple, Callable, Any

Array = np.ndarray


class DifferenceMap:
    '''
        General purpose helper for applying the difference map to n-dimensional arrays.

        Parameters
        ----------

        iterand: Array
            The item that iteration will be performed upon.
        proj_A: Callable[[Array], Array]
            A minimal constraint projection for constraint A.
        proj_B: Callable[[Array], Array]
            A minimal constraint project for constraint B.
    '''
    def __init__(self, iterand: Array, proj_A: Callable[[Array], Array], proj_B: Callable[[Array], Array]):

        self.iterand = iterand
        self.iterations = 0
        self.proj_A = proj_A
        self.proj_B = proj_B

    def __call__(self, beta: float) -> Tuple[Array, float]:
        '''
            Generator style difference-map

            Parameters
            ----------
            
            beta: float
                Difference map beta value.

            Example
            -------

            ```

            ```
        '''

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

            # Yield current iterand and approximate error
            yield self.iterand, error
            self.iterations += 1