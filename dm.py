'''
    Difference map context manager.

    This is a more general purpose abstraction,
    it permits many forms of the difference map algorithm.
    With a supplied fourier and support projection it may be used to perform phase retrieval.
'''

import numpy as np
from typing import Tuple, Callable

Array = np.ndarray


class DifferenceMap:
    '''
        General purpose helper for applying the difference map to n-dimensional arrays.

        Parameters
        ----------

        proj_A
            A minimal constraint projection for constraint A.
        proj_B
            A minimal constraint project for constraint B.

        Example
        -------

        ```
        # Assuming errors, fourier_projection, modulus, support_projection, and support have been defined
        # This example comes from phase retrieval

        dmap = DifferenceMap(lambda i: fourier_projection(i, modulus), lambda i: support_projection(i, support))
        self.iteration_limit = 1000  # Set max number of iterations to perform
        self.target_error = 0.5      # Set a threshold for error where iteration will stop early

        for image, error in dmap(image, BETA):
            # any additional operations
            errors.append(error)
        ```
    '''
    def __init__(self, proj_A: Callable[[Array], Array], proj_B: Callable[[Array], Array]) -> None:

        
        self.iterand = None
        '''Most recently produced iteration.'''
        self.iterations = 0
        '''Number of performed iterations.'''
        self.iteration_limit = None
        '''Maximum number of iterations to be performed.'''

        self.error = float("inf")
        '''Approximate error of most recently produced iteration.'''
        self.target_error = None
        '''Target for error, iteration halts when error falls below this threshold.'''

        self.change = float("inf")
        '''Approximate change in error between most recent two iterations.'''
        self.target_change = None
        '''Target for change in error, iteration halts when error chance falls below this threshold.'''

        self.proj_A = proj_A
        self.proj_B = proj_B

    def __may_step(self) -> bool:
        '''
            Determines if the iteration should continue.
        '''

        if self.iteration_limit is not None:
            if self.iterations >= self.iteration_limit:
                return False # Reached iteration limit
        
        if self.target_error is not None:
            if self.error <= self.target_error:
                return False # Reached target error

        if self.target_change is not None:
            if self.change <= self.target_change:
                return False # Reached target change in error

        return True


    def __call__(self, iterand: Array, beta: float) -> Tuple[Array, float]:
        '''
            Generator style difference-map

            Parameters
            ----------
            
            iterand
                The item that iteration will be performed upon.
            beta
                Difference map beta value.
        '''

        # Reset iterations
        self.iterand = iterand
        self.iterations = 0

        # Ideal gamma values identified in [2]
        y_A = 1/beta
        y_B = -1/beta

        while self.__may_step():

            # Perform difference map operation
            p_A = (1 + y_A)*self.proj_A(self.iterand) - y_A*self.iterand
            p_B = (1 + y_B)*self.proj_B(self.iterand) - y_B*self.iterand
            p_diff = self.proj_B(p_A) - self.proj_A(p_B)
            new_error = np.linalg.norm(p_diff)

            self.change = np.abs(self.error - new_error)
            self.error = new_error

            self.iterand = self.iterand + beta*(p_diff)

            self.iterations += 1

            yield self.iterand, self.error