import numpy as np
import typing


class Solution(typing.NamedTuple):
    """
    Represents a solution found during optimization.

    Attributes:
        solution (np.ndarray): The solution vector.
        fitness (float): The fitness value of the solution.
    """
    solution: np.ndarray
    fitness: float