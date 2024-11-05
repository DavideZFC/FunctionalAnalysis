import numpy as np

def inf(matrix):
    """
    Computes the infinity norm of a matrix.

    Parameters:
    matrix (array-like): A 2D list or NumPy array representing the matrix.

    Returns:
    float: The infinity norm of the matrix.
    """
    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    
    # Compute the infinity norm
    rowsinf = np.sum(np.abs(matrix), axis=1)
    
    infinity_norm = np.max(rowsinf) # np.sort(np.sum(np.abs(matrix), axis=1))[int(k-1)] #
    idx = np.argmax(rowsinf)
    return infinity_norm, idx