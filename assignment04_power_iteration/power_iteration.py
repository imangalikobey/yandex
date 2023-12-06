import numpy as np
def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
        # Initialize a random vector
    eigenvector = np.random.rand(data.shape[0])

    # Iteratively apply the power method
    for _ in range(num_steps):
        # Compute the matrix-vector product
        product = np.dot(data, eigenvector)
        
        # Compute the eigenvalue (Rayleigh quotient)
        eigenvalue = np.dot(eigenvector, product) / np.dot(eigenvector, eigenvector)
        
        # Normalize the eigenvector
        eigenvector = product / np.linalg.norm(product)
    eigenvalue=float(eigenvalue)
    return eigenvalue, eigenvector
