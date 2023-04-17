import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the dataset
data = scipy.io.loadmat('USPS.mat')
A = data['A']

# Perform mean-centering on the data
A_mean = np.mean(A, axis=0)
A_centered = A - A_mean

# Perform SVD on mean-centered data
U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)

def reconstruct_images(U, S, Vt, A_mean, num_components):
    return (U[:, :num_components] @ np.diag(S[:num_components]) @ Vt[:num_components, :]) + A_mean

# Reconstruct images using different numbers of principal components
A_reconstructed_p10 = reconstruct_images(U, S, Vt, A_mean, 10)
A_reconstructed_p50 = reconstruct_images(U, S, Vt, A_mean, 50)
A_reconstructed_p100 = reconstruct_images(U, S, Vt, A_mean, 100)
A_reconstructed_p200 = reconstruct_images(U, S, Vt, A_mean, 200)

# Visualize the original and reconstructed images (example for the second image)
A2 = A[1].reshape(16, 16).T
A2_reconstructed_p10 = A_reconstructed_p10[1].reshape(16, 16).T
A2_reconstructed_p50 = A_reconstructed_p50[1].reshape(16, 16).T
A2_reconstructed_p100 = A_reconstructed_p100[1].reshape(16, 16).T
A2_reconstructed_p200 = A_reconstructed_p200[1].reshape(16, 16).T

plt.figure(figsize=(10, 2))
plt.subplot(1, 5, 1)
plt.imshow(A2, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 5, 2)
plt.imshow(A2_reconstructed_p10, cmap='gray')
plt.title('10 Principal Components')
plt.axis('off')
plt.subplot(1, 5, 3)
plt.imshow(A2_reconstructed_p50, cmap='gray')
plt.title('50 Principal Components')
plt.axis('off')
plt.subplot(1, 5, 4)
plt.imshow(A2_reconstructed_p100, cmap='gray')
plt.title('100 Principal Components')
plt.axis('off')
plt.subplot(1, 5, 5)
plt.imshow(A2_reconstructed_p200, cmap='gray')
plt.title('200 Principal Components')
plt.axis('off')
plt.show()
