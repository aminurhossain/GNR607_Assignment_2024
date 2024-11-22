import numpy as np
import rasterio


def perform_pca_reconstruction_fast(bands, num_components):
    normalize_bands, mean, std = normalize_data(bands)
    eigenvectors, principal_components, shape, eigenvalues = perform_pca(normalize_bands)
    reconstructed_image, selected_components = inverse_pca_and_reconstruct(eigenvectors, principal_components, shape,
                                                                           num_components, mean,
                                                                           std)
    return reconstructed_image, eigenvalues, selected_components


def perform_pca(bands):
    num_bands, height, width = bands.shape
    reshaped_bands = bands.reshape(num_bands, height * width).T

    # Compute covariance matrix
    covariance_matrix = np.cov(reshaped_bands, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project the data onto the eigenvectors
    principal_components = np.dot(reshaped_bands, eigenvectors)

    return eigenvectors, principal_components, (num_bands, height, width), eigenvalues


def normalize_data(bands):
    # Reshape bands to (num_bands, height * width)
    num_bands, height, width = bands.shape
    reshaped_bands = bands.reshape(num_bands, height * width)

    # Standardize each band
    mean = reshaped_bands.mean(axis=1, keepdims=True)
    std = reshaped_bands.std(axis=1, keepdims=True)
    normalized_bands = (reshaped_bands - mean) / std

    return normalized_bands.reshape(num_bands, height, width), mean, std


def inverse_pca_and_reconstruct(eigenvectors, principal_components, shape, num_components, mean, std):
    # Select the top num_components eigenvectors

    selected_eigenvectors = eigenvectors[:, :num_components]

    # Select the corresponding principal components

    selected_components = principal_components[:, :num_components]

    # Reconstruct the image

    reconstructed = np.dot(selected_components, selected_eigenvectors.T)

    reconstructed = reconstructed.T

    # De-normalize the reconstructed data

    num_bands, height, width = shape

    reconstructed = reconstructed.reshape(num_bands, height * width)

    reconstructed = (reconstructed * std) + mean

    reconstructed = reconstructed.reshape(num_bands, height, width)
    selected_components_reshaped = selected_components.reshape(height, width, num_components)
    # print(selected_components_reshaped.shape)

    return reconstructed, selected_components_reshaped


def read_multiband_image(file_path):
    with rasterio.open(file_path) as src:
        bands = src.read()

        meta = src.meta

    return bands, meta


if __name__ == "__main__":
    file = "C:\\Users\\aminur\\Desktop\\PCA_Projects\\PCA_final\\input\\landsat_8_mumbai_small.dat"
    bands, meta = read_multiband_image(file)
    perform_pca_reconstruction_fast(bands, 3)
    # , perform_pca_reconstruction)
