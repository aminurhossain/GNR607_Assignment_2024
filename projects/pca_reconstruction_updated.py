import matplotlib.pyplot as plt
import rasterio

from concurrent.futures import ThreadPoolExecutor

def compute_band_covariance(flattened_bands, i, j, N, normalization_factor):
    b1 = flattened_bands[i]
    b2 = flattened_bands[j]
    cov = sum(b1[k] * b2[k] for k in range(N))
    return i, j, cov * normalization_factor

def compute_covariance_matrix(image):
    num_bands = len(image)
    height = len(image[0])
    width = len(image[0][0])
    N = height * width
    normalization_factor = 1 / (N - 1)

    flattened_bands = [
        [pixel for row in band for pixel in row] for band in image
    ]

    covariance_matrix = [[0 for _ in range(num_bands)] for _ in range(num_bands)]

    # Use a ThreadPoolExecutor to compute band pairs in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_bands):
            for j in range(i, num_bands):
                futures.append(
                    executor.submit(
                        compute_band_covariance, flattened_bands, i, j, N, normalization_factor
                    )
                )

        for future in futures:
            i, j, value = future.result()
            covariance_matrix[i][j] = value
            if i != j:
                covariance_matrix[j][i] = value

    return covariance_matrix
def perform_pca_reconstruction(bands, num_components):
    normalize_bands, mean, std = normalize_data(bands)
    eigenvectors, principal_components, shape, eigenvalues = perform_pca(normalize_bands)
    reconstructed_image, selected_components = inverse_pca_and_reconstruct(eigenvectors, principal_components, shape,
                                                                           num_components, mean,
                                                                           std)
    return reconstructed_image, eigenvalues, selected_components


def calculate_variation_percentage(eigenvalues, components):
    total_variance = np.sum(eigenvalues)

    variance_captured = np.sum(eigenvalues[:components])

    percentage_variation = (variance_captured / total_variance) * 100

    return percentage_variation


def read_multiband_image(file_path):
    with rasterio.open(file_path) as src:
        bands = src.read()

        meta = src.meta

    return bands, meta


def write_image(file_path, image, meta):
    with rasterio.open(file_path, 'w', **meta) as dest:
        dest.write(image)


# Get the shape of the input image (num_bands, height, width)

def normalize_data(bands):
    num_bands = len(bands)

    height = len(bands[0])

    width = len(bands[0][0])

    # Initialize arrays for the normalized image, mean, and std

    normalized_image = np.empty_like(bands, dtype=np.float64)

    mean = np.zeros((num_bands, 1), dtype=np.float64)

    std = np.zeros((num_bands, 1), dtype=np.float64)

    # Flatten the image bands and calculate mean and std manually

    for band_index in range(num_bands):

        band = bands[band_index]

        # Calculate mean manually

        sum_band = 0.0

        for i in range(height):

            for j in range(width):
                sum_band += band[i][j]

        mean[band_index] = sum_band / (height * width)

        # Calculate standard deviation manually

        sum_squared_diff = 0.0

        for i in range(height):

            for j in range(width):
                sum_squared_diff += (band[i][j] - mean[band_index]) ** 2

        std[band_index] = (sum_squared_diff / (height * width)) ** 0.5

        # Normalize the band

        normalized_image[band_index] = ((band - mean[band_index]) / std[band_index])

    return normalized_image, mean, std


def normalize_data_build(bands):
    # Reshape bands to (num_bands, height * width)
    num_bands, height, width = bands.shape
    reshaped_bands = bands.reshape(num_bands, height * width)

    # Standardize each band
    mean = reshaped_bands.mean(axis=1, keepdims=True)
    std = reshaped_bands.std(axis=1, keepdims=True)
    normalized_bands = (reshaped_bands - mean) / std

    return normalized_bands.reshape(num_bands, height, width), mean, std


def compute_covariance_matrix_SLOW(image):
    """

    Compute the covariance matrix of an image without using np.cov or np.dot.
    Parameters:

    image (numpy.ndarray): A 3D array of shape (num_bands, height, width)

    Returns:

    numpy.ndarray: The covariance matrix of the image bands

    """

    num_bands = len(image)

    height = len(image[0])

    width = len(image[0][0])

    N = height * width

    # Initialize the covariance matrix

    covariance_matrix = np.zeros((num_bands, num_bands))

    # Compute the covariance matrix manually

    for i in range(num_bands):

        for j in range(num_bands):

            b1 = image[i]

            b2 = image[j]

            cov = 0

            for k in range(height):

                for l in range(width):
                    cov += b1[k][l] * b2[k][l]

            covariance_matrix[i, j] = cov / (N - 1)

    return covariance_matrix


def sort_indices_desc(eigenvalues):
    n = len(eigenvalues)

    # Initialize indices array

    indices = [i for i in range(n)]
    # Implement selection sort to sort indices based on eigenvalues in descending order

    for i in range(n):

        # Find the index of the maximum element in the remaining unsorted portion

        max_idx = i

        for j in range(i + 1, n):

            if eigenvalues[indices[j]] > eigenvalues[indices[max_idx]]:
                max_idx = j
        # Swap the found maximum element with the first element of the unsorted portion

        indices[i], indices[max_idx] = indices[max_idx], indices[i]

    # Convert the sorted indices list to a numpy array

    # sorted_indices = np.array(indices)
    return indices


def perform_pca(bands):
    # Compute covariance matrix

    covariance_matrix = compute_covariance_matrix(bands)

    # Compute eigenvalues and eigenvectors

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order

    idx = sort_indices_desc(eigenvalues)

    eigenvalues = eigenvalues[idx]

    eigenvectors = eigenvectors[:, idx]

    num_bands = len(bands)

    height = len(bands[0])

    width = len(bands[0][0])

    # Project the data onto the eigenvectors

    principal_components = np.zeros_like(bands)

    for i in range(num_bands):

        ev = eigenvectors[:, i]

        pc = np.zeros_like(bands[0])

        for j in range(num_bands):
            pc = pc + bands[j] * ev[j]

        principal_components[i] = pc

    # Example matrix principal_components (7x10x15)

    n_bands = len(principal_components)

    height = len(principal_components[0])

    width = len(principal_components[0][0])

    # Initialize the new matrix with the desired shape (150x7)

    reshaped_principal_components = np.zeros((height * width, n_bands))

    # Manually reshape the matrix

    for i in range(n_bands):  # Loop through the first dimension (7)

        for j in range(height):  # Loop through the second dimension (10)

            for k in range(width):  # Loop through the third dimension (15)

                # Calculate the new row index for the reshaped matrix

                new_index = j * width + k

                reshaped_principal_components[new_index, i] = principal_components[i, j, k]

    return eigenvectors, reshaped_principal_components, (num_bands, height, width), eigenvalues


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


import numpy as np

from skimage.metrics import structural_similarity as ssim


# Function to calculate PSNR

def calculate_psnr(original, compressed):
    # print((original-compressed).max())

    mse = np.mean((original - compressed) ** 2)

    if mse < 10 ** -8:  # MSE is zero means no noise is present in the signal, PSNR is infinite

        return float('inf')

    max_pixel = 65536.0

    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr_value


# Function to calculate SSIM

def calculate_ssim(original, compressed):
    ssim_value = ssim(original, compressed, multichannel=True, data_range=65536)

    return ssim_value


def main(image_file_path, num_components=2):
    reconstructed_list = []

    bands, meta = read_multiband_image(image_file_path)

    normalize_bands, mean, std = normalize_data(bands)

    eigenvectors, principal_components, shape, eigenvalues = perform_pca(normalize_bands)
    reconstructed_image = inverse_pca_and_reconstruct(eigenvectors, principal_components, shape, num_components, mean,
                                                      std)
    output_path = f'reconstructed_with_{num_components}_components.dat'
    meta.update(dtype=rasterio.float32, count=reconstructed_image.shape[0])
    write_image(output_path, reconstructed_image, meta)
    rgb_image = reconstructed_image[[3, 2, 1], :, :]  # Bands 4, 3, 2 are at indices 3, 2, 1
    plt.imshow(np.transpose(rgb_image / np.max(rgb_image), (1, 2, 0)))  # for visualization only

    plt.title(f'Reconstructed with {num_components} components')
    plt.show()

    return reconstructed_list


def main_all(image_file_path):
    reconstructed_list = []

    bands, meta = read_multiband_image(image_file_path)

    normalize_bands, mean, std = normalize_data(bands)

    eigenvectors, principal_components, shape, eigenvalues = perform_pca(normalize_bands)
    for num_components in range(2, bands.shape[0]):
        reconstructed_image = inverse_pca_and_reconstruct(eigenvectors, principal_components, shape, num_components,
                                                          mean, std)

        output_path = f'reconstructed_with_{num_components}_components.dat'

        reconstructed_list.append(reconstructed_image)

        meta.update(dtype=rasterio.float32, count=reconstructed_image.shape[0])
        write_image(output_path, reconstructed_image, meta)

        print(f'Reconstructed image with {num_components} components saved to {output_path}')

        # Display the reconstructed image using RGB bands (4, 3, 2)

        rgb_image = reconstructed_image[[3, 2, 1], :, :]  # Bands 4, 3, 2 are at indices 3, 2, 1

        # rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # print(rgb_image.max())

        plt.imshow(np.transpose(rgb_image / np.max(rgb_image), (1, 2, 0)))  # for visualization only

        plt.title(f'Reconstructed with {num_components} components')

        plt.show()

    return reconstructed_list


if __name__ == "__main__":
    file = "C:\\Users\\aminur\\Desktop\\PCA_Projects\\input\\landsat_8_mumbai.dat"
    bands, meta = read_multiband_image(file)
    perform_pca_reconstruction(bands, 3)
    # , perform_pca_reconstruction)
