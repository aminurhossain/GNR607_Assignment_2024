Problem No 22
Objective:
Given a multiband image of N bands, compute principal components and generate the approximate version of the input image by performing inverse principal component transform using 2, 3, ..., N-1 components.

Team Members:
Aminur Hossain (ID: 24D1384)
Amartya Ray (ID: 24D1383)
Input Data:
Dataset: Landsat-8 satellite image
Bands: 7 bands (for the Mumbai Scene)
Steps to Solve the Problem:
Load the Multiband Image:

Import the Landsat-8 image with 7 bands for the Mumbai scene.
Compute Principal Components (PCA):

Perform Principal Component Analysis (PCA) on the 7-band image to reduce dimensionality.
Reconstruction Using 2, 3, ..., N-1 Principal Components:

Reconstruct the image using inverse PCA with 2, 3, 4, 5, and 6 components to generate approximate images.
Comparison:

Compare the quality of reconstructed images using various quality metrics such as PSNR and SSIM to evaluate how much detail is retained with fewer components.
Visualization:

Display the original image alongside the reconstructed versions with different numbers of components for a visual comparison.
Expected Results:
Images reconstructed using varying numbers of components will show a trade-off between reduced complexity (lower number of components) and image quality (higher number of components).
Evaluate which number of components gives the best balance between compression and quality retention.
