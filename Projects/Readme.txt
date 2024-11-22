There are two .py file for this programming projects.

pca_reconstruction.py 
For pca calculation and inverse image reconstruction
This .py have all code with out using any inbuild function like numpy mean and cov ..


pca_gui.py
For GUI display of image and recontruction with input and output file selection 


One extra .py file
pca_slow.py
which will have same code as pca_reconstruction.py but it is using numpy inbuild function for mwan or variance and covariance calculatoor



To run code use this command

python pca_gui.py




It will open GUI for PCA viewer 

"1. **Select Input Image**: Choose the input image for PCA processing.
"2. **Select Output File**: Specify a file to save the reconstructed PCA image with extention .dat, .tif.
"3. **Number of Components**: Enter the number of principal components to use.
"4. **Regenerate**: Process the input image with the specified PCA components.
"5. **Statistics**: View PSNR, SSIM, and explained variance for the reconstruction.


"Tips
Use the mouse to pan and zoom the images.
Ensure the input image is in a supported format (e.g., PNG, JPG, TIFF).
