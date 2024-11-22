import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
from skimage import exposure
from sklearn.decomposition import PCA

from pca_reconstruction import perform_pca_reconstruction
# from pca_fast import perform_pca_reconstruction_fast as perform_pca_reconstruction
from pca_reconstruction import read_multiband_image, write_image, calculate_psnr, calculate_ssim, \
    calculate_variation_percentage


def pca_inbuild(img_array, n_components):
    H, W, C = img_array.shape
    image_2d = img_array.reshape(-1, C)

    # Perform PCA
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(image_2d)  # Project the image data to PCA space

    # Reconstruct the image using the first n_components
    image_reconstructed_2d = pca.inverse_transform(image_pca)
    reconstructed_image = image_reconstructed_2d.reshape(H, W, C)
    return reconstructed_image


def convert_16bit_to_8bit_minmax(bands):
    """Convert a 16-bit image to 8-bit using min-max scaling"""
    # Assuming bands is a 3D numpy array (bands, height, width)
    bands_8bit = np.zeros_like(bands, dtype=np.uint8)

    for i in range(bands.shape[0]):  # Loop over each band
        min_val = bands[i].min()
        max_val = bands[i].max()

        # Min-max scaling to 0-255
        bands_8bit[i] = ((bands[i] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return bands_8bit


def enhance_contrast(bands_8bit):
    """Enhance contrast of each band using contrast stretching"""
    enhanced_bands = np.zeros_like(bands_8bit, dtype=np.uint8)

    for i in range(bands_8bit.shape[0]):  # Loop over each band
        p2, p98 = np.percentile(bands_8bit[i], (2, 98))
        enhanced_bands[i] = exposure.rescale_intensity(bands_8bit[i], in_range=(p2, p98))

    return enhanced_bands


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCA Image Viewer")

        # Set a larger window size
        self.root.geometry("1600x900")

        # Initialize variables
        self.input_file = None
        self.output_file = None
        self.input_image = None
        self.input_image_org = None
        self.output_image = None
        self.input_image_tk = None
        self.output_image_tk = None
        self.input_image_synthesis = None
        self.meta = None
        self.eigenvalues = None
        self.pca_image = None  # To store the PCA image
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas_input_image = None
        self.canvas_output_image = None

        # Create a frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill=tk.X, pady=10)

        # Create input file button
        self.input_button = tk.Button(self.button_frame, text="Select Input Image", command=self.select_input_file)
        self.input_button.grid(row=0, column=0, padx=5)

        # Create output file button for selecting the output file name for saving PCs
        self.output_button = tk.Button(self.button_frame, text="Select Output File", command=self.select_output_file)
        self.output_button.grid(row=0, column=1, padx=5)

        # Create label and entry for n_components
        self.n_components_label = tk.Label(self.button_frame, text="Number of Components:")
        self.n_components_label.grid(row=0, column=2, padx=5)
        self.n_components_entry = tk.Entry(self.button_frame)
        self.n_components_entry.grid(row=0, column=3, padx=5)
        self.n_components_entry.insert(0, "3")  # Default value
        self.n_components_final = None
        self.PCs = None

        # Create a regenerate button
        self.regenerate_button = tk.Button(self.button_frame, text="Regenerate", command=self.compute_and_display_pca)
        self.regenerate_button.grid(row=0, column=4, padx=5)

        # Create frames for canvas
        self.frame = tk.Frame(root)
        self.frame.pack()



        # Create canvas for input image display
        self.canvas_input = tk.Canvas(self.frame, bg="grey", width=550, height=550)
        self.canvas_input.grid(row=0, column=0, padx=10, pady=10)

        # Create canvas for Reconstructed image display
        self.canvas_output = tk.Canvas(self.frame, bg="grey", width=550, height=550)
        self.canvas_output.grid(row=0, column=1, padx=10, pady=10)
        # Add captions to the bottom-middle of the canvases
        # Add captions below the canvases using the grid system
        self.input_caption = tk.Label(self.frame, text="Input Image", font=("Arial", 12))
        self.input_caption.grid(row=1, column=0, pady=(0, 10))

        self.output_caption = tk.Label(self.frame, text="Reconstructed Image", font=("Arial", 12))
        self.output_caption.grid(row=1, column=1, pady=(0, 10))

        # Add a Help button to the button frame
        self.help_button = tk.Button(self.button_frame, text="Help", command=self.show_help)
        self.help_button.grid(row=0, column=7, padx=5)


        # Create a frame for statistics
        self.stats_frame = tk.Frame(root)
        self.stats_frame.pack(pady=20, side=tk.TOP)

        # Table for statistics comparison
        # self.stats_table = tk.Label(self.stats_frame, text="Statistics: PSNR, SSIM, Explained Variance (%)", font=("Arial", 12))
        # self.stats_table.pack()
        self.stats_table = tk.Label(self.button_frame, text="PSNR: --, SSIM: --, Explained Variance (%): --",
                                    font=("Arial", 12))
        self.stats_table.grid(row=0, column=6, padx=5)
        # self.stats_table.pack()

        # Bind mouse events for synchronized panning and zooming
        self.canvas_input.bind("<ButtonPress-1>", self.start_pan)
        self.canvas_input.bind("<B1-Motion>", self.pan_image)
        self.canvas_input.bind("<MouseWheel>", self.zoom_image)

        self.canvas_output.bind("<ButtonPress-1>", self.start_pan)
        self.canvas_output.bind("<B1-Motion>", self.pan_image)
        self.canvas_output.bind("<MouseWheel>", self.zoom_image)

    def select_input_file(self):
        # Open file dialog to select input image file
        self.input_file = filedialog.askopenfilename(title="Select Input Image File",
                                                     filetypes=[("Image Files",
                                                                 "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.dat")])
        if self.input_file:
            self.load_input_image()

    def select_output_file(self):
        # Open file dialog to select output image file name for saving PCA
        self.output_file = filedialog.asksaveasfilename(title="Select Output Image File for PCA",
                                                        defaultextension=".tif",
                                                        filetypes=[("Image Files",
                                                                    "*.png;*.jpg;*.jpeg;*.bmp*;*.tif;*.tiff;*.dat")])
        if self.output_file:
            self.save_pca_image()  # Save PCA image once output file is selected



    # Define the Help function
    def show_help(self):
        help_text = (
            "PCA Image Viewer Help\n\n"
            "1. **Select Input Image**: Choose the input image for PCA processing.\n"
            "2. **Select Output File**: Specify a file to save the reconstructed PCA image with extention .dat, .tif.\n"
            "3. **Number of Components**: Enter the number of principal components to use.\n"
            "4. **Regenerate**: Process the input image with the specified PCA components.\n"
            "5. **Statistics**: View PSNR, SSIM, and explained variance for the reconstruction.\n\n"
            "Tips:\n"
            "- Use the mouse to pan and zoom the images.\n"
            "- Ensure the input image is in a supported format (e.g., PNG, JPG, TIFF)."
        )
        messagebox.showinfo("Help", help_text)

    def load_input_image(self):
        try:
            # Load and display the input image immediately
            # self.input_image = Image.open(self.input_file)

            sat_img, meta = read_multiband_image(self.input_file)
            self.meta = meta
            # print(self.meta)
            sat_img_full = sat_img.copy()
            sat_img = sat_img[[4, 3, 2]]
            self.input_image_org = np.transpose(sat_img_full, (1, 2, 0))

            sat_img = convert_16bit_to_8bit_minmax(sat_img)
            self.input_image = enhance_contrast(sat_img)

            self.input_image = Image.fromarray(np.transpose(self.input_image, (1, 2, 0)))

            self.zoom_factor = 1.0
            self.display_images()  # Show the input image as soon as it is loaded

            # Compute PCA and display it
            self.compute_and_display_pca()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load input image: {str(e)}")

    def display_images(self):
        if self.input_image is None:
            return

        # Resize the input image
        width_input, height_input = int(self.input_image.width * self.zoom_factor), int(
            self.input_image.height * self.zoom_factor)
        resized_input_image = self.input_image.resize((width_input, height_input), Image.Resampling.LANCZOS)

        self.input_image_tk = ImageTk.PhotoImage(resized_input_image)

        if self.canvas_input_image is None:
            self.canvas_input_image = self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.input_image_tk)
        else:
            self.canvas_input.itemconfig(self.canvas_input_image, image=self.input_image_tk)

        self.canvas_input.config(scrollregion=self.canvas_input.bbox(self.canvas_input_image))

        # If PCA image is available, display it
        if self.pca_image is not None:
            resized_pca_image = self.pca_image.resize((width_input, height_input), Image.Resampling.LANCZOS)
            self.output_image_tk = ImageTk.PhotoImage(resized_pca_image)

            if self.canvas_output_image is None:
                self.canvas_output_image = self.canvas_output.create_image(0, 0, anchor=tk.NW,
                                                                           image=self.output_image_tk)
            else:
                self.canvas_output.itemconfig(self.canvas_output_image, image=self.output_image_tk)

            self.canvas_output.config(scrollregion=self.canvas_output.bbox(self.canvas_output_image))

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event):
        if self.canvas_input_image is None or self.canvas_output_image is None:
            return

        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        self.canvas_input.move(self.canvas_input_image, dx, dy)
        self.canvas_output.move(self.canvas_output_image, dx, dy)

        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def zoom_image(self, event):
        if self.input_image is None or self.pca_image is None:
            return

        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor *= 0.9

        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))

        self.display_images()

    def compute_and_display_pca(self):
        if self.input_image is None:
            return

        try:
            n_components = int(self.n_components_entry.get())
            self.n_components_final = n_components
            if n_components <= 0:
                raise ValueError("Number of components must be positive.")

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter a valid number of components: {e}")
            return

        # img_array = np.array(self.input_image.convert('RGB'))
        img_array = self.input_image_org

        # reconstructed_image = pca_inbuild(img_array, n_components)

        img_array = np.transpose(img_array, (2, 0, 1))
        reconstructed_image, eigenvalues, selected_components = perform_pca_reconstruction(img_array, n_components)

        self.PCs = selected_components
        self.eigenvalues = eigenvalues
        reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
        self.input_image_synthesis = reconstructed_image
        # Reshape the image back to its original HxWxC shape

        reconstructed_image = np.transpose(reconstructed_image, (2, 0, 1))
        reconstructed_image = convert_16bit_to_8bit_minmax(reconstructed_image)
        reconstructed_image = enhance_contrast(reconstructed_image)
        reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
        reconstructed_image = reconstructed_image[:, :, [4, 3, 2]]
        # Convert the PCA image back to PIL format
        self.pca_image = Image.fromarray(reconstructed_image)

        # Update the display with PCA
        self.display_images()

        # Calculate and display statistics
        self.calculate_statistics()

    def calculate_statistics(self):
        if self.input_image is None or self.pca_image is None:
            return

        # input_array = np.array(self.input_image.convert('L'))
        # pca_array = np.array(self.pca_image.convert('L'))

        input_array = np.array(self.input_image_org)
        pca_array = np.array(self.input_image_synthesis)

        # Calculate PSNR
        # psnr_value = psnr(input_array, pca_array)
        psnr_value = calculate_psnr(input_array, pca_array)

        # Calculate SSIM
        # ssim_value = ssim(input_array, pca_array)
        ssim_value = calculate_ssim(input_array, pca_array)
        # Calculate Explained Variance
        variance_percent = calculate_variation_percentage(self.eigenvalues, self.n_components_final)

        # # Calculate Standard Deviation of the Difference
        # diff = input_array - pca_array
        # std_diff = np.std(diff)

        # Display results in the table
        stats_text = f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.6f}, Explained Variance (%): {variance_percent:.2f}"
        self.stats_table.config(text=stats_text)

    def save_pca_image(self):
        if self.output_file and self.pca_image:
            try:
                # Save the PCA image to the output file
                # self.pca_image.save(self.output_file)
                # write_image(self.output_file, np.transpose(self.input_image_synthesis, (2,0,1)), self.meta)
                meta = self.meta
                meta.update({
                    "count": self.n_components_final,  # Update number of bands
                    "height": self.PCs.shape[0],  # Update height
                    "width": self.PCs.shape[1],  # Update width
                    "dtype": 'float32'
                })
                # print(meta)
                # print(self.PCs.dtype)
                # print(self.PCs.min())
                # print(self.PCs.max())

                write_image(self.output_file, np.transpose(self.PCs, (2, 0, 1)), meta)
                messagebox.showinfo("Success", f"PC image saved successfully!\n{str(self.output_file)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save PC image: {str(e)}")


# Main program
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
