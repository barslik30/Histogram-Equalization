import os
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import PIL

GITHUB_PATH = 'https://raw.githubusercontent.com/dnevo/ImageProcessing/main/images/'


def read_file(file_name: str, file_path: str = GITHUB_PATH) -> np.ndarray:
    file_path = os.path.join(file_path, file_name)
    response = requests.get(file_path)
    fp = BytesIO(response.content)

    img_pil = PIL.Image.open(fp)
    return np.array(img_pil, dtype='int16')


def plot_img(img: np.array, figsize: (int, int) = None):
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)


def hist_shape(img_src: np.array, img_dest: np.array) -> np.array:
    hist_src, _ = np.histogram(img_src.flatten(), bins=256, range=[0, 256])
    cdf_src = hist_src.cumsum()
    cdf_norm_src = ((cdf_src - cdf_src.min()) * 255) / (cdf_src.max() - cdf_src.min())
    img_new = np.interp(img_src.flatten(), range(256), cdf_norm_src).reshape(img_src.shape)
    return img_new.astype('int16')


def create_tiled_image(img_big: np.array, img_small: np.array) -> np.array:
    tile_size = img_small.shape[0]
    img_new = np.zeros_like(img_big)
    for i in range(0, img_big.shape[0], tile_size):
        for j in range(0, img_big.shape[1], tile_size):
            tile_big = img_big[i:i + tile_size, j:j + tile_size]
            tile_small = img_small[i:i + tile_size, j:j + tile_size]
            img_new[i:i + tile_size, j:j + tile_size] = hist_shape(tile_big, tile_small)
    return img_new.astype('int16')


# Example: Change 'example_image.tiff' to the image you want to use
example_img = read_file(file_name='fftDemo3.tiff')
plot_img(example_img)

# Example: Change 'flatHistShape.tiff' to the reference image for histogram equalization
reference_img = read_file(file_name='flatHistShape.tiff')

# Example: Apply histogram equalization to the example image using the reference image
enhanced_img = hist_shape(example_img, reference_img)
plot_img(enhanced_img)

# Example: Change 'smallMozart.tiff' to the small image for creating a tiled image
small_img = read_file(file_name='smallMozart.tiff')

# Example: Change 'bigMozart.tiff' to the big image for creating a tiled image
big_img = read_file(file_name='bigMozart.tiff')

# Example: Create a tiled image using histogram equalization
tiled_img = create_tiled_image(big_img, small_img)
plot_img(tiled_img, figsize=(12, 12))
plt.show()
