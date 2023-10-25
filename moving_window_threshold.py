import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter

def crop_image(image, x_center, y_top, width, height):
    cropped = np.zeros((height, width), dtype=image.dtype)  # 원본 이미지와 같은 dtype 사용
    
    x_start = max(x_center - width // 2, 0)
    y_start = max(y_top, 0)
    x_end = min(x_center + width // 2, image.shape[1])
    y_end = min(y_top + height, image.shape[0])
    
    cropped[0:y_end-y_start, x_start-x_center+width//2:x_end-x_center+width//2] = image[y_start:y_end, x_start:x_end]
    
    return cropped

def threshold_and_filter(image, threshold_value, radius):
    thresholded_image = (image == threshold_value).astype(image.dtype)
    filtered_image = median_filter(thresholded_image, size=radius)
    return filtered_image

def process_images(h5_file_path, dataset_path, start_frame, end_frame, output_dataset, window_width, window_height, x_position_init, y_position_init, object_speed, threshold_value, median_filter_radius):
    with h5py.File(h5_file_path, 'a') as file:
        images = file[dataset_path]
        moving_window_images = []

        x_position = x_position_init
        speed_in_pixels_per_frame = (object_speed * 1000) / (4.42 * 40000)

        for frame in tqdm(range(start_frame-1, end_frame), desc="Processing images"):
            image = images[frame]
            processed_image = threshold_and_filter(image, threshold_value, median_filter_radius)
            cropped = crop_image(processed_image, round(x_position), y_position_init, window_width, window_height)
            moving_window_images.append(cropped)
            x_position += speed_in_pixels_per_frame
            
        if output_dataset in file:
            del file[output_dataset]
            
        file.create_dataset(output_dataset, data=moving_window_images)

# Adjust the parameters as needed
h5_file_path = "G:/230714_ME-1573/1_HDF5/0702_Ti64-Virgin_3.hdf5"
dataset_path = "/seg8"
output_dataset = "/moving_window_keyhole_seg"
start_frame = 1
end_frame = 214
window_width = 220  # adjust the window width
window_height = 180  # adjust the window height
x_position_init = 43  # initial x_position (center of the window) / 0702_Ti64-Virgin_3: 43 / 0718_Ti64-Blue_3: 43
y_position_init = 251  # initial y_position (top of the window) / 0702_Ti64-Virgin_3: 251 / 0718_Ti64-Blue_3: 297
object_speed = 750  # adjust the object speed
threshold_value = 1  # Choose the pixel value to keep (0-4)
median_filter_radius = 4  # Choose the radius for the median filter

process_images(h5_file_path, dataset_path, start_frame, end_frame, output_dataset, window_width, window_height, x_position_init, y_position_init, object_speed, threshold_value, median_filter_radius)
