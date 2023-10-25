# 원시 이미지에서 플랫 필드 보정을 수행하고 수정된 이미지를 HDF5 파일에 저장
# 그런 다음 수정된 이미지를 로드하고 처음 1000개 이미지에서 배경을 빼고 픽셀 값을 0-255 범위로 재조정
# 결과를 8비트 정수로 변환하고 결과 이미지를 다시 HDF5 파일에 저장

# Import necessary packages
from pathlib import Path
import os
import numpy as np
import h5py
from tqdm import tqdm, trange  # Import the progress bar libraries
from skimage import exposure  # Import exposure module for histogram equalization


#############################################################################
# Define default script location
#############################################################################
__location__ = Path("E:/230714_ME-1573/1_HDF5/Process_folder")  # Input directory

# Get all .hdf5 files in the directory
h5_files = list(__location__.glob("*.hdf5"))

# h5_file = "0702_Ti64-Virgin_2.hdf5"  # Input HDF5 file

#############################################################################
# Main body
#############################################################################
for h5_file in tqdm(h5_files, desc='Processing files'):  # Wrap the file loop with tqdm
    with h5py.File(__location__ / h5_file, 'r+') as f:  # Open file in read/write mode
        # Check if corrected dataset already exists
        if '/ffCorr' in f:
            del f['/ffCorr']  # If it exists, delete it
        
        # Load the flats and raw frames from the HDF5 file
        flats = f['/flats'][:200].astype(np.float32)  # Get first 200 flats and convert to float32
        raw = f['/raw'][:].astype(np.float32)  # Convert raw data to float32
        
        # Compute the average flat image
        flatAvg = np.mean(flats, axis=0)
        
        # Initialize an empty list to hold corrected frames
        ffCorr = []
        
        # Process each frame in the raw data
        for frame in tqdm(raw, desc='Processing frames', leave=False):  # Wrap with tqdm for a progress bar
            # Flat field correction
            corrected_frame = frame / flatAvg  # Subtract the average flat from the raw frames

            ffCorr.append(corrected_frame)  # Add the corrected frame to our list
        
        # Convert the list of corrected frames to a numpy array
        ffCorr = np.array(ffCorr)

        # Save corrected frames to the HDF5 file
        f.create_dataset('/ffCorr', data=ffCorr)  # Create a new ffCorr dataset
        
        # Load the ffCorr frames from the HDF5 file
        ffCorr = f['/ffCorr'][:]
        
        # Compute the average of the first 200 ffCorr images
        bgAvg = np.mean(ffCorr[:200], axis=0)
        
        # Background subtraction
        num_images = 1000  # Number of images to subtract background from (601-1600)
        bgSub = np.empty_like(ffCorr[:num_images])  # Initialize bgSub array
        for i in trange(num_images, desc='Subtracting background', leave=False):  # Use trange for progress bar
            bgSub[i] = ffCorr[i+601] / bgAvg  # Subtract bgAvg from each image

        # Define original range and target range
        orig_min, orig_max = 0.4, 1.6 # Ti64: 0.6, 1.4 / GRCop-42: 0.4, 1.6 / Inconel718: 0.3, 1.7
        new_min, new_max = 0, 255

        # Rescale data to new range
        bgSub_rescaled = (bgSub - orig_min) * (new_max - new_min) / (orig_max - orig_min) + new_min

        # Ensure values are within 0-255 range
        bgSub_clipped = np.clip(bgSub_rescaled, new_min, new_max)

        # Cast to 8-bit
        bgSub8 = bgSub_clipped.astype(np.uint8)

        # Save background subtracted frames to the HDF5 file
        if '/bgSub8' in f:
            del f['/bgSub8']  # If exists, delete the dataset
        f.create_dataset('/bgSub8', data=bgSub8)  # Create a new bgSub8 dataset
