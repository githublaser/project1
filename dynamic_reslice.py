import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Parameters
h5_file_path = "G:/230714_ME-1573/1_HDF5/0718_Ti64-Blue_3.hdf5"
dataset_path = "/moving_window_keyhole_seg"
y_position = 64
max_x = 220
frame_rate = 40e3  # 40 kHz
pixel_size = 4.315  # µm/pixel


# Functions
def load_data(h5_file_path, dataset_path):
    with h5py.File(h5_file_path, 'r') as file:
        data = np.array(file[dataset_path])
    return data


def find_boundary_x(image_stack, y=64, max_x=220):
    rear_wall = []
    front_wall = []

    for img in tqdm(image_stack, desc='Processing images', unit='image'):
        row = img[y, :]
        rear_found = False
        for x in range(1, len(row)):
            if row[x-1] < 0.5 and row[x] >= 0.5 and not rear_found:
                rear_wall.append((max_x - x) * pixel_size)
                rear_found = True
            elif row[x-1] >= 0.5 and row[x] < 0.5:
                front_wall.append((max_x - x) * pixel_size)
                break

    return rear_wall, front_wall


def calculate_amplitude(rear_wall, front_wall):
    return [abs(r - f) for r, f in zip(rear_wall, front_wall)]


def calculate_frequency(amplitudes, threshold):
    frequency = 0
    above_threshold = False
    for amp in amplitudes:
        if amp >= threshold and not above_threshold:
            frequency += 1
            above_threshold = True
        elif amp < threshold:
            above_threshold = False
    return frequency


def analyze_boundaries(rear_wall, front_wall, time):
    amplitudes = calculate_amplitude(rear_wall, front_wall)
    quarter, three_quarters = len(amplitudes) // 4, (3 * len(amplitudes)) // 4
    amplitudes_25_75 = amplitudes[quarter:three_quarters]

    max_amplitude = max(amplitudes_25_75)
    freq_50 = calculate_frequency(amplitudes_25_75, 0.5 * max_amplitude)
    freq_75 = calculate_frequency(amplitudes_25_75, 0.75 * max_amplitude)

    return max_amplitude, freq_50, freq_75


def plot_mean_lines(time, rear_wall, front_wall, quarter, three_quarters):
    rear_mean = sum(rear_wall[quarter:three_quarters]) / len(rear_wall[quarter:three_quarters])
    front_mean = sum(front_wall[quarter:three_quarters]) / len(front_wall[quarter:three_quarters])
    
    plt.axhline(y=rear_mean, color='r', linestyle='dashed', label='25-75% Mean Rear Wall')
    plt.axhline(y=front_mean, color='b', linestyle='dashed', label='25-75% Mean Front Wall')
    
    plt.text(time[-1], rear_mean, f'{rear_mean:.2f}', color = 'r')
    plt.text(time[-1], front_mean, f'{front_mean:.2f}', color = 'b')


# Main code
image_stack = load_data(h5_file_path, dataset_path)
rear_wall, front_wall = find_boundary_x(image_stack, y=y_position, max_x=max_x)
time = [1000 * i / frame_rate for i in range(len(rear_wall))]
quarter, three_quarters = len(rear_wall) // 4, (3 * len(rear_wall)) // 4

# Analyze boundaries
max_amplitude, freq_50, freq_75 = analyze_boundaries(rear_wall, front_wall, time)

# Save to Excel
file_path = 'walls_data.xlsx'
df = pd.DataFrame({'Time_ms': time, 'Rear_Wall_µm': rear_wall, 'Front_Wall_µm': front_wall,
                   'Max_Amplitude_µm': [max_amplitude] * len(time),
                   'Frequency_50%': [freq_50] * len(time), 'Frequency_75%': [freq_75] * len(time)})
df.to_excel(file_path, index=False)
print(f"The data has been saved to '{file_path}'")

# Create and save the plot
plt.plot(time, rear_wall, 'r', label='Rear Wall')
plt.plot(time, front_wall, 'b', label='Front Wall')
plot_mean_lines(time, rear_wall, front_wall, quarter, three_quarters)
plt.xlabel('Time (ms)')
plt.ylabel('Boundary x coordinate (µm)')
plt.title(f'Boundary x coordinates at y={y_position}')
plt.legend()
plt.grid(True)
plt.savefig('walls_graph.png', dpi=300)
plt.show()