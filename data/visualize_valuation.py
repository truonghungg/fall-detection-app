import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
data_path = 'data/dataset.csv'
data = pd.read_csv(data_path)

# Inspect the dataset to ensure column names and structure
print(data.head())
print(data.columns)

# Define the actions based on unique labels in the 'ActivityLabel' column
actions = data['ActivityLabel'].unique()

# Color palette for plotting
colors = ['blue', 'orange', 'green']
axes = ['X', 'Y', 'Z']

# Loop through each action and create separate plots
for action in actions:
    # Filter data for the specific action
    action_data = data[data['ActivityLabel'] == action]
    
    # Create a figure for the action
    plt.figure(figsize=(20, 12))
    
    # Accelerometer Subplot
    plt.subplot(2, 1, 1)
    plt.title(f'Time Series - {action} (Accelerometer)', fontsize=16)
    plt.ylabel('Acceleration (m/s²)', fontsize=12)
    for i, axis in enumerate(['AccelX', 'AccelY', 'AccelZ']):
        if axis in action_data.columns:  # Check if the column exists
            plt.plot(action_data[axis].values[:3000], color=colors[i], label=f'Acceleration {axes[i]}')
    plt.legend(fontsize=12)
    
    # Gyroscope Subplot
    plt.subplot(2, 1, 2)
    plt.title(f'Time Series - {action} (Gyroscope)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Angular Velocity (deg/s)', fontsize=12)
    for i, axis in enumerate(['GyroX', 'GyroY', 'GyroZ']):
        if axis in action_data.columns:  # Check if the column exists
            plt.plot(action_data[axis].values[:3000], color=colors[i], label=f'Rotation {axes[i]}')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

print("Visualization of data for all actions is complete.")
