import numpy as np

# Load the settings file 
ops = np.load('settings.npy', allow_pickle=True).item()

# Check specific numerical float values
print("Threshold Scaling:", ops.get('threshold_scaling'))
print("Tau:", ops.get('tau'))
print("SNR Threshold:", ops.get('snr_thresh'))
print("Max Reg Shift:", ops.get('maxregshift'))

# To print a list of every single setting to see what else is configured:
for key, value in ops.items():
     print(f"{key}: {value}")