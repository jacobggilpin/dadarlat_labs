import sys
import os
import numpy as np
import h5py
from sbxreader import sbx_memmap

def convert_sbx_to_suite2p_h5(sbx_path, chunk_size=200):
    sbx_path = os.path.abspath(sbx_path)
    if not os.path.exists(sbx_path):
        raise FileNotFoundError(f"Could not find file: {sbx_path}")

    h5_path = sbx_path[:-4] + '.h5' if sbx_path.endswith('.sbx') else sbx_path + '.h5'

    # Remove old .h5 file if it exists so we start 100% fresh
    if os.path.exists(h5_path):
        os.remove(h5_path)

    dat = sbx_memmap(sbx_path)
    nframes, nplanes, nchannels, height, width = dat.shape
    ndeadcols = dat.ndeadcols if hasattr(dat, 'ndeadcols') and dat.ndeadcols > 0 else 0
    out_width = width - ndeadcols

    print(f"Loaded {sbx_path}")
    print(f"Shape: {nframes} frames, {nplanes} planes, {nchannels} channels, {height}x{out_width}")
    print(f"Frame rate (fs): {dat.metadata.get('frame_rate', 15.625)} Hz")

    total_2d_frames = nframes * nplanes * nchannels
    frames_per_tp = nplanes * nchannels

    with h5py.File(h5_path, 'w') as hf:
        dset = hf.create_dataset(
            'data',
            shape=(total_2d_frames, height, out_width),
            dtype='int16',
            chunks=(min(chunk_size * frames_per_tp, total_2d_frames), height, out_width)
        )

        for start in range(0, nframes, chunk_size):
            end = min(start + chunk_size, nframes)
            # Load all planes and channels: shape (chunk, nplanes, nchannels, height, width)
            block = (np.array(dat[start:end])[:, :, :, :, ndeadcols:] // 2).astype(np.int16)
            # Reshape into (chunk * nplanes * nchannels, height, out_width)
            dset[start * frames_per_tp : end * frames_per_tp] = block.reshape(-1, height, out_width)
            print(f"Converted {end}/{nframes} frames...", end='\r')

    print(f"\nDone! Saved Suite2p HDF5 file: {h5_path}")
    print(f"--> MUST SET IN SUITE2P GUI: nplanes={nplanes}, nchannels={nchannels}, fs={dat.metadata.get('frame_rate', 15.625)}")

if __name__ == '__main__':
    default_path = r'C:\Users\jacob\Documents\dadarlat\day_1\6416_1R1L_000_000.sbx'
    sbx_file = sys.argv[1] if len(sys.argv) > 1 else default_path
    if sbx_file.startswith('\\') and not sbx_file.startswith('\\\\') and not os.path.exists(sbx_file):
        sbx_file = sbx_file.lstrip('\\')
    convert_sbx_to_suite2p_h5(sbx_file)