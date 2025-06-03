import json
import numpy as np
import struct

def convert_tfjs_to_numpy():
    with open('model.json', 'r') as f:
        model_json = json.load(f)
    with open('group1-shard1of1.bin', 'rb') as f:
        binary_data = f.read()
    weights = {}
    offset = 0
    for weight_entry in model_json['weightsManifest'][0]['weights']:
        name = weight_entry['name']
        shape = weight_entry['shape']
        dtype = weight_entry['dtype']
        size = np.prod(shape)
        if dtype == 'float32':
            bytes_per_element = 4
            data = struct.unpack(f'{size}f', binary_data[offset:offset + size * bytes_per_element])
            weight = np.array(data, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        weights[name] = weight
        offset += size * bytes_per_element
    np.save('model_weights.npy', weights, allow_pickle=True)
    print("Weights converted and saved as model_weights.npy")

if __name__ == "__main__":
    convert_tfjs_to_numpy()