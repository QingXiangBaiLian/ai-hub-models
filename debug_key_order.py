"""Debug script to check if DatasetEntries preserves insertion order."""
from qai_hub.client import DatasetEntries
import numpy as np

# Test if DatasetEntries preserves insertion order
d = {'z_input': [np.zeros((1,2))], 'a_input': [np.zeros((2,3))], 'm_input': [np.zeros((3,4))]}
entries = DatasetEntries(d)
print('Original keys:', list(d.keys()))
print('DatasetEntries keys:', list(entries.keys()))
print('Same order:', list(d.keys()) == list(entries.keys()))
