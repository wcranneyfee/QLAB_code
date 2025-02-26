import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

arr_first_half = [n for i, n in enumerate(arr) if i < len(arr)/2]
arr_last_half = [n for i, n in enumerate(arr) if i >= len(arr)/2]

print(arr_first_half, arr_last_half)