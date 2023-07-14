import numpy as np
data = np.load('tempres_multi_blipp_large.npy')

sample_per_sec_values = []
Tflops_values = []
for entry in data:
    values = entry.split()
    sample_per_sec_values.append(float(values[6]))
    Tflops_values.append(float(values[8]))

average_sample_per_sec = np.mean(sample_per_sec_values)
average_Tflops = np.mean(Tflops_values)



print(f"Average samples per sec: {average_sample_per_sec}")
print(f"Average Tflops: {average_Tflops}")


