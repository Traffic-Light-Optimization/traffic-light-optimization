import numpy as np

step = 220

rates = np.linspace(step, step*24, 24)

with open('./insertion_rate_generator/output.txt', 'w') as file:
    line = ' '.join(map(str, rates))
    file.write(line)