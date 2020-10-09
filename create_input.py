# prints out an array of 1024 randomly generated float numbers
# output redirected to a file that seves as input to the sampling_discrete.cu code

import random
for i in range(1024):
print(str(random.randint(0,99)/100), end=" ")
print()
