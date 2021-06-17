w = open("randomNums.txt","w")

import random

for i in range(10000):
    w.write(str(random.randrange(1,9)))
