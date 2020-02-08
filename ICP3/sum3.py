'''import numpy as np
x = np.random.randint(1,20,15)
print("Original array:")
print(x)
y=x.reshape(5,3)
y[y.argmax()] = 0

print("Maximum value replaced by 0:")
print(y)
print(np.where(y< 8,1,0 ))'''

import numpy as np

x = np.random.randint(1,20,15)
print(x)

y=x.reshape((3,5))
print(y)

maxRep=np.max(y,axis=1)
print(maxRep)

y[np.where(y == np.max(y,axis=1,keepdims=True))] = 0
print("Result:\n",y)