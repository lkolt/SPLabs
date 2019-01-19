import numpy as np
import matplotlib.pyplot as plt

cap = []
ren = []
with open("task_1_capital.txt", 'r') as file:
  next(file)
  for line in file.readlines():
    a = line.split()
    cap.append([int(a[0])])
    ren.append([int(a[1])])

x = np.array(cap)
A = np.ones((x.shape[0],1), dtype=int)
A = np.concatenate((A, x), axis=1)
y = np.array(ren)
w = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
z = w[0] + w[1] * x

plt.plot(x, y, '.', x, z)
plt.xlabel('Capital')
plt.ylabel('Rental')
plt.show()
