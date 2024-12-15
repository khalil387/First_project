import matplotlib
import sys
import matplotlib.pyplot as plt
import numpy
z=numpy.random.randint(9999)
x = numpy.random.uniform(0.0, 3.0, z)
#plt.plot(x, 'o:g')
plt.hist(x, 10)
print(z)

plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

