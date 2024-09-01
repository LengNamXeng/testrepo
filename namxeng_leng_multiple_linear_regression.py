# This activity is about performing a multiple linear regression "by hand",
# meaning that our model for our data is assumed to be a linear combination of
# multiple independent variables, as well as an intercept, to produce our
# dependent variable. Mathematically, we write this as:
#   y = b_0 + b_1 * x_1 + b_2 * x_2 + ... + b_n * x_n
# We can include as many independent variables as we like, but for this
# exercise, we will restrict ourselves to 2 to make the visualization easier.

import numpy as np
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from sklearn import datasets

# load the data
linnerud = datasets.load_linnerud(as_frame=True)

# print the description
print(linnerud.DESCR)

# We are going to use the number of chin-ups and the number of sit-ups as our
# independent variables (x and y, respectively), and the waist size (in inches)
# as the dependent variable (z).
x = linnerud["data"]["Chins"]
y = linnerud["data"]["Situps"]
z = linnerud["target"]["Waist"]

# Let's make a plot. It will be 3d because we have 3 dimensions of data.
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x,y,z)
ax.set_title("Raw Data from the Linnerud Dataset")
ax.set_xlabel("Chin-Ups")
ax.set_ylabel("Sit-Ups")
ax.set_zlabel("Waist Circumference [in]")
ax.set_box_aspect(aspect=None, zoom=0.8)  # zoom out to avoid cutting off labels
plt.show()

# Now let's make 2 2d plots showing the same information
fig = plt.figure(figsize=(10, 4))
plt.suptitle("Raw Data from the Linnerud Dataset")

# left-hand plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x,z)
ax1.set_xlabel("Chin-Ups")
ax1.set_ylabel("Waist Circumference [in]")

# right-hand plot
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(y,z)
ax2.set_xlabel("Sit-Ups")
ax2.set_ylabel("Waist Circumference [in]")

plt.tight_layout()
plt.show()

# Now we're going to use our old friend linear least squares to find the
# best-fit parameters for our multiple linear regression. Let's set it up:
npts = len(x)
Amat = np.zeros((npts, 3), dtype=np.float64)
Amat[:, 0] = x
Amat[:, 1] = y
Amat[:, 2] = 1

# set up target vector
bvec = z

# solve the system
result = lsq_linear(Amat, bvec)

# extract our solved parameters
m1 = result.x[0]  # slope 1
m2 = result.x[1]  # slope 2
b = result.x[2]   # z-intercept (offset)

# print out the equation
print(
    "The equation for the best-fit surface to the data is: "
    f"z = {m1:0.3f} x + {m2:0.3f} y + {b:0.2f}"
)

# plot the resulting surface
def surface(x, y):
    return m1 * x + m2 * y * b

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# plot the data
ax.scatter(x,y,z)

# plot the surface
xmodel = np.linspace(x.min(),x.max(), num=1000)  # should go from minimum to maximum values of x
ymodel = np.linspace(y.min(),y.min(), num=1000)  # should go from minimum to maximum values of y
X, Y = np.meshgrid(xmodel, ymodel)
Z = surface(X, Y)
ax.plot_surface(X, Y, Z, alpha=0.6)

# add labels
ax.set_title("Multiple Linear Regression")
ax.set_xlabel("Chin-Ups")
ax.set_ylabel("Sit-Ups")
ax.set_zlabel("Waist Circumference [in]")

plt.show()
