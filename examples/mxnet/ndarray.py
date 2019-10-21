from mxnet import nd

# Creating a 2D array
nd.array(((1, 2, 3), (5, 6, 7)))

# Create a very simple matrix with the same shape (2 rows by 3 columns) filled with ones
x = nd.ones((2, 3))

# Sampling values uniformly between -1 and 1 (the same shape)
y = nd.random.uniform(-1, 1, (2, 3))

# Fill an array of a given shape with a given value, such as 2.0
x = nd.full((2, 3), 2.0)

# As with NumPy, the dimensions of each NDArray are accessible by accessing the .shape attribute.
# We can also query its size, which is equal to the product of the components of the shape.
# In addition, .dtype tells the data type of the stored values.
(x.shape, x.size, x.dtype)

# Operations
x * y

y.exp()

# Grab a matrix’s transpose to compute a proper matrix-matrix product
nd.dot(x, y.T)

# Indexing
# MXNet NDArrays support slicing in all the ridiculous ways you might imagine accessing your data.

# Here’s an example of reading a particular element, which returns a 1D array with shape (1,).
var = y[1, 2]

# Read the second and third columns from y.
y[:, 1:3]

# and writing to a specific element
y[:, 1:3] = 2

# Multi-dimensional slicing is also supported.
y[1:2, 0:2] = 4

# Converting between MXNet NDArray and NumPy
# Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do not share memory.
a = x.asnumpy()
(type(a), a)
