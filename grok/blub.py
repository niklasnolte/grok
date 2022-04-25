import numpy as np
import functools
import time
from numba import jit

n_gratings = 1000
n_wavelengths = 10000

x = np.random.rand(n_gratings, 2, 2, n_wavelengths) * 1.1

def multiply(x:np.array):
  # matmul all gratings
  x = x.reshape(n_gratings//2, 2,2, n_wavelengths * 2)
  return x

start = time.perf_counter()
ans2 = multiply(x)
end = time.perf_counter()

print(end - start)

# approach 1
# ans = x[0]
# for xi in x[1:]:
#   ans = np.einsum("ijl,jkl->ikl", ans, xi)


start = time.perf_counter()
ans2 = functools.reduce(lambda i, j: np.einsum("ijl,jkl->ikl", i, j), x)
end = time.perf_counter()

print(end - start)

# # matts approach
# @jit(nopython=True)
# def do_things(x,L):
#   Ln = np.zeros((n_wavelengths, 2, 2))
#   for i in range(n_gratings):
#     Ln[:,0,0] = L[:,0,0] * x[i,0,0,:] + L[:,0,1] * x[i,1,0,:] 
#     Ln[:,1,0] = L[:,1,0] * x[i,0,0,:] + L[:,1,1] * x[i,1,0,:]
#     Ln[:,0,1] = L[:,0,0] * x[i,0,1,:] + L[:,0,1] * x[i,1,1,:]
#     Ln[:,1,1] = L[:,1,0] * x[i,0,1,:] + L[:,1,1] * x[i,1,1,:]
#     L, Ln = Ln, L
#   return L

# start = time.perf_counter()
# L = np.repeat(np.eye(2)[None, ...],n_wavelengths, axis=0)
# do_things(x, L)

# mid = time.perf_counter()
# L = np.repeat(np.eye(2)[None, ...],n_wavelengths, axis=0)
# L = do_things(x,L)
# end = time.perf_counter()

# print(mid - start)
# print(end - mid)

# assert L.reshape(2, 2, n_wavelengths).all() == ans2.all()
# assert (ans == ans2).all()


# check
# xs = x[..., 0]
# ans1 = xs[0]
# for xi in xs[1:]:
#   ans1 = ans1 @ xi
# print(ans1)
