import numpy as np
import matplotlib.pyplot as plt

def cubic_bspline (r, fac):
    ret = 0.0
    if (r <= 1 and r >= 0):
        ret = (1 - 1.5 * r * r *(1 - 0.5 *r)) * fac
    elif (r > 1 and r <= 2):
        ret = (2-r)*(2-r)*(2-r) * fac /4
    
    return ret

def scorr(x, p):
    energy = 0.0
    fac = 10/(7*np.pi)
    n = x.shape[0]
    for i in range(n):
        Wij = cubic_bspline(np.linalg.norm(x[i] - p), fac)
        Wdq = cubic_bspline(0.98, fac)
        energy += 0.5 * 1 * (Wij - Wdq) * (Wij - Wdq)
    return energy

x = np.array([[0,0],[0.1,0],[0.1,0.1],[0,0.1]])
pos = np.meshgrid(np.linspace(-0.1,0.2,100), np.linspace(-0.1,0.2,100))
energy_grid = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        p = [pos[0][i][j], pos[1][i][j]]
        energy_grid[i][j] = scorr(x, p)

plt.contourf(pos[0], pos[1], energy_grid, 100)
plt.scatter(x[:,0], x[:,1], c='r')
plt.colorbar()
plt.show()