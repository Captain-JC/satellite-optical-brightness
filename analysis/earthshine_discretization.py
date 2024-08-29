# %%
"""
### Earthshine Discretization
This code shows a comparison between using standard spherical coordinates
to discretize the Earth's surface and our custom coordinate system.
"""

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt

# %%
# Spherical coordinates (phi, theta)
# The zenith angle is phi
# The azimuthal angle is theta
phi = np.linspace(0, 45, 10)
theta = np.linspace(0, 360, 40)
phi, theta = np.radians(phi), np.radians(theta)
phi, theta = np.meshgrid(phi, theta)

# %%
# Custom coordinates (psi, omega)
# The angle-off-plane is psi
# The angle-on-plane is omega
psi = np.linspace(-45, 45, 20)
omega = np.linspace(-45, 45, 20)
psi, omega = np.radians(psi), np.radians(omega)
psi, omega = np.meshgrid(psi, omega)

# %%
# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4), dpi = 300)

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

# Standard spherical coordinates
x, y = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta)
ax1.scatter(x, y, s = 10)
ax1.set_xlabel(r"Spherical Coordinates $(\theta, \phi)$", fontsize = 15)

# Custom coordinate system
z = 1 / np.sqrt( 1 + np.tan(psi)**2 + np.tan(omega)**2 )
x = z * np.tan(psi)
y = z * np.tan(omega)

ax2.scatter(x, y, s = 10)
ax2.set_xlabel(r"Custom Coordinates $(\Psi, \Omega)$", fontsize = 15)

plt.show()