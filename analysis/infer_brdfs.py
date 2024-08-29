# %%
"""
### BRDF from Brightness Observation
This code infers the BRDFs of the Starlink v1.5's primary surfaces
by reducing the error between observation and model prediction. The
BRDFs are assumed to be Phong BRDFs with 3 free parameters.
"""

# %%
# Imports
import pandas as pd
import numpy as np
import astropy.time
import astropy.coordinates
import scipy.optimize
from lumos.geometry import Surface
import lumos.brdf.library

# %%
# Read Pomenis database

data = pd.read_csv("../data/pomenis_observations.csv", comment = "#")

satellite_heights = 1000 * (data['satellite_height'].to_numpy())
observation_times = data['observation_time'].to_numpy()
satellite_altitudes = data['satellite_altitude'].to_numpy()
satellite_azimuths = data['satellite_azimuth'].to_numpy()
measured_magnitudes = data['ab_magnitude'].to_numpy()

mount_lemmon = astropy.coordinates.EarthLocation(lat = 32.4434, lon = -110.7881)

# %%
# Converts from observation time to Sun's altitude and azimuth
sun_alts = []
sun_azs = []
for time in observation_times:
    obs_time = astropy.time.Time(time, format = 'isot')
    sun_alt, sun_az = lumos.calculator.get_sun_alt_az(obs_time, mount_lemmon)
    sun_alts.append(sun_alt)
    sun_azs.append(sun_az)

# %%
# Main function to minimize RMS between model prediction and observations
def optimization(params):

    p1, p2, p3, p4, p5, p6 = params

    calculated_intensities = np.zeros_like(measured_magnitudes)

    # Sets up surfaces of satellite, one for the chassis and one for the solar array
    # Phong parameters, p1 - p6 vary
    surfaces = [
        Surface(1.0, np.array([0, 0, -1]), lumos.brdf.library.PHONG(p1, p2, p3)),
        Surface(1.0, np.array([0, 1, 0]), lumos.brdf.library.PHONG(p4, p5, p6))
        ]

    # Calculates predicted AB Mag for every observation
    for i, (sat_alt, sat_az, sat_h, sun_alt, sun_az) \
        in enumerate(zip(satellite_altitudes, satellite_azimuths, satellite_heights, sun_alts, sun_azs)):

        calculated_intensities[i] = lumos.calculator.get_intensity_observer_frame(
            surfaces,
            sat_h,
            sat_alt,
            sat_az,
            sun_alt,
            sun_az,
            include_sun = True,
            include_earthshine = False
            )
    
    calculated_magnitudes = lumos.conversions.intensity_to_ab_mag(calculated_intensities)

    # Return RMS between prediction and actual observation
    error = calculated_magnitudes - measured_magnitudes
    rms = np.sqrt( np.mean( error**2 ) )
    return rms

sol = scipy.optimize.minimize(
    optimization,
    x0 = (1, 1, 1, 1, 1, 1),
    bounds = ( (0, 20), (0, 20), (0, 200), 
               (0, 20), (0, 20), (0, 200)) )

Kd1, Ks1, n1, Kd2, Ks2, n2 = sol.x
print(f"RMS = {sol.fun:0.3f}")

# %%
print("Chassis Inferred BRDF Parameters")
print("------------------------------------")
print(f"Kd = {Kd1:0.2f}")
print(f"Ks = {Ks1:0.2f}")
print(f"n = {n1:0.2f}")

print("Solar Array Inferred BRDF Parameters")
print("------------------------------------")
print(f"Kd = {Kd2:0.2f}")
print(f"Ks = {Ks2:0.2f}")
print(f"n = {n2:0.2f}")