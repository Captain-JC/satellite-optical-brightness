# %%
"""
### Diffuse Sphere Model Fitting
This code fits the albedo-area parameter in the diffuse sphere model to
observations of Starlink v1.5 brightness from the Pomenis observatory
"""

# %%
# Imports
import pandas as pd
import numpy as np
import astropy.time
import astropy.coordinates
import scipy.optimize
import matplotlib.pyplot as plt
import satellite_models.diffuse_sphere as diffuse_sphere
import lumos.calculator

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
# Calculate diffuse sphere model intensities
# for albedo-area product of 1

sphere_intensities = np.zeros_like(measured_magnitudes)

for i, (sat_alt, sat_az, sat_h, time) \
    in enumerate(zip(satellite_altitudes, satellite_azimuths, satellite_heights, observation_times)):

    observation_time = astropy.time.Time(time, format = 'isot')
    sun_alt, sun_az = lumos.calculator.get_sun_alt_az(observation_time, mount_lemmon)

    sphere_intensities[i] = \
        diffuse_sphere.get_intensity(
            1,
            sat_h,
            sat_alt,
            sat_az,
            sun_alt,
            sun_az)

# %%
# Scale our precalculated values to a given area-albedo product
# and convert to AB Magnitude

def sphere_model_optimization(area_albedo):
    
    sphere_magnitudes = lumos.conversions.intensity_to_ab_mag(area_albedo * sphere_intensities)

    error = sphere_magnitudes - measured_magnitudes
    rms = np.sqrt( np.mean( error**2 ) )

    return rms

sol = scipy.optimize.minimize_scalar(sphere_model_optimization, bounds = (0, 100))

print(f"Best Fit Albedo-Area Product: {sol.x:0.3f}")
print(f"RMS : {sol.fun:0.3f}")
        