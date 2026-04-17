"""
Physical and astronomical constants used throughout the system.
"""

# Physical constants
SPEED_OF_LIGHT = 299792.458  # km/s (exact, defined)
GRAVITATIONAL_CONSTANT = 6.67430e-20  # km^3 kg^-1 s^-2 (2018 CODATA)
ASTRONOMICAL_UNIT = 149597870.7  # km (2012 IAU exact definition)

# Time constants
J2000_EPOCH = 2451545.0  # Julian date at J2000.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 31557600.0  # sidereal year

# Planetary parameters (simplified)
SUN_MASS = 1.32712440018e11  # km^3/s^2 (GM, not mass)
EARTH_MASS = 398600.4418  # km^3/s^2 (GM)
MARS_MASS = 42828.37562  # km^3/s^2 (GM)
JUPITER_MASS = 1.26686534e8  # km^3/s^2 (GM)

# Navigation accuracy requirements (typical)
LUNAR_ORBIT_ACCURACY = 0.1  # km (100 m)
MARS_ORBIT_ACCURACY = 1.0  # km
DEEP_SPACE_ACCURACY = 10.0  # km
INTERSTELLAR_ACCURACY = 1000.0  # km (1 million km)

# Default noise parameters
DEFAULT_RANGE_NOISE = {
    "radio": 0.01,      # 10 m precision
    "xray": 1000.0,     # 1000 km (pulsar TOA ~3ms)
    "optical": 10.0,    # 10 km (image centroiding)
    "laser": 0.001,     # 1 m (laser retroreflector)
    "gravity": 1e6,     # Very poor
}

DEFAULT_DIRECTION_NOISE = {
    "radio": 0.0001,    # 0.006 deg
    "xray": 0.001,      # 0.057 deg
    "optical": 0.001,   # 0.057 deg
    "laser": 0.00001,   # 0.0006 deg
    "gravity": 0.1,     # 5.7 deg
}

# Filter parameters
DEFAULT_PROCESS_NOISE_POS = 1e-6  # km^2/s^3
DEFAULT_PROCESS_NOISE_VEL = 1e-9  # km^2/s

# Chi-squared thresholds for innovation gating (95% confidence)
CHI2_THRESHOLD = {
    1: 3.841,
    2: 9.210,
    3: 11.345,
    4: 13.277,
    5: 15.086,
    6: 16.812,
}