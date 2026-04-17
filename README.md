# dsprobe

A modular, open‑source Python framework for autonomous deep‑space navigation using celestial beacons — pulsars, radio relays, and optical landmarks. 

This system implements extended/unscented Kalman filters, fault detection, beacon selection, collision avoidance, and more. It is designed for rapid prototyping, education, and technology development for future lunar, Martian, and interstellar missions.


🎯 Purpose & Vision

Space navigation beyond Earth’s GPS relies on ground‑based tracking (e.g., NASA’s Deep Space Network) which introduces communication delays and single‑point failures. This project explores fully autonomous, beacon‑based navigation where a spacecraft locks onto natural or artificial “signposts” (pulsars, planets, relay satellites) to determine its position without Earth contact.

The system serves as:

  Research testbed for advanced filters (EKF, UKF, Particle, GPU‑EKF)
  Educational platform for aerospace students
  Prototyping suite for small‑sat missions (CubeSats, lunar landers)
  Technology demonstrator for future interstellar probes
  
Vision

To enable spacecraft to “find their way” across the solar system — and eventually to the stars — using a network of reliable beacons.


## Features

### Navigation Filters
- **EKF**: Extended Kalman Filter (baseline)
- **UKF**: Unscented Kalman Filter (nonlinear systems)
- **Particle Filter**: For multimodal distributions
- **GPU-EKF**: Batch processing on GPU (CuPy)

### Beacon Types
- **Pulsars**: X-ray pulsar navigation (XNAV)
- **Radio Beacons**: DSN, relay satellites
- **Optical**: Planetary landmarks, asteroids
- **Laser**: Retroreflector ranging

### Advanced Capabilities
- **Fault Detection**: RANSAC, chi-squared gating, Mahalanobis distance
- **Beacon Selection**: Greedy PDOP minimization, adaptive reliability weighting, reinforcement learning
- **Collision Avoidance**: Keep-out zones, trajectory prediction, maneuver computation
- **Relativistic Corrections**: Light-time, Shapiro delay
- **Optical Flow Integration**: Close-range navigation
- **IMU Fusion**: Dead reckoning integration

### Integration
- **SPICE**: NASA's SPICE toolkit for accurate ephemerides
- **ROS2**: Optional bridge for robotic spacecraft
- **CCSDS**: Telemetry standard compatibility


## Installation

```bash

git clone https://github.com/DapperHoldings/dsprobe.git
cd dsprobe
pip install -e .[ml,gpu,cv,viz]  # install with optional dependencies

```
## 🚀 **How to Run the System**

### 1. **Install dependencies**
```bash
pip install -r requirements.txt
# For ML features:
pip install -e .[ml]
# For GPU:
pip install -e .[gpu]
```

### 2. **Run simulation**
```bash
python main.py simulate --filter ukf --output mission_results.json
```

### 3. **Run tests**
```bash
pytest tests/ -v
```

### 4. **Launch dashboard** (requires Plotly/Dash)
```python
from visualization.dashboard import launch_dashboard
launch_dashboard()
```




