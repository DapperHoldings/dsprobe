"""
Main entry point for the DSProbe system.
Provides CLI for running simulations, demos, and missions.
"""

import numpy as np
from datetime import datetime, timezone, timedelta
import click
import json
from typing import Optional

from config.settings import NavConfig, FilterType, BeaconType
from core.beacon import Beacon, create_planet_ephemeris
from core.ephemeris import KeplerianEphemeris
from navigation.navigator import AdvancedBeaconNavigator
from utils.logging import NavLogger
from utils.timing import Timer

@click.group()
def cli():
    """DSProbe System Command Line Interface"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='JSON configuration file')
@click.option('--filter', '-f', type=click.Choice(['ekf', 'ukf', 'particle', 'gpu']),
              default='ekf', help='Filter type')
@click.option('--output', '-o', type=click.Path(), default='output.json',
              help='Output file for results')
def simulate(config: Optional[str], filter: str, output: str):
    """
    Run a full navigation simulation.
    """
    print("=" * 80)
    print("DSProbe SYSTEM - SIMULATION MODE")
    print("=" * 80)
    
    # Load config
    if config:
        nav_config = NavConfig.from_json(config)
        click.echo(f"Loaded configuration from {config}")
    else:
        nav_config = NavConfig(
            filter_type=FilterType(filter),
            debug_logging=True
        )
        click.echo("Using default configuration")
    
    # Create beacons (solar system example)
    beacons = []
    planet_names = ["earth", "mars", "jupiter"]
    for name in planet_names:
        ephem = create_planet_ephemeris(name)
        beacon = Beacon(
            id=f"planet_{name}",
            name=f"Planet {name.capitalize()}",
            beacon_type=BeaconType.OPTICAL,
            ephemeris=ephem,
            base_uncertainty=(10.0, 0.001)  # 10 km range, 0.057 deg
        )
        beacons.append(beacon)
    
    # Add an artificial radio beacon at Earth L2
    earth_pos = beacons[0].get_position(0.0)
    l2_pos = earth_pos + np.array([1.5e6, 0, 0])
    beacon_l2 = Beacon(
        id="earth_l2",
        name="Earth L2 Relay",
        beacon_type=BeaconType.RADIO,
        fixed_position=l2_pos,
        base_uncertainty=(0.01, 0.0001)  # 10 m, 0.006 deg
    )
    beacons.append(beacon_l2)
    
    # Initialize navigator
    navigator = AdvancedBeaconNavigator(beacons, nav_config, filter_type=filter)
    
    # Set initial state (near Earth, with some error)
    true_start = beacons[0].get_position(0.0) + np.array([0, 1000, 500])
    initial_state = true_start + np.random.normal(0, 100, 3)  # 100 km error
    navigator.initialize(
        initial_position=initial_state,
        initial_velocity=np.array([5.0, 0, 0])  # ~5 km/s away from Earth
    )
    
    # Simulation timeline
    t0 = 0.0
    dt = 600.0  # 10 minute steps
    steps = 144  # 1 day
    
    results = []
    
    with Timer("Total simulation", navigator.logger):
        for step in range(steps):
            t = t0 + step * dt
            current_time = datetime(2000,1,1,12,0,0, tzinfo=timezone.utc) + timedelta(seconds=t)
            
            # Update time
            navigator.update_time(current_time)
            
            # Predict
            navigator.predict(dt)
            
            # Select beacons
            selected = navigator.beacon_selector.select_beacons(
                list(navigator.beacons.values()),
                navigator.filter,
                t,
                n_select=4
            )
            
            # Acquire measurements
            measurements = navigator.acquire_measurements(selected)
            
            # Process
            state, cov = navigator.process_measurements(measurements)
            
            # Get solution
            sol = navigator.get_solution()
            results.append(sol)
            
            if step % 12 == 0:  # every 2 hours
                click.echo(f"Step {step:3d}: pos={state[0]/1e3:.1f}, {state[1]/1e3:.1f}, {state[2]/1e3:.1f} km, "
                          f"PDOP={sol['pdop']:.1f}")
    
    # Save results
    with open(output, 'w') as f:
        # Convert numpy arrays to lists for JSON
        serializable = []
        for r in results:
            r_ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in r.items()}
            serializable.append(r_ser)
        json.dump(serializable, f, indent=2)
    click.echo(f"Results saved to {output}")
    
    # Summary
    final = results[-1]
    click.echo("\nFINAL STATE:")
    click.echo(f"Position: {final['position']/1e3:.1f} km")
    click.echo(f"Velocity: {final['velocity']:.3f} km/s")
    click.echo(f"PDOP: {final['pdop']:.2f} km")
    click.echo(f"Collision warnings: {final['collision_warnings']}")
    click.echo(f"Maneuvers suggested: {len(final['avoidance_maneuvers'])}")

@cli.command()
@click.option('--ephemeris', type=click.Choice(['j2000', 'jpl', 'simple']), default='simple')
def test_ephemeris(ephemeris: str):
    """Test ephemeris generation for planets"""
    from core.ephemeris import create_planet_ephemeris
    earth = create_planet_ephemeris("earth")
    pos = earth.get_position(0.0)
    click.echo(f"Earth at J2000: {pos/1e3:.1f} km (should be ~149,600,000 km from Sun)")
    click.echo(f"That's {np.linalg.norm(pos)/1e6:.1f} million km")

@cli.command()
def demo():
    """Run interactive demo with plotting"""
    # This would launch a dashboard or plot
    click.echo("Demo mode - would open visualization window")
    # Could call visualization.dashboard.run()

if __name__ == "__main__":
    cli()