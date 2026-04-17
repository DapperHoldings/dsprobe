"""
ROS2 (Robot Operating System 2) bridge for beacon navigation.
Allows navigation system to interface with ROS2 ecosystem.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime, timezone

# ROS2 message types (would be generated from .msg files)
# These are placeholders; in reality you'd import from your package
try:
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from sensor_msgs.msg import Imu, NavSatFix
    from beacon_nav_msgs.msg import BeaconMeasurement, BeaconInfo
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    # Create dummy classes for type checking
    class Header:
        seq: int = 0
        stamp: Any = None
        frame_id: str = ""
    class PoseStamped:
        header: Header
        pose: Any
    class TwistStamped:
        header: Header
        twist: Any
    class Imu:
        header: Header
        orientation: Any
        angular_velocity: Any
        linear_acceleration: Any
    class NavSatFix:
        header: Header
        status: Any
        latitude: float = 0.0
        longitude: float = 0.0
        altitude: float = 0.0
    class BeaconMeasurement:
        header: Header
        beacon_id: str = ""
        measurement_type: str = ""
        value: List[float] = []
        uncertainty: float = 0.0
    class BeaconInfo:
        header: Header
        beacon_id: str = ""
        position: List[float] = []
        beacon_type: str = ""
    ROS2_AVAILABLE = False

from config.settings import NavConfig
from navigation.navigator import AdvancedBeaconNavigator
from core.beacon import Beacon
from core.measurement import Measurement, MeasurementKind
from sensors.imu import IMUReading

class ROS2Bridge(Node):
    """
    ROS2 node that acts as bridge between beacon navigation system and ROS2 ecosystem.
    
    Publishes:
        - /beacon_nav/odometry (nav_msgs/Odometry): Estimated state
        - /beacon_nav/beacon_list (beacon_nav_msgs/BeaconInfo[]): Available beacons
        
    Subscribes:
        - /beacon_nav/measurement (beacon_nav_msgs/BeaconMeasurement): Raw measurements
        - /imu/data (sensor_msgs/Imu): IMU data
        - /gps/fix (sensor_msgs/NavSatFix): GPS for initialization
        
    Provides services:
        - /beacon_nav/set_beacon (SetBeacon): Add/update beacon
        - /beacon_nav/reset (Empty): Reset filter
    """
    
    def __init__(self,
                 navigator: AdvancedBeaconNavigator,
                 node_name: str = 'beacon_navigation_bridge',
                 use_sim_time: bool = False):
        """
        Args:
            navigator: The navigation system instance
            node_name: ROS2 node name
            use_sim_time: Use simulation time from /clock
        """
        if not ROS2_AVAILABLE:
            raise ImportError(
                "ROS2 Python packages not found. Install ROS2 and source setup.bash"
            )
            
        super().__init__(node_name)
        
        self.navigator = navigator
        self.use_sim_time = use_sim_time
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry,
            '/beacon_nav/odometry',
            qos_profile=reliable_qos
        )
        self.beacon_list_pub = self.create_publisher(
            BeaconInfo,
            '/beacon_nav/beacon_list',
            qos_profile=reliable_qos
        )
        self.measurement_viz_pub = self.create_publisher(
            BeaconMeasurement,  # Could also be MarkerArray for RViz
            '/beacon_nav/measurement_viz',
            qos_profile=best_effort_qos
        )
        
        # Subscribers
        self.measurement_sub = self.create_subscription(
            BeaconMeasurement,
            '/beacon_nav/measurement',
            self.measurement_callback,
            qos_profile=best_effort_qos
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            qos_profile=best_effort_qos
        )
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/gps/fix',
            self.gps_callback,
            qos_profile=best_effort_qos
        )
        
        # Services
        from std_srvs.srv import Empty, SetBool
        from beacon_nav_msgs.srv import SetBeacon
        
        self.reset_service = self.create_service(
            Empty,
            '/beacon_nav/reset',
            self.reset_callback
        )
        self.set_beacon_service = self.create_service(
            SetBeacon,
            '/beacon_nav/set_beacon',
            self.set_beacon_callback
        )
        
        # Timer for publishing
        self.publish_timer = self.create_timer(0.1, self.publish_callback)  # 10 Hz
        
        # Buffers
        self.last_imu: Optional[IMUReading] = None
        self.gps_initialized = False
        
        self.get_logger().info("ROS2 Bridge initialized")
        
    def measurement_callback(self, msg: BeaconMeasurement):
        """Handle incoming measurement from ROS2"""
        # Convert ROS2 message to internal Measurement
        kind_map = {
            'RANGE': MeasurementKind.RANGE,
            'DIRECTION': MeasurementKind.DIRECTION,
            'BOTH': MeasurementKind.BOTH
        }
        
        measurement = Measurement(
            beacon_id=msg.beacon_id,
            timestamp=self._rosstamp_to_datetime(msg.header.stamp),
            kind=kind_map.get(msg.measurement_type, MeasurementKind.RANGE),
            value=np.array(msg.value) if len(msg.value) > 1 else msg.value[0],
            uncertainty=msg.uncertainty,
            quality=msg.quality if hasattr(msg, 'quality') else 1.0
        )
        
        # Process
        if measurement.beacon_id in self.navigator.beacons:
            self.navigator.process_measurements([measurement])
        else:
            self.get_logger().warn(f"Unknown beacon ID: {measurement.beacon_id}")
            
    def imu_callback(self, msg: Imu):
        """Handle IMU data"""
        # Convert ROS2 Imu to IMUReading
        # Note: ROS2 orientation is quaternion (x,y,z,w)
        # Our IMUReading expects accelerometer and gyro in body frame
        
        # For simplicity, just store linear acceleration as specific force
        # and angular velocity as gyro
        self.last_imu = IMUReading(
            timestamp=self._rosstamp_to_datetime(msg.header.stamp),
            accelerometer=np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]),
            gyroscope=np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
        )
        
    def gps_callback(self, msg: NavSatFix):
        """Handle GPS fix for initialization"""
        if not self.gps_initialized and msg.status.status >= 0:
            # Convert lat/lon/alt to ECEF (simplified)
            # Would need proper geodesy
            lat = msg.latitude * np.pi/180
            lon = msg.longitude * np.pi/180
            alt = msg.altitude
            
            # Approximate WGS84 Earth radius
            R_earth = 6378.137  # km
            x = (R_earth + alt) * np.cos(lat) * np.cos(lon)
            y = (R_earth + alt) * np.cos(lat) * np.sin(lon)
            z = (R_earth + alt) * np.sin(lat)
            
            self.navigator.initialize(
                initial_position=np.array([x, y, z]) * 1000,  # convert km to m? Our system uses km
                initial_velocity=np.zeros(3)
            )
            self.gps_initialized = True
            self.get_logger().info(f"Initialized from GPS: {x/1000:.1f}, {y/1000:.1f}, {z/1000:.1f} km")
            
    def reset_callback(self, request, response):
        """Handle reset service call"""
        self.navigator.filter.reset(np.zeros(6), np.eye(6)*1e6)
        response.success = True
        response.message = "Filter reset"
        return response
    
    def set_beacon_callback(self, request, response):
        """Handle add/update beacon service"""
        try:
            beacon = Beacon(
                id=request.beacon_id,
                name=request.name,
                beacon_type=BeaconType(request.beacon_type),
                fixed_position=np.array(request.position),
                base_uncertainty=(request.range_std, request.dir_std)
            )
            self.navigator.beacons[request.beacon_id] = beacon
            response.success = True
            response.message = f"Beacon {request.beacon_id} added/updated"
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response
    
    def publish_callback(self):
        """Periodic publication of navigation state"""
        # Get current solution
        sol = self.navigator.get_solution()
        
        # Create Odometry message
        odom = Odometry()
        odom.header.stamp = self._datetime_to_rosstamp(sol['timestamp'])
        odom.header.frame_id = 'inertial'  # or 'earth' etc.
        odom.child_frame_id = 'spacecraft'
        
        # Position
        odom.pose.pose.position.x = sol['position'][0]
        odom.pose.pose.position.y = sol['position'][1]
        odom.pose.pose.position.z = sol['position'][2]
        
        # Covariance (only position, 6x6 matrix)
        # Fill only position part from position_covariance
        for i in range(3):
            for j in range(3):
                odom.pose.covariance[i*6 + j] = sol['position_covariance'][i, j]
                
        # Velocity
        odom.twist.twist.linear.x = sol['velocity'][0]
        odom.twist.twist.linear.y = sol['velocity'][1]
        odom.twist.twist.linear.z = sol['velocity'][2]
        for i in range(3):
            for j in range(3):
                odom.twist.covariance[i*6 + j] = sol['velocity_covariance'][i, j]
                
        self.odom_pub.publish(odom)
        
        # Publish beacon list
        for beacon in self.navigator.beacons.values():
            b_msg = BeaconInfo()
            b_msg.header.stamp = odom.header.stamp
            b_msg.beacon_id = beacon.id
            b_msg.name = beacon.name
            b_msg.position = beacon.get_position(0.0).tolist()
            b_msg.beacon_type = beacon.beacon_type.value
            self.beacon_list_pub.publish(b_msg)
            
    def _rosstamp_to_datetime(self, stamp) -> datetime:
        """Convert ROS2 Time to datetime"""
        # ROS2 Time: sec + nanosec
        sec = stamp.sec
        nsec = stamp.nanosec
        total_sec = sec + nsec / 1e9
        # Assuming Unix epoch (ROS2 uses Unix time)
        return datetime.fromtimestamp(total_sec, tz=timezone.utc)
    
    def _datetime_to_rosstamp(self, dt: datetime) -> Any:
        """Convert datetime to ROS2 Time"""
        if not ROS2_AVAILABLE:
            return None
        from builtin_interfaces.msg import Time
        timestamp = Time()
        timestamp.sec = int(dt.timestamp())
        timestamp.nanosec = int((dt.timestamp() - timestamp.sec) * 1e9)
        return timestamp

def main(args=None):
    """ standalone ROS2 node entry point"""
    rclpy.init(args=args)
    
    # Create navigator (would load config, beacons, etc.)
    # This is a placeholder - user should customize
    navigator = AdvancedBeaconNavigator([], NavConfig())
    
    bridge = ROS2Bridge(navigator)
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()