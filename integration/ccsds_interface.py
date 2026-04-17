"""
CCSDS (Consultative Committee for Space Data Systems) telemetry/telecommand interface.
Implements standard packet formats for spacecraft communication.
"""

import struct
import crcmod
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import io

# CCSDS constants
CCSDS_PRIMARY_HEADER_FORMAT = '>HHBB'  # big-endian: version, type, sec_header, packet_id, seq_ctrl, packet_len
CCSDS_SECONDARY_HEADER_FORMAT = '>HHII'  # for time codes

# Packet IDs (example)
PACKET_ID_NAVIGATION = 0x100
PACKET_ID_MEASUREMENT = 0x101
PACKET_ID_COMMAND = 0x200

# CRC-16-CCITT polynomial (used in CCSDS)
crc16 = crcmod.predefined.Crc('crc-ccitt-false')

@dataclass
class CCSDSPacket:
    """
    Generic CCSDS packet.
    Format:
        Primary Header (6 bytes)
        Secondary Header (optional, 4 or 8 bytes)
        Data Field (variable)
        Trailer (optional, 2 or 4 bytes CRC)
    """
    packet_id: int  # 16-bit: version(3) type(1) sec_header(1) attributes(3) group(2) source(6)
    sequence_count: int  # 14-bit sequence count
    packet_data: bytes
    timestamp: Optional[datetime] = None
    is_secondary_header: bool = False
    priority: int = 1  # 1-3, 1=highest
    
    # CRC will be computed automatically
    crc: Optional[int] = None
    
    def pack(self) -> bytes:
        """Pack into binary CCSDS packet"""
        # Build primary header
        version = 0  # CCSDS version 0 for standard
        packet_type = 0  # 0=telemetry, 1=telecommand
        secondary_header_flag = 1 if self.is_secondary_header else 0
        apid = 0x000  # Application ID (subset of packet_id)
        grouping = 0  # 0=no grouping, 1=first, 2=middle, 3=last
        
        # Construct packet_id field (16 bits)
        # Bits: [15:14]=version, [13]=type, [12]=sec_hdr, [11:8]=apid, [7:6]=group, [5:0]=??
        # Simplified: just use 16-bit packet_id as given
        packet_id_field = self.packet_id & 0xFFFF
        
        # Sequence count (14 bits) and packet seq control (2 bits)
        seq_count = self.sequence_count & 0x3FFF
        seq_flags = 0  # 00 = no grouping
        
        packet_header = struct.pack(
            '>HH',
            packet_id_field,
            (seq_count << 2) | seq_flags
        )
        
        # If secondary header, add time
        if self.is_secondary_header and self.timestamp:
            # Convert to CUC ( CCSDS Unsegmented Time Code )
            # 4-byte seconds + 2-byte subseconds
            epoch = datetime(1958, 1, 1, tzinfo=timezone.utc)  # GPS epoch? Actually CCSDS uses 1958
            # For simplicity, use seconds since J2000
            j2000 = datetime(2000,1,1,12,0,0, tzinfo=timezone.utc)
            seconds = int((self.timestamp - j2000).total_seconds())
            subseconds = int(((self.timestamp - j2000).total_seconds() - seconds) * 2**16)
            
            sec_header = struct.pack('>HIH', 0, seconds, subseconds)
            packet = packet_header + sec_header + self.packet_data
        else:
            packet = packet_header + self.packet_data
            
        # Compute CRC over packet data (or whole packet? Typically data field only)
        crc_value = self._compute_crc(packet)
        packet += struct.pack('>H', crc_value)
        
        # Total length
        total_len = len(packet) - 1  # Length field is packet length - 1
        # Replace length in header (bytes 4-5)
        packet = packet[:4] + struct.pack('>H', total_len) + packet[6:]
        
        return packet
    
    @staticmethod
    def _compute_crc(data: bytes) -> int:
        """Compute CCSDS CRC-16"""
        crc16.reset()
        crc16.update(data)
        return crc16.crcValue
    
    @classmethod
    def unpack(cls, data: bytes) -> 'CCSDSPacket':
        """Parse binary CCSDS packet"""
        if len(data) < 6:
            raise ValueError("Packet too short for primary header")
            
        # Primary header
        packet_id, seq_control = struct.unpack('>HH', data[:4])
        packet_len = struct.unpack('>H', data[4:6])[0] + 1  # stored as len-1
        
        if len(data) < packet_len:
            raise ValueError(f"Packet length mismatch: header says {packet_len}, got {len(data)}")
            
        # Extract fields
        seq_count = seq_control >> 2
        seq_flags = seq_control & 0x3
        
        # Determine if secondary header
        sec_hdr_flag = (packet_id >> 12) & 0x1
        is_secondary = bool(sec_hdr_flag)
        
        # Extract data field (between header and CRC)
        if is_secondary:
            # Secondary header is 6 or 8 bytes? Using 6 for CUC (4+2)
            header_len = 6 + 6  # primary + secondary
        else:
            header_len = 6
            
        trailer_len = 2  # CRC
        data_field = data[header_len:-trailer_len]
        
        # CRC check
        received_crc = struct.unpack('>H', data[-trailer_len:])[0]
        computed_crc = cls._compute_crc(data[:-trailer_len])
        if received_crc != computed_crc:
            raise ValueError(f"CRC mismatch: received {received_crc:04X}, computed {computed_crc:04X}")
            
        # Timestamp from secondary header?
        timestamp = None
        if is_secondary:
            # Parse time from secondary header
            # Assuming 4-byte seconds + 2-byte subseconds (CUC)
            sec_data = data[6:12]  # after primary
            # Skip 2-byte purpose field, then 4-byte seconds, 2-byte subseconds
            _, seconds, subseconds = struct.unpack('>HIH', sec_data)
            j2000 = datetime(2000,1,1,12,0,0, tzinfo=timezone.utc)
            timestamp = j2000 + timedelta(seconds=seconds, microseconds=subseconds/2**16*1e6)
            
        return cls(
            packet_id=packet_id,
            sequence_count=seq_count,
            packet_data=data_field,
            timestamp=timestamp,
            is_secondary_header=is_secondary,
            crc=received_crc
        )

class NavigationToCCSDS:
    """
    Convert navigation system data to CCSDS packets.
    """
    
    def __init__(self, 
                 apid: int = 0x01,
                 use_secondary_header: bool = True):
        """
        Args:
            apid: Application ID (5 bits? Actually 8 bits in CCSDS, but total 16-bit packet_id)
            use_secondary_header: Include timestamp in secondary header
        """
        self.apid = apid & 0x1FF  # 9 bits for APID in some standards, but we'll use 8
        self.use_secondary_header = use_secondary_header
        self.sequence_counter = 0
        
    def pack_navigation_state(self, 
                            position: np.ndarray,
                            velocity: np.ndarray,
                            pdop: float,
                            timestamp: datetime) -> bytes:
        """
        Pack navigation state into CCSDS packet.
        
        Format:
            - Position (3 floats, km)
            - Velocity (3 floats, km/s)
            - PDOP (float)
            - Status flags (1 byte)
        """
        # Data: 3*4 + 3*4 + 4 + 1 = 29 bytes (padded to 4 bytes? We'll pack as floats)
        # Using little-endian float32 for payload, but header is big-endian
        payload = struct.pack(
            '<3f3ffB',  # little-endian for payload convenience
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            pdop,
            0  # status flags
        )
        
        packet = CCSDSPacket(
            packet_id=(0x0 << 13) | (0 << 12) | (self.apid << 6) | PACKET_ID_NAVIGATION & 0x3F,
            sequence_count=self.sequence_counter,
            packet_data=payload,
            timestamp=timestamp,
            is_secondary_header=self.use_secondary_header
        )
        
        self.sequence_counter = (self.sequence_counter + 1) & 0x3FFF
        return packet.pack()
    
    def pack_measurement(self,
                        measurement: Measurement,
                        beacon_id: str,
                        timestamp: datetime) -> bytes:
        """
        Pack measurement into CCSDS packet.
        
        Format:
            - Beacon ID (string up to 16 bytes, null-terminated)
            - Measurement type (1 byte enum)
            - Value count (1 byte)
            - Values (N floats)
            - Uncertainty (float)
        """
        beacon_bytes = beacon_id.encode('utf-8')[:15]  # max 15 + null
        beacon_padded = beacon_bytes.ljust(16, b'\x00')
        
        kind_map = {
            MeasurementKind.RANGE: 1,
            MeasurementKind.DIRECTION: 2,
            MeasurementKind.RANGE_RATE: 3,
            MeasurementKind.BOTH: 4
        }
        kind_byte = kind_map.get(measurement.kind, 0)
        
        val_array = measurement.as_vector()
        n_values = len(val_array)
        
        # Pack: beacon(16), kind(1), n(1), values(n*4), uncertainty(4)
        payload = struct.pack(
            f'<16sBB{n_values}fBf',
            beacon_padded,
            kind_byte,
            n_values,
            *val_array,
            measurement.uncertainty if isinstance(measurement.uncertainty, float) else 0.0,
            int(measurement.quality * 255)  # quality as byte
        )
        
        packet = CCSDSPacket(
            packet_id=PACKET_ID_MEASUREMENT,
            sequence_count=self.sequence_counter,
            packet_data=payload,
            timestamp=timestamp,
            is_secondary_header=self.use_secondary_header
        )
        
        self.sequence_counter = (self.sequence_counter + 1) & 0x3FFF
        return packet.pack()

class CCSDSToNavigation:
    """
    Parse CCSDS packets into navigation system inputs.
    """
    
    def __init__(self):
        self.packet_buffer = bytearray()
        
    def parse_packet(self, packet_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse a CCSDS packet.
        
        Returns:
            Dictionary with parsed data, or None if not a navigation packet
        """
        try:
            pkt = CCSDSPacket.unpack(packet_bytes)
        except ValueError as e:
            # Not a valid packet or CRC error
            return None
            
        # Check packet ID
        if pkt.packet_id == PACKET_ID_NAVIGATION:
            return self._parse_navigation_packet(pkt)
        elif pkt.packet_id == PACKET_ID_MEASUREMENT:
            return self._parse_measurement_packet(pkt)
        else:
            return None
            
    def _parse_navigation_packet(self, pkt: CCSDSPacket) -> Dict[str, Any]:
        """Extract navigation state from packet"""
        # Expect 3f3ffB format
        if len(pkt.packet_data) >= 29:
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, pdop, status = struct.unpack('<3f3ffB', pkt.packet_data[:29])
            return {
                'type': 'navigation',
                'position': np.array([pos_x, pos_y, pos_z]),
                'velocity': np.array([vel_x, vel_y, vel_z]),
                'pdop': pdop,
                'status': status,
                'timestamp': pkt.timestamp,
                'sequence': pkt.sequence_count
            }
        return None
        
    def _parse_measurement_packet(self, pkt: CCSDSPacket) -> Dict[str, Any]:
        """Extract measurement from packet"""
        # Unpack: 16sBB{n}fBf
        # We don't know n; need to parse dynamically
        if len(pkt.packet_data) < 18:  # min: 16+1+1
            return None
            
        beacon_bytes, kind_byte, n_values = struct.unpack('<16sBB', pkt.packet_data[:18])
        beacon_id = beacon_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')
        
        expected_len = 18 + n_values*4 + 4 + 1
        if len(pkt.packet_data) < expected_len:
            return None
            
        values = struct.unpack(f'<{n_values}f', pkt.packet_data[18:18+n_values*4])
        uncertainty, quality_byte = struct.unpack('<fB', pkt.packet_data[18+n_values*4:18+n_values*4+5])
        
        kind_map = {1: MeasurementKind.RANGE, 2: MeasurementKind.DIRECTION, 
                    3: MeasurementKind.RANGE_RATE, 4: MeasurementKind.BOTH}
        kind = kind_map.get(kind_byte, MeasurementKind.RANGE)
        
        val = values[0] if n_values == 1 else np.array(values)
        quality = quality_byte / 255.0
        
        return {
            'type': 'measurement',
            'beacon_id': beacon_id,
            'kind': kind,
            'value': val,
            'uncertainty': uncertainty,
            'quality': quality,
            'timestamp': pkt.timestamp,
            'sequence': pkt.sequence_count
        }

def ccsds_stream_to_file(packets: List[bytes], filepath: str):
    """Write list of CCSDS packets to binary file"""
    with open(filepath, 'wb') as f:
        for pkt in packets:
            f.write(pkt)
            
def ccsds_file_to_stream(filepath: str) -> List[bytes]:
    """Read CCSDS packets from binary file"""
    packets = []
    with open(filepath, 'rb') as f:
        while True:
            # Read at least 6 bytes for primary header to get length
            header = f.read(6)
            if len(header) < 6:
                break
            packet_len = struct.unpack('>H', header[4:6])[0] + 1
            # Read rest of packet
            remaining = packet_len - 6
            data = f.read(remaining)
            if len(data) < remaining:
                break
            packets.append(header + data)
    return packets