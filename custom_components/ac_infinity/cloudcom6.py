"""Local Cloudcom 6 support layered on the pinned ac-infinity-ble 1.0.0.

The Cloudcom 6 advertises device type 64 at protocol version 20, which
ac-infinity-ble 1.0.0 rejects as unsupported. Unlike older Cloudcom models its
advertisement carries no readings: the temperature/humidity bytes stay zero.
Readings and port state arrive only as GATT telemetry, which it streams about
once a second as soon as a client subscribes to the notify characteristic.

Telemetry is a ``1e ff 02 09`` frame with a two-byte length at offset 5 (like
the AI 2 family). Byte 11 is the number of port records and byte 12 the number
of sensor groups. Port records start at byte 15 and are 12 bytes each:
``id, raw_type(2), load_id(2), kind, level/fault, ...`` using the AI family's
field meanings; port 0 is the unit itself. The configured mode only comes from
GET (tag 16); see ``_decode_port``. Sensor groups follow: each
is ``kind, index, count`` then ``count`` four-byte records ``id, flags, int16``.
Group index 0 is the built-in sensor; an external probe adds a group (seen as
kind 6, index 1). Record IDs and accuracy flags follow the library's AI
conventions (0/1 temperature F/C, 2 humidity, 3 VPD). Captured from one unit,
software revision 1.0.49, with a fan on port 1 and a probe attached:

    1eff0209 1400 3e 1080000002 02 0100
    00 0000 ffff 00 00 00 01f00000         port 0 (unit, no load)
    01 13cd ffff 00 00 02 01fc4803         port 1: 5069 ohm -> fan, level 0
    0000 03 00611b58 02621acc 0366004f     built-in 70.00 F, 68.60 %, 0.79 kPa
    0601 03 00601ac2 02601630 03600067     probe    68.50 F, 56.80 %, 1.03 kPa

Commands use the AI 2 parameter codec (addressed ports, two-byte parameter
lengths, manual speed in the upper nibble of tag 18). Replies carry the
protocol version in the high byte of the command word (``14 01`` for a GET),
which the library's ``00 01`` match rejects; ``Cloudcom6Protocol`` expects the
unit's advertised version there instead.

This belongs upstream in ac-infinity-ble. Until then, ``install()`` patches the
library narrowly: every hook defers to the original unless the type is 64.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from ac_infinity_ble import device as _device
from ac_infinity_ble import models as _models
from ac_infinity_ble import protocol as _protocol
from ac_infinity_ble.capabilities import load_kind, recognized_kind
from ac_infinity_ble.const import MANUFACTURER_ID
from ac_infinity_ble.exceptions import FrameValidationError
from ac_infinity_ble.models import DeviceInfo, PortState
from ac_infinity_ble.protocol import Protocol
from ac_infinity_ble.routing import ProtocolProfile, TelemetryLayout
from ac_infinity_ble.telemetry import frame_length

CLOUDCOM6_TYPE = 64
MODEL_NAME = "Cloudcom 6"
TELEMETRY_HEADER = b"\x1e\xff\x02\x09"
PORTS_OFFSET = 15
PORT_RECORD_SIZE = 12
SENSOR_RECORD_SIZE = 4
_READING_KEYS = {0: "tmp", 1: "tmp", 2: "hum", 3: "vpd"}

PROFILE = ProtocolProfile(
    model_name=MODEL_NAME,
    # Matches the library's Controller AI 2 command profile, minus the
    # aggregate fan: port 0 is the unit itself, loads are on numbered ports.
    telemetry=TelemetryLayout.H4,
    addressed=True,
    parameter_length_size=2,
    writable=True,
    aggregate_fan=False,
    packed_manual_level=True,
)


@dataclass(frozen=True)
class ProbeReading:
    """One external probe's readings, keyed by its group index."""

    index: int
    kind: int
    tmp: float | None = None
    hum: float | None = None
    vpd: float | None = None


@dataclass(frozen=True)
class Cloudcom6Info(DeviceInfo):
    """DeviceInfo plus external probes; library replace() calls preserve it."""

    probes: tuple[ProbeReading, ...] = ()


class Cloudcom6Protocol(Protocol):
    """Expect replies whose command word carries the protocol version."""

    def __init__(self, version: int) -> None:
        super().__init__()
        self.version = version

    def parse_response(
        self,
        data: bytes | bytearray,
        expected_sequence: int | None = None,
        expected_command: int | None = None,
    ) -> bytes:
        payload = super().parse_response(data, expected_sequence)
        if expected_command is not None and data[8:10] != bytes(
            (self.version, expected_command)
        ):
            raise FrameValidationError("Unexpected response command")
        return payload


def is_cloudcom6(state: DeviceInfo) -> bool:
    return state.type == CLOUDCOM6_TYPE


def probes(state: DeviceInfo) -> tuple[ProbeReading, ...]:
    """External probes present in the latest telemetry (empty for other models)."""
    return getattr(state, "probes", ())


def parse_advertisement(data: bytes) -> DeviceInfo:
    """Identity only; the Cloudcom 6 advertisement has no live readings."""
    if len(data) not in (24, 27) or data[12] != CLOUDCOM6_TYPE:
        raise ValueError("Not a Cloudcom 6 advertisement")
    try:
        name = data[6:11].decode("ascii")
    except UnicodeDecodeError as ex:
        raise ValueError("Invalid AC Infinity device name") from ex
    return DeviceInfo(type=CLOUDCOM6_TYPE, version=data[11], name=f"C-{name}")


def parse_manufacturer_data(data: bytes) -> DeviceInfo:
    """Drop-in for the library parser that also recognizes the Cloudcom 6."""
    if len(data) > 12 and data[12] == CLOUDCOM6_TYPE:
        return parse_advertisement(data)
    return _protocol.parse_manufacturer_data(data)


def parse_telemetry(data: bytes | bytearray, state: DeviceInfo) -> DeviceInfo:
    # The library's frame_length already covers this two-byte length header.
    if data[:4] != TELEMETRY_HEADER or len(data) != frame_length(data, state):
        raise ValueError("Invalid Cloudcom 6 telemetry framing")
    if len(data) < PORTS_OFFSET:
        raise ValueError("Truncated Cloudcom 6 telemetry")
    offset = PORTS_OFFSET
    ports: dict[int, PortState] = {}
    for _ in range(data[11]):
        record = data[offset : offset + PORT_RECORD_SIZE]
        if len(record) != PORT_RECORD_SIZE or record[0] in ports:
            raise ValueError("Invalid Cloudcom 6 port records")
        ports[record[0]] = _decode_port(record, state.ports.get(record[0]))
        offset += PORT_RECORD_SIZE
    builtin = {"tmp": None, "hum": None, "vpd": None}
    found: list[ProbeReading] = []
    indexes: set[int] = set()
    for _ in range(data[12]):
        if offset + 3 > len(data):
            raise ValueError("Invalid Cloudcom 6 sensor groups")
        kind, index, count = data[offset : offset + 3]
        if index in indexes:
            raise ValueError("Duplicate Cloudcom 6 sensor group")
        indexes.add(index)
        offset += 3
        end = offset + count * SENSOR_RECORD_SIZE
        if end > len(data):
            raise ValueError("Invalid Cloudcom 6 sensor count")
        readings = _decode_records(data[offset:end])
        offset = end
        if index == 0:
            builtin.update(readings)
        else:
            found.append(ProbeReading(index, kind, **readings))
    if offset != len(data):
        raise ValueError("Invalid Cloudcom 6 sensor groups")
    if not isinstance(state, Cloudcom6Info):
        state = Cloudcom6Info(**{f.name: getattr(state, f.name) for f in fields(state)})
    return replace(state, **builtin, probes=tuple(found), ports=ports)


def _decode_port(record: bytes | bytearray, previous: PortState | None) -> PortState:
    raw_type = int.from_bytes(record[1:3], "big")
    load_id = int.from_bytes(record[3:5], "big")
    kind = load_kind(record[5]) or recognized_kind(raw_type, load_id)
    # Byte 7 is not the configured mode: it read 02 for a fan both in Cycle and
    # in Off. Keep the mode last read by GET (or confirmed by SET) for this load.
    same_load = previous is not None and (previous.kind, previous.raw_type) == (
        kind,
        raw_type,
    )
    return PortState(
        record[0],
        connected=record[1] != 255,
        raw_type=raw_type,
        kind=kind,
        level=(record[6] >> 2) & 15,
        mode=previous.mode if same_load else None,
        fault=bool(record[6] & 1),
    )


def _decode_records(block: bytes | bytearray) -> dict[str, float | None]:
    readings: dict[str, float | None] = {}
    for start in range(0, len(block), SENSOR_RECORD_SIZE):
        sensor_id, flags = block[start], block[start + 1]
        raw = int.from_bytes(block[start + 2 : start + 4], "big", signed=True)
        value = None if raw == -32768 else raw * (10, 1, 0.1, 0.01)[(flags >> 5) & 3]
        if sensor_id == 0 and value is not None:
            value = (value - 32) / 1.8
        if (key := _READING_KEYS.get(sensor_id)) is not None:
            readings[key] = None if value is None else round(value, 2)
    return readings


def install() -> None:
    """Patch ac-infinity-ble once; non-Cloudcom 6 devices see original behavior."""
    if getattr(_device, "_cloudcom6_installed", False):
        return

    original_profile = _models.resolve_profile
    original_parse_telemetry = _device.parse_telemetry
    controller = _device.ACInfinityController
    original_init = controller.__init__
    original_set_adv = controller.set_ble_device_and_advertisement_data

    def resolve_profile(device_type: int, version: int) -> ProtocolProfile:
        if device_type == CLOUDCOM6_TYPE:
            return PROFILE
        return original_profile(device_type, version)

    def parse(data: bytes | bytearray, state: DeviceInfo) -> DeviceInfo:
        if is_cloudcom6(state):
            return parse_telemetry(data, state)
        return original_parse_telemetry(data, state)

    def __init__(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        if is_cloudcom6(self._state):
            self._protocol = Cloudcom6Protocol(self._state.version)

    def set_ble_device_and_advertisement_data(self, ble_device, advertisement_data):
        if not is_cloudcom6(self._state):
            return original_set_adv(self, ble_device, advertisement_data)
        self._ble_device = ble_device
        self._advertisement_data = advertisement_data
        info = parse_advertisement(
            advertisement_data.manufacturer_data[MANUFACTURER_ID]
        )
        # Keep telemetry readings and ports; advertisements carry identity only.
        self._state = replace(self._state, name=info.name, version=info.version)
        self._protocol.version = info.version
        self._fire_callbacks(_device.CallbackType.ADVERTISEMENT)

    _models.resolve_profile = resolve_profile
    _device.parse_manufacturer_data = parse_manufacturer_data  # controller __init__
    _device.parse_telemetry = parse
    controller.__init__ = __init__
    controller.set_ble_device_and_advertisement_data = (
        set_ble_device_and_advertisement_data
    )
    _device._cloudcom6_installed = True
