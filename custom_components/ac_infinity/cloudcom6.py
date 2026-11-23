"""Local Cloudcom 6 support layered on the pinned ac-infinity-ble 1.0.0.

The Cloudcom 6 advertises device type 64 at protocol version 20, which
ac-infinity-ble 1.0.0 rejects as unsupported. It is a read-only sensor, but
unlike older Cloudcom models its advertisement carries no readings: the
temperature/humidity bytes stay zero. Readings only arrive as GATT telemetry,
which it streams about once a second as soon as a client subscribes to the
notify characteristic; no command is needed.

Telemetry is a ``1e ff 02 09`` frame with a two-byte length at offset 5 (like
the AI 2 family). Byte 12 is the number of sensor groups, which start at byte
27. Each group is ``kind, index, count`` followed by ``count`` four-byte records
``id, flags, int16``. Index 0 is the unit's built-in sensor; an external probe
adds a group (seen as kind 6, index 1). Record IDs and accuracy flags use the
library's existing AI conventions (0/1 temperature F/C, 2 humidity, 3 VPD).
Captured from one unit, software revision 1.0.49, with a probe attached:

    1eff0209 1400 32 1080000001 02 0000 00000000ffff00000001f00000
    0000 03 00601b07 02601ad5 0360004c  ->  69.19 F, 68.69 %, 0.76 kPa
    0601 03 00601aae 0260161c 03600067  ->  68.30 F, 56.60 %, 1.03 kPa

This belongs upstream in ac-infinity-ble. Until then, ``install()`` patches the
library narrowly: every hook defers to the original unless the type is 64.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields, replace

from ac_infinity_ble import device as _device
from ac_infinity_ble import models as _models
from ac_infinity_ble import protocol as _protocol
from ac_infinity_ble.const import MANUFACTURER_ID
from ac_infinity_ble.models import DeviceInfo
from ac_infinity_ble.routing import ProtocolProfile, TelemetryLayout

CLOUDCOM6_TYPE = 64
MODEL_NAME = "Cloudcom 6"
TELEMETRY_HEADER = b"\x1e\xff\x02\x09"
SENSOR_RECORD_SIZE = 4
_GROUPS_OFFSET = 27
_TELEMETRY_TIMEOUT = 5
_READING_KEYS = {0: "tmp", 1: "tmp", 2: "hum", 3: "vpd"}

PROFILE = ProtocolProfile(
    model_name=MODEL_NAME,
    # Closest existing layout: read-only, no ports, no output entities.
    telemetry=TelemetryLayout.SENSOR,
    addressed=False,
    parameter_length_size=2,
    writable=False,
    aggregate_fan=False,
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


def is_cloudcom6(state: DeviceInfo) -> bool:
    return state.type == CLOUDCOM6_TYPE


def probes(state: DeviceInfo) -> tuple[ProbeReading, ...]:
    """External probes present in the latest telemetry (empty for other models)."""
    return getattr(state, "probes", ())


def streams_telemetry(state: DeviceInfo) -> bool:
    """Whether readings require a GATT connection instead of advertisements."""
    return state.profile.writable or is_cloudcom6(state)


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


def telemetry_frame_length(data: bytes | bytearray) -> int | None:
    if len(data) < 7:
        return None
    return 7 + int.from_bytes(data[5:7], "big")


def parse_telemetry(data: bytes | bytearray, state: DeviceInfo) -> DeviceInfo:
    if data[:4] != TELEMETRY_HEADER or len(data) != telemetry_frame_length(data):
        raise ValueError("Invalid Cloudcom 6 telemetry framing")
    if len(data) < _GROUPS_OFFSET:
        raise ValueError("Truncated Cloudcom 6 telemetry")
    builtin = {"tmp": None, "hum": None, "vpd": None}
    found: list[ProbeReading] = []
    offset = _GROUPS_OFFSET
    for _ in range(data[12]):
        if offset + 3 > len(data):
            raise ValueError("Invalid Cloudcom 6 sensor groups")
        kind, index, count = data[offset : offset + 3]
        offset += 3
        end = offset + count * SENSOR_RECORD_SIZE
        if end > len(data):
            raise ValueError("Invalid Cloudcom 6 sensor count")
        readings = _decode_records(data[offset:end])
        offset = end
        if index == 0:
            builtin.update(readings)
        elif any(p.index == index for p in found):
            raise ValueError("Duplicate Cloudcom 6 probe")
        else:
            found.append(ProbeReading(index, kind, **readings))
    if offset != len(data):
        raise ValueError("Invalid Cloudcom 6 sensor groups")
    if not isinstance(state, Cloudcom6Info):
        state = Cloudcom6Info(**{f.name: getattr(state, f.name) for f in fields(state)})
    return replace(state, **builtin, probes=tuple(found))


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
    original_frame_length = _device.telemetry_frame_length
    original_parse_telemetry = _device.parse_telemetry
    controller = _device.ACInfinityController
    original_set_adv = controller.set_ble_device_and_advertisement_data
    original_refresh = controller.refresh_telemetry

    def resolve_profile(device_type: int, version: int) -> ProtocolProfile:
        if device_type == CLOUDCOM6_TYPE:
            return PROFILE
        return original_profile(device_type, version)

    def frame_length(data: bytes | bytearray, state: DeviceInfo) -> int | None:
        if is_cloudcom6(state):
            return telemetry_frame_length(data)
        return original_frame_length(data, state)

    def parse(data: bytes | bytearray, state: DeviceInfo) -> DeviceInfo:
        if is_cloudcom6(state):
            return parse_telemetry(data, state)
        return original_parse_telemetry(data, state)

    def set_ble_device_and_advertisement_data(self, ble_device, advertisement_data):
        if not is_cloudcom6(self._state):
            return original_set_adv(self, ble_device, advertisement_data)
        self._ble_device = ble_device
        self._advertisement_data = advertisement_data
        info = parse_advertisement(
            advertisement_data.manufacturer_data[MANUFACTURER_ID]
        )
        # Keep telemetry readings; only identity comes from advertisements.
        self._state = replace(self._state, name=info.name, version=info.version)
        self._fire_callbacks(_device.CallbackType.ADVERTISEMENT)

    async def refresh_telemetry(self) -> None:
        if not is_cloudcom6(self._state):
            return await original_refresh(self)
        # Same as the library's writable-device path, which 1.0.0 gates on
        # profile.writable: connect, subscribe, wait for one frame.
        async with self._operation_lock:
            self._telemetry_event.clear()
            try:
                await self._ensure_connected()
                async with asyncio.timeout(_TELEMETRY_TIMEOUT):
                    await self._telemetry_event.wait()
            except BaseException:
                await self._execute_disconnect()
                raise
            finally:
                if self._client and self._client.is_connected:
                    self._reset_disconnect_timer()

    _models.resolve_profile = resolve_profile
    _device.parse_manufacturer_data = parse_manufacturer_data  # controller __init__
    _device.telemetry_frame_length = frame_length
    _device.parse_telemetry = parse
    controller.set_ble_device_and_advertisement_data = (
        set_ble_device_and_advertisement_data
    )
    controller.refresh_telemetry = refresh_telemetry
    _device._cloudcom6_installed = True
