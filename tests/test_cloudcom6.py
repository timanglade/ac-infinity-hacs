"""Cloudcom 6 support, exercised with bytes captured from a real unit."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ac_infinity_ble import ACInfinityController, CallbackType, DeviceInfo
from ac_infinity_ble.const import MANUFACTURER_ID
from test_lifecycle import integration  # noqa: F401
from test_polling import setup_polling

# Bytes 0-5 (the unit's MAC) are replaced with a dummy address.
ADV_24 = bytes.fromhex("0123456789ab434233414114400000000000000000000006")
ADV_27 = bytes.fromhex("0123456789ab434233414114408000000000000000000006000000")
# 69.10 F, 68.90 %RH, 0.75 kPa; software revision 1.0.49.
FRAME = bytes.fromhex(
    "1eff02091400231080000001010000000000ffff00000001f0000000000300601afe02601aea0360004b"
)


@pytest.fixture
def cc6(integration):  # noqa: F811
    return importlib.import_module("custom_components.ac_infinity.cloudcom6")


def make_controller(*, fire=None):
    controller = ACInfinityController(
        SimpleNamespace(address="AA:BB"), DeviceInfo(64, "C-CB3AA", 20)
    )
    if fire is not None:
        controller.register_callback(fire)
    return controller


@pytest.mark.parametrize("adv", [ADV_24, ADV_27])
def test_advertisement_is_identity_only_read_only_sensor(cc6, adv):
    state = cc6.parse_manufacturer_data(adv)
    assert (state.type, state.version, state.name) == (64, 20, "C-CB3AA")
    assert state.display_name == "Cloudcom 6 (C-CB3AA)"
    assert state.tmp is state.hum is state.vpd is None
    assert not state.profile.writable
    assert state.outputs == ()
    assert cc6.streams_telemetry(state)


def test_other_models_keep_library_behavior(cc6):
    assert not cc6.streams_telemetry(DeviceInfo(14, "Test", 4))
    assert cc6.streams_telemetry(DeviceInfo(11, "Test", 7))
    assert DeviceInfo(51, "Test", 20).profile.model_name == "Controller AI 2"
    with pytest.raises(ValueError):
        cc6.parse_manufacturer_data(ADV_27[:12] + bytes([63]) + ADV_27[13:])


def test_fragmented_telemetry_updates_readings(cc6):
    async def exercise():
        fire = Mock()
        controller = make_controller(fire=fire)
        for start in range(0, len(FRAME), 20):
            controller._notification_handler(None, bytearray(FRAME[start : start + 20]))
        assert controller.temperature == pytest.approx(20.61)
        assert controller.humidity == pytest.approx(68.9)
        assert controller.vpd == pytest.approx(0.75)
        fire.assert_called_once()
        assert fire.call_args.args[1] == CallbackType.NOTIFICATION
        return controller

    controller = asyncio.run(exercise())
    assert controller.state.ports == {}


def test_advertisements_do_not_erase_telemetry(cc6):
    async def exercise():
        controller = make_controller()
        controller._notification_handler(None, bytearray(FRAME))
        controller.set_ble_device_and_advertisement_data(
            SimpleNamespace(address="AA:BB"),
            SimpleNamespace(manufacturer_data={MANUFACTURER_ID: ADV_27}),
        )
        assert controller.humidity == pytest.approx(68.9)

    asyncio.run(exercise())


def test_missing_probe_reading_and_bad_counts(cc6):
    state = DeviceInfo(64, "C-CB3AA", 20)
    missing = FRAME[:-4] + bytes.fromhex("03608000")
    assert cc6.parse_telemetry(missing, state).vpd is None
    bad = bytearray(FRAME)
    bad[29] = 2
    with pytest.raises(ValueError, match="sensor count"):
        cc6.parse_telemetry(bad, state)


def test_refresh_connects_and_waits_for_a_frame(cc6):
    async def exercise():
        controller = make_controller()
        controller._ensure_connected = AsyncMock(
            side_effect=lambda: controller._notification_handler(None, bytearray(FRAME))
        )
        await controller.refresh_telemetry()
        controller._ensure_connected.assert_awaited_once()
        assert controller.temperature == pytest.approx(20.61)

    asyncio.run(exercise())


def test_coordinator_polls_cloudcom6_for_telemetry(integration, monkeypatch):  # noqa: F811
    async def exercise():
        coordinator, device, _, service = setup_polling(integration, monkeypatch)
        device.state = DeviceInfo(64, "C-CB3AA", 20)
        assert coordinator._needs_poll(service, None)
        await coordinator._async_update(service)
        device.refresh_telemetry.assert_awaited_once()
        device.update.assert_awaited_once_with(0)  # no-op for read-only profiles

    asyncio.run(exercise())
