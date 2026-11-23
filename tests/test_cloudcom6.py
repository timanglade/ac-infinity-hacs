"""Cloudcom 6 support, exercised with bytes captured from a real unit."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ac_infinity_ble import ACInfinityController, CallbackType, DeviceInfo
from ac_infinity_ble.const import MANUFACTURER_ID
from test_entities import entities, make_entry  # noqa: F401
from test_lifecycle import integration  # noqa: F401
from test_polling import setup_polling

# Bytes 0-5 (the unit's MAC) are replaced with a dummy address.
ADV_24 = bytes.fromhex("0123456789ab434233414114400000000000000000000006")
ADV_27 = bytes.fromhex("0123456789ab434233414114408000000000000000000006000000")
# 69.10 F, 68.90 %RH, 0.75 kPa; software revision 1.0.49.
FRAME = bytes.fromhex(
    "1eff02091400231080000001010000000000ffff00000001f0000000000300601afe02601aea0360004b"
)
# Same unit with an external probe: built-in 69.19 F, 68.69 %, 0.76 kPa;
# probe (kind 6, index 1) 68.30 F, 56.60 %, 1.03 kPa.
FRAME_PROBE = bytes.fromhex(
    "1eff02091400321080000001020000000000ffff00000001f0000000000300601b0702601ad5"
    "0360004c06010300601aae0260161c03600067"
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
    for index, value in ((29, 2), (12, 2)):
        bad = bytearray(FRAME)
        bad[index] = value
        with pytest.raises(ValueError, match="sensor"):
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


def test_external_probe_is_a_separate_reading_set(cc6):
    async def exercise():
        controller = make_controller()
        for start in range(0, len(FRAME_PROBE), 20):
            chunk = bytearray(FRAME_PROBE[start : start + 20])
            controller._notification_handler(None, chunk)
        state = controller.state
        assert (state.tmp, state.hum, state.vpd) == (
            pytest.approx(20.66),
            pytest.approx(68.69),
            pytest.approx(0.76),
        )
        (probe,) = cc6.probes(state)
        assert (probe.index, probe.kind) == (1, 6)
        assert (probe.tmp, probe.hum, probe.vpd) == (
            pytest.approx(20.17),
            pytest.approx(56.6),
            pytest.approx(1.03),
        )
        # Advertisements keep probes; unplugging the probe drops it.
        controller.set_ble_device_and_advertisement_data(
            SimpleNamespace(address="AA:BB"),
            SimpleNamespace(manufacturer_data={MANUFACTURER_ID: ADV_27}),
        )
        assert cc6.probes(controller.state) == (probe,)
        controller._notification_handler(None, bytearray(FRAME))
        assert cc6.probes(controller.state) == ()
        assert controller.temperature == pytest.approx(20.61)

    asyncio.run(exercise())


def test_probe_entities_appear_with_the_probe(cc6, entities):  # noqa: F811
    sensor = entities["sensor"]
    entry, listeners = make_entry(64)
    device = entry.runtime_data.device
    device.state = DeviceInfo(64, "C-CB3AA", 20)
    added = []
    asyncio.run(sensor.async_setup_entry(None, entry, lambda e: added.extend(e)))
    assert len(added) == 3  # built-in temperature, humidity, VPD only

    device.state = cc6.parse_telemetry(FRAME_PROBE, device.state)
    for listener in listeners:
        listener()
    for listener in listeners:  # a second update must not duplicate entities
        listener()
    probe_entities = added[3:]
    assert [e.unique_id for e in probe_entities] == [
        "00:11:22:33:44:55_probe1_tmp",
        "00:11:22:33:44:55_probe1_hum",
        "00:11:22:33:44:55_probe1_vpd",
    ]
    assert [e.name for e in probe_entities] == [
        "Test Probe 1 Temperature",
        "Test Probe 1 Humidity",
        "Test Probe 1 VPD",
    ]
    assert [e._attr_native_value for e in probe_entities] == [
        pytest.approx(20.17),
        pytest.approx(56.6),
        pytest.approx(1.03),
    ]
    assert all(e.available for e in probe_entities)

    device.state = cc6.parse_telemetry(FRAME, device.state)
    for entity in probe_entities:
        entity._async_update_attrs()
        assert not entity.available
        assert entity._attr_native_value is None
