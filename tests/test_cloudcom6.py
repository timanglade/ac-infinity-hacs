"""Cloudcom 6 support, exercised with bytes captured from a real unit."""

import asyncio
import importlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ac_infinity_ble import ACInfinityController, CallbackType, DeviceInfo
from ac_infinity_ble.const import MANUFACTURER_ID
from ac_infinity_ble.protocol import Protocol
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
# Fan on port 1 (5069 ohm, stopped) plus probe: built-in 70.00 F, 68.60 %,
# 0.79 kPa; probe 68.50 F, 56.80 %, 1.03 kPa.
FRAME_FAN = bytes.fromhex(
    "1eff020914003e1080000002020100000000ffff00000001f000000113cdffff00000201fc48"
    "0300000300611b5802621acc0366004f06010300601ac20260163003600067"
)
# GET tags 16-23 for port 1, sequence 10: the reply's command word is 14 01.
GET_PORT1 = bytes.fromhex("a500000a000acab200011011121314151617ff01ff2b")
GET_PORT1_REPLY = bytes.fromhex(
    "a51f0031000a9e4514011000010211000100120001111300001400040000000015000400000000"
    "160008000000000000000017000400000000ff011976"
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
def test_advertisement_is_identity_only(cc6, adv):
    state = cc6.parse_manufacturer_data(adv)
    assert (state.type, state.version, state.name) == (64, 20, "C-CB3AA")
    assert state.display_name == "Cloudcom 6 (C-CB3AA)"
    assert state.tmp is state.hum is state.vpd is None
    profile = state.profile
    assert profile.writable and profile.addressed and profile.packed_manual_level
    assert not profile.aggregate_fan
    assert state.outputs == ()


def test_other_models_keep_library_behavior(cc6):
    assert DeviceInfo(51, "Test", 20).profile.model_name == "Controller AI 2"
    assert not DeviceInfo(14, "Test", 4).profile.writable

    async def protocol_name(device_type):
        state = DeviceInfo(device_type, "Test", 20)
        return type(ACInfinityController(SimpleNamespace(), state)._protocol).__name__

    assert asyncio.run(protocol_name(51)) == "Protocol"
    assert asyncio.run(protocol_name(64)) == "Cloudcom6Protocol"
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
    assert controller.state.ports[0].kind is None
    assert controller.state.outputs == ()


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
        device.update.assert_awaited_once_with(0)

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


def test_fan_port_is_an_output_and_keeps_its_configured_mode(cc6):
    state = cc6.parse_telemetry(FRAME_FAN, DeviceInfo(64, "C-CB3AA", 20))
    assert (state.tmp, state.hum, state.vpd) == (
        pytest.approx(21.11),
        pytest.approx(68.6),
        pytest.approx(0.79),
    )
    assert cc6.probes(state)[0].hum == pytest.approx(56.8)
    fan = state.ports[1]
    assert (fan.kind, fan.raw_type, fan.level, fan.connected) == ("fan", 5069, 0, True)
    assert fan.mode is None  # unknown until a GET; telemetry has no mode
    assert state.ports[0].kind is None
    assert [p.id for p in state.outputs] == [1]
    # A mode read by GET survives telemetry for the same load, not a new one.
    configured = replace(state, ports={**state.ports, 1: replace(fan, mode=6)})
    assert cc6.parse_telemetry(FRAME_FAN, configured).ports[1].mode == 6
    swapped = replace(state, ports={**state.ports, 1: replace(fan, mode=6, raw_type=1)})
    assert cc6.parse_telemetry(FRAME_FAN, swapped).ports[1].mode is None


def test_get_reply_with_version_in_command_word_is_accepted(cc6):
    async def exercise():
        controller = make_controller()
        controller._notification_handler(None, bytearray(FRAME_FAN))

        async def reply(command):
            assert command == GET_PORT1
            controller._pending_sequence, controller._pending_command = 10, 1
            controller._notify_future = asyncio.get_running_loop().create_future()
            for start in range(0, len(GET_PORT1_REPLY), 20):
                chunk = bytearray(GET_PORT1_REPLY[start : start + 20])
                controller._notification_handler(None, chunk)
            return bytes(await controller._notify_future)

        controller._send_command = reply
        controller._sequence = 9
        await controller.update(1)
        port = controller.state.ports[1]
        assert (port.mode, port.level_on, port.level_off) == (2, 1, 0)
        # The stock protocol rejects the same reply.
        with pytest.raises(ValueError, match="command"):
            Protocol().parse_response(GET_PORT1_REPLY, 10, 1)
        with pytest.raises(ValueError, match="command"):
            controller._protocol.parse_response(GET_PORT1_REPLY, 10, 3)

    asyncio.run(exercise())
