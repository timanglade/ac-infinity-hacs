"""The ac_infinity sensor platform."""

from __future__ import annotations

from ac_infinity_ble import ACInfinityController
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, UnitOfPressure, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .cloudcom6 import ProbeReading, is_cloudcom6, probes
from .coordinator import ACInfinityDataUpdateCoordinator
from .entity import ACInfinityEntity
from .models import ACInfinityConfigEntry, ACInfinityData


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ACInfinityConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AC Infinity entities."""
    data: ACInfinityData = entry.runtime_data
    entities = [
        TemperatureSensor(data.coordinator, data.device, entry.title),
        HumiditySensor(data.coordinator, data.device, entry.title),
    ]
    if data.device.state.version >= 3:
        entities.append(VpdSensor(data.coordinator, data.device, entry.title))
    async_add_entities(entities)
    if not is_cloudcom6(data.device.state):
        return

    # Cloudcom 6 external probes appear in telemetry, possibly after setup.
    seen: set[int] = set()

    @callback
    def discover_probes() -> None:
        for probe in probes(data.device.state):
            if probe.index not in seen:
                seen.add(probe.index)
                async_add_entities(
                    cls(data.coordinator, data.device, entry.title, probe.index)
                    for cls in (
                        ProbeTemperatureSensor,
                        ProbeHumiditySensor,
                        ProbeVpdSensor,
                    )
                )

    entry.async_on_unload(data.coordinator.async_add_listener(discover_probes))
    discover_probes()


class ACInfinitySensor(ACInfinityEntity, SensorEntity):
    """Representation of AC Infinity sensor."""

    def __init__(
        self,
        coordinator: ACInfinityDataUpdateCoordinator,
        device: ACInfinityController,
        name: str,
    ) -> None:
        """Initialize an AC Infinity sensor."""
        super().__init__(coordinator, device, name)
        self._device = device
        self._name = name
        self._async_update_attrs()

    @property
    def available(self) -> bool:
        return super().available and self.coordinator.readings_available

    @callback
    def _async_update_attrs(self) -> None:
        """Handle updating _attr values."""
        raise NotImplementedError("Not yet implemented.")


class TemperatureSensor(ACInfinitySensor):
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        return f"{self._name} Temperature"

    @property
    def unique_id(self) -> str:
        """Return a unique, Home Assistant friendly identifier for this entity."""
        return f"{self._device.address}_tmp"

    @callback
    def _async_update_attrs(self) -> None:
        """Handle updating _attr values."""
        self._attr_native_value = self._device.temperature


class HumiditySensor(ACInfinitySensor):
    _attr_name = "Humidity"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_device_class = SensorDeviceClass.HUMIDITY
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        return f"{self._name} Humidity"

    @property
    def unique_id(self) -> str:
        """Return a unique, Home Assistant friendly identifier for this entity."""
        return f"{self._device.address}_hum"

    @callback
    def _async_update_attrs(self) -> None:
        """Handle updating _attr values."""
        self._attr_native_value = self._device.humidity


class VpdSensor(ACInfinitySensor):
    _attr_native_unit_of_measurement = UnitOfPressure.KPA
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        return f"{self._name} VPD"

    @property
    def unique_id(self) -> str:
        """Return a unique, Home Assistant friendly identifier for this entity."""
        return f"{self._device.address}_vpd"

    @callback
    def _async_update_attrs(self) -> None:
        """Handle updating _attr values."""
        self._attr_native_value = self._device.vpd


class _ProbeSensor:
    """External probe reading; unavailable while the probe is unplugged."""

    _key: str
    _label: str

    def __init__(
        self,
        coordinator: ACInfinityDataUpdateCoordinator,
        device: ACInfinityController,
        name: str,
        index: int,
    ) -> None:
        self._index = index
        super().__init__(coordinator, device, name)

    @property
    def _probe(self) -> ProbeReading | None:
        return next(
            (p for p in probes(self._device.state) if p.index == self._index), None
        )

    @property
    def available(self) -> bool:
        return super().available and self._probe is not None

    @property
    def name(self) -> str:
        return f"{self._name} Probe {self._index} {self._label}"

    @property
    def unique_id(self) -> str:
        return f"{self._device.address}_probe{self._index}_{self._key}"

    @callback
    def _async_update_attrs(self) -> None:
        probe = self._probe
        self._attr_native_value = getattr(probe, self._key) if probe else None


class ProbeTemperatureSensor(_ProbeSensor, TemperatureSensor):
    _key, _label = "tmp", "Temperature"


class ProbeHumiditySensor(_ProbeSensor, HumiditySensor):
    _key, _label = "hum", "Humidity"


class ProbeVpdSensor(_ProbeSensor, VpdSensor):
    _key, _label = "vpd", "VPD"
