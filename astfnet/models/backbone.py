"""Backbone registry and factory for ASTF-net models.

All raw ``nn.Module`` backbones are registered here so that the unified
:class:`~astfnet.models.base.ASTFModule` can build any architecture from
a flat configuration dictionary.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_backbone(name: str) -> Callable:
    """Class decorator that adds a backbone to the registry.

    Args:
        name: Key used in config ``model_name`` to select this backbone.

    Returns:
        The original class, unmodified.
    """

    def _register(cls: type) -> type:
        if name in _BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' is already registered.")
        _BACKBONE_REGISTRY[name] = cls
        return cls

    return _register


def list_backbones() -> list[str]:
    """Return sorted list of registered backbone names."""
    return sorted(_BACKBONE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backbone(config: Dict[str, Any]) -> nn.Module:
    """Construct a backbone ``nn.Module`` from a configuration dict.

    The ``model_name`` key selects the backbone class. Remaining keys are
    forwarded as constructor kwargs (unknown keys are silently ignored so
    that the same config can drive both the backbone and the Lightning
    module).

    Args:
        config: Flat configuration dictionary.

    Returns:
        Instantiated backbone module.

    Raises:
        ValueError: If ``model_name`` is not in the registry.
    """
    name = config.get("model_name", "simplecnn").lower()
    if name not in _BACKBONE_REGISTRY:
        available = ", ".join(list_backbones())
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}")

    cls = _BACKBONE_REGISTRY[name]
    return _build_from_config(cls, config)


def _build_from_config(cls: Callable[..., nn.Module], config: Dict[str, Any]) -> nn.Module:
    """Instantiate *cls* passing only the kwargs it accepts.

    Args:
        cls: The backbone class.
        config: Full configuration dictionary.

    Returns:
        Instantiated module.
    """
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_keys = {p.name for p in sig.parameters.values() if p.name != "self"}
    filtered = {k: v for k, v in config.items() if k in valid_keys}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Trigger registrations by importing backbone modules
# ---------------------------------------------------------------------------


def _ensure_registrations() -> None:
    """Import backbone modules so ``@register_backbone`` decorators run."""
    import astfnet.models.cnn  # noqa: F401
    import astfnet.models.crdnn  # noqa: F401
    import astfnet.models.transformer  # noqa: F401
    import astfnet.models.unet1d  # noqa: F401


_ensure_registrations()
