"""Compatibility helpers for Isaac Sim Python environments.

Some Isaac Sim binary bundles still ship secondary libraries that look for the
legacy ``libxml2.so.2`` soname. Recent Arch-based distributions provide
``libxml2.so.16`` instead, but Isaac already bundles a matching private copy in
the asset-converter extension cache. We create a local compatibility symlink
inside that bundle when needed so the runtime can start without mutating system
libraries.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable


_ASSET_CONVERTER_GLOB = "isaacsim/extscache/omni.kit.asset_converter-*/asset_converter_native_bindings/libs"
def _iter_candidate_site_roots(site_roots: Iterable[Path] | None = None) -> list[Path]:
    roots = list(site_roots or [])
    if site_roots is None:
        roots.extend(Path(entry) for entry in sys.path if entry)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _iter_asset_converter_lib_dirs(site_roots: Iterable[Path] | None = None) -> list[Path]:
    lib_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in _iter_candidate_site_roots(site_roots):
        for lib_dir in root.glob(_ASSET_CONVERTER_GLOB):
            if lib_dir in seen or not lib_dir.is_dir():
                continue
            seen.add(lib_dir)
            lib_dirs.append(lib_dir)
    return lib_dirs


def prepare_isaac_runtime_environment(site_roots: Iterable[Path] | None = None) -> list[Path]:
    """Patch known Linux packaging mismatches before importing Isaac Sim.

    Returns the bundle directories that were modified.
    """

    prepared: list[Path] = []
    for lib_dir in _iter_asset_converter_lib_dirs(site_roots):
        bundled_xml = lib_dir / "libxml2.so.16"
        legacy_xml = lib_dir / "libxml2.so.2"
        fbx_sdk = lib_dir / "libfbxsdk.so"
        if not fbx_sdk.exists() or legacy_xml.exists() or not bundled_xml.exists():
            continue
        legacy_xml.symlink_to(bundled_xml.name)
        prepared.append(lib_dir)
    return prepared


__all__ = [
    "prepare_isaac_runtime_environment",
]
