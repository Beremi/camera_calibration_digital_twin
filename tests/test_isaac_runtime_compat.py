"""Regression tests for Isaac runtime environment preparation."""

from __future__ import annotations

from pathlib import Path

from calib_sim.sim.isaac_compat import prepare_isaac_runtime_environment


def _make_asset_converter_bundle(site_packages: Path) -> Path:
    lib_dir = (
        site_packages
        / "isaacsim"
        / "extscache"
        / "omni.kit.asset_converter-5.1.1+110.0.0.lx64.r.cp312.u7f4"
        / "asset_converter_native_bindings"
        / "libs"
    )
    lib_dir.mkdir(parents=True)
    (lib_dir / "libfbxsdk.so").write_bytes(b"fbx")
    (lib_dir / "libxml2.so.16").write_bytes(b"xml16")
    return lib_dir


def test_prepare_isaac_runtime_environment_creates_legacy_libxml_symlink(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    lib_dir = _make_asset_converter_bundle(site_packages)

    prepared = prepare_isaac_runtime_environment([site_packages])

    compat_link = lib_dir / "libxml2.so.2"
    assert prepared == [lib_dir]
    assert compat_link.is_symlink()
    assert compat_link.resolve() == lib_dir / "libxml2.so.16"


def test_prepare_isaac_runtime_environment_is_noop_without_expected_bundle(tmp_path: Path) -> None:
    prepared = prepare_isaac_runtime_environment([tmp_path / "site-packages"])
    assert prepared == []
