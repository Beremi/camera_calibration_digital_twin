"""Public reporting helpers for the slim tabletop workflow."""

from __future__ import annotations

from calib_sim.reporting.tabletop_autodemo_dataset import (
    derive_noisy_dataset_variant,
    export_tabletop_autodemo_dataset,
    write_dataset_convenience_exports,
    write_tabletop_autodemo_corner_dataset_report,
    write_tabletop_autodemo_pixel_precision_report,
    write_tabletop_autodemo_dataset_report,
)


__all__ = [
    "derive_noisy_dataset_variant",
    "export_tabletop_autodemo_dataset",
    "write_dataset_convenience_exports",
    "write_tabletop_autodemo_corner_dataset_report",
    "write_tabletop_autodemo_pixel_precision_report",
    "write_tabletop_autodemo_dataset_report",
]
