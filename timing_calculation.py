"""Predict THz reflection-geometry pulse arrival times for a tilted dielectric slab.

In a reflection-geometry THz-TDS setup you typically first record a reference
reflection from a flat metal mirror (e.g. gold) arriving at a known time T0.
You then replace the mirror with a transparent slab (e.g. fused-silica glass)
tilted at some angle of incidence. The slab produces TWO reflections that reach
the detector:

  1. Front-face reflection  - a partial Fresnel reflection off the first surface.
  2. Back-face reflection    - the part that refracts into the slab, reflects off
                               the back surface, and refracts back out.

This script predicts when each of those pulses should arrive, relative to the
metal-mirror reference, so you can use the pulse positions as an alignment target.

Two physical subtleties are handled explicitly, because both shift the timing by
several picoseconds and are easy to get wrong:

  * Oblique walk-off. The back-face echo does not exit the slab at the same point
    it entered. For a slab tilted in the plane of incidence, the exit point ends
    up displaced ALONG the detection direction, giving the back-face echo a
    head start that SHORTENS its delay. (See `exit_walkoff_time_s`.)

  * Front-face displacement. If you position the slab so its BACK face sits on the
    reference plane, the FRONT face necessarily lies one slab-thickness closer to
    the source. A 45 deg surface moved toward the source along its normal makes its
    reflection arrive earlier. (See `front_face_translation_time_s`.)

All optical paths assume the surrounding medium is air (refractive index ~ 1) and
that material dispersion is negligible across the THz band of interest, so the
single supplied refractive index is treated as both the phase and group index.
"""

import argparse
import math
from dataclasses import dataclass
from enum import Enum

SPEED_OF_LIGHT_M_PER_S = 2.99792458e8

MILLIMETRES_PER_METRE = 1.0e3
PICOSECONDS_PER_SECOND = 1.0e12


class ReferenceFace(str, Enum):
    """Which slab face is positioned to coincide with the metal-mirror reference plane."""

    FRONT = "front"
    BACK = "back"

    def __str__(self) -> str:
        return self.value


@dataclass
class ReflectionTimingResult:
    """Predicted arrival times and the physical components that produce them.

    All times are in seconds. Helper properties convert to picoseconds for display.
    """

    front_reflection_time_s: float
    back_reflection_time_s: float
    reference_time_s: float
    refraction_angle_rad: float

    # Component breakdown, all in seconds, for diagnostics and sanity checks.
    in_glass_transit_time_s: float
    exit_walkoff_time_s: float
    front_face_translation_time_s: float
    front_to_back_separation_time_s: float

    @property
    def front_reflection_time_ps(self) -> float:
        return self.front_reflection_time_s * PICOSECONDS_PER_SECOND

    @property
    def back_reflection_time_ps(self) -> float:
        return self.back_reflection_time_s * PICOSECONDS_PER_SECOND


def calculate_refraction_angle_rad(refractive_index: float, incidence_angle_rad: float) -> float:
    """Return the in-slab refraction angle from Snell's law (air -> slab).

    Snell's law with an air first medium (n1 = 1):  sin(theta_i) = n * sin(theta_t).
    """
    sine_of_refraction_angle = math.sin(incidence_angle_rad) / refractive_index
    return math.asin(sine_of_refraction_angle)


def in_glass_transit_time_s(
    refractive_index: float,
    thickness_m: float,
    refraction_angle_rad: float,
) -> float:
    """Time for the pulse to travel down to the back face and back, inside the slab.

    The one-way geometric path through the slab is `thickness / cos(theta_t)` because
    the ray travels obliquely to the surface normal. The round trip is twice that, and
    the optical path (which sets the time) is the geometric path scaled by the index.
    """
    one_way_path_length_m = thickness_m / math.cos(refraction_angle_rad)
    round_trip_optical_path_m = refractive_index * 2.0 * one_way_path_length_m
    return round_trip_optical_path_m / SPEED_OF_LIGHT_M_PER_S


def exit_walkoff_time_s(
    thickness_m: float,
    refraction_angle_rad: float,
    incidence_angle_rad: float,
) -> float:
    """Time saved because the back-face echo exits ahead along the detection axis.

    Ray-tracing the down-and-back path through a slab tilted in the plane of
    incidence shows the exit point is displaced from the entry/front-reflection
    point by `2 * d * tan(theta_t)` within the surface plane. Projected onto the
    detection (specular) direction this advance equals:

        2 * d * tan(theta_t) * sin(theta_i)

    Because the exit point is already that far along the propagation direction, the
    back-face echo has correspondingly less air to cross to reach the detector, so
    this is a time the echo SAVES (it is subtracted from the in-slab transit time).
    """
    exit_advance_along_detection_axis_m = (
        2.0 * thickness_m * math.tan(refraction_angle_rad) * math.sin(incidence_angle_rad)
    )
    return exit_advance_along_detection_axis_m / SPEED_OF_LIGHT_M_PER_S


def front_face_translation_time_s(thickness_m: float, incidence_angle_rad: float) -> float:
    """Time shift of the front-face reflection when the slab is moved by one thickness.

    Positioning the BACK face on the reference plane forces the FRONT face one slab
    thickness toward the source, measured along the surface normal. Translating a flat
    reflector by `d` along its normal changes the reflected round-trip path by
    `2 * d * cos(theta_i)`; moving toward the source shortens it, so the front-face
    reflection arrives EARLIER by this amount.
    """
    round_trip_path_change_m = 2.0 * thickness_m * math.cos(incidence_angle_rad)
    return round_trip_path_change_m / SPEED_OF_LIGHT_M_PER_S


def front_to_back_separation_time_s(
    refractive_index: float,
    thickness_m: float,
    refraction_angle_rad: float,
    incidence_angle_rad: float,
) -> float:
    """Time between the front-face and back-face echoes.

    This separation is intrinsic to the slab (it depends only on index, thickness,
    and angle), so it is independent of where the slab sits along the beam. It is the
    in-slab round-trip transit minus the oblique walk-off head start.
    """
    transit = in_glass_transit_time_s(refractive_index, thickness_m, refraction_angle_rad)
    walkoff = exit_walkoff_time_s(thickness_m, refraction_angle_rad, incidence_angle_rad)
    return transit - walkoff


def predict_reflection_times(
    refractive_index: float,
    thickness_m: float,
    incidence_angle_rad: float,
    reference_time_s: float,
    reference_face: ReferenceFace,
) -> ReflectionTimingResult:
    """Compute front-face and back-face reflection arrival times against the reference.

    `reference_face` selects which slab face is aligned to the metal-mirror plane:

      * ReferenceFace.FRONT - the front face sits on the reference plane, so the
        front-face reflection arrives at exactly `reference_time_s`, and the back-face
        echo arrives one slab-separation later.

      * ReferenceFace.BACK - the back face sits on the reference plane, so the front
        face is one thickness closer to the source: the front-face reflection arrives
        earlier than the reference, and the back-face echo follows one separation later.
    """
    refraction_angle_rad = calculate_refraction_angle_rad(refractive_index, incidence_angle_rad)

    transit = in_glass_transit_time_s(refractive_index, thickness_m, refraction_angle_rad)
    walkoff = exit_walkoff_time_s(thickness_m, refraction_angle_rad, incidence_angle_rad)
    front_translation = front_face_translation_time_s(thickness_m, incidence_angle_rad)
    separation = transit - walkoff

    if reference_face is ReferenceFace.FRONT:
        front_reflection_time_s = reference_time_s
    else:
        front_reflection_time_s = reference_time_s - front_translation

    back_reflection_time_s = front_reflection_time_s + separation

    return ReflectionTimingResult(
        front_reflection_time_s=front_reflection_time_s,
        back_reflection_time_s=back_reflection_time_s,
        reference_time_s=reference_time_s,
        refraction_angle_rad=refraction_angle_rad,
        in_glass_transit_time_s=transit,
        exit_walkoff_time_s=walkoff,
        front_face_translation_time_s=front_translation,
        front_to_back_separation_time_s=separation,
    )


def format_timing_report(
    result: ReflectionTimingResult,
    reference_face: ReferenceFace,
) -> str:
    """Render a human-readable report of the predicted times and their components."""
    to_ps = PICOSECONDS_PER_SECOND
    lines = [
        "THz slab reflection timing prediction",
        "=" * 44,
        f"Reference face on plane : {reference_face.value}",
        f"Refraction angle        : {math.degrees(result.refraction_angle_rad):.3f} deg",
        "",
        "Component breakdown:",
        f"  In-slab round-trip transit  : {result.in_glass_transit_time_s * to_ps:+.3f} ps",
        f"  Oblique walk-off (head start): {-result.exit_walkoff_time_s * to_ps:+.3f} ps",
    ]
    if reference_face is ReferenceFace.BACK:
        lines.append(
            f"  Front-face moved to source   : {-result.front_face_translation_time_s * to_ps:+.3f} ps"
        )
    lines += [
        f"  Front-to-back separation     :  {result.front_to_back_separation_time_s * to_ps:.3f} ps",
        "",
        "Target arrival times:",
        f"  Front-face reflection : {result.front_reflection_time_ps:.2f} ps",
        f"  Back-face reflection  : {result.back_reflection_time_ps:.2f} ps",
    ]
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict front/back reflection arrival times for a tilted THz slab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "refractive_index",
        type=float,
        help="Slab refractive index across the THz band (dimensionless).",
    )
    parser.add_argument(
        "thickness_mm",
        type=float,
        help="Slab thickness measured along the surface normal, in millimetres.",
    )
    parser.add_argument(
        "incidence_angle_deg",
        type=float,
        help="Angle of incidence in air, in degrees (e.g. 45).",
    )
    parser.add_argument(
        "reference_time_ps",
        type=float,
        help="Arrival time of the metal-mirror reference reflection, in picoseconds.",
    )
    parser.add_argument(
        "--reference-face",
        type=ReferenceFace,
        choices=list(ReferenceFace),
        default=ReferenceFace.BACK,
        metavar="{front,back}",
        help="Which slab face is aligned to the reference plane.",
    )
    return parser


def validate_inputs(refractive_index: float, thickness_mm: float, incidence_angle_deg: float) -> None:
    """Raise a clear error for physically invalid inputs before computing."""
    if refractive_index < 1.0:
        raise ValueError("Refractive index must be >= 1 for an air-to-slab interface.")
    if thickness_mm <= 0.0:
        raise ValueError("Thickness must be positive.")
    if not 0.0 <= incidence_angle_deg < 90.0:
        raise ValueError("Angle of incidence must be in the range [0, 90) degrees.")


def report_from_values(
    refractive_index: float,
    thickness_mm: float,
    incidence_angle_deg: float,
    reference_time_ps: float,
    reference_face: ReferenceFace = ReferenceFace.BACK,
) -> ReflectionTimingResult:
    """Compute, print, and return the timing result from plain numeric values.

    Use this when calling the tool programmatically (from a notebook or another
    module) with real numbers, e.g. report_from_values(1.94, 2.08, 45.0, 150.0).
    The command-line entry point `main` is for the STRING arguments the shell
    provides and must not be passed floats directly.
    """
    validate_inputs(refractive_index, thickness_mm, incidence_angle_deg)
    result = predict_reflection_times(
        refractive_index=refractive_index,
        thickness_m=thickness_mm / MILLIMETRES_PER_METRE,
        incidence_angle_rad=math.radians(incidence_angle_deg),
        reference_time_s=reference_time_ps / PICOSECONDS_PER_SECOND,
        reference_face=reference_face,
    )
    print(format_timing_report(result, reference_face))
    return result


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    `argv` is an optional list of STRING arguments, as the shell would provide
    (e.g. ["1.94", "2.08", "45", "150"]). When None, argparse reads them from
    sys.argv. Do not pass floats here: argparse expects strings and converts them
    via each argument's `type=float`. For numeric calls from Python, use
    `report_from_values` instead.
    """
    arguments = build_argument_parser().parse_args(argv)
    report_from_values(
        refractive_index=arguments.refractive_index,
        thickness_mm=arguments.thickness_mm,
        incidence_angle_deg=arguments.incidence_angle_deg,
        reference_time_ps=arguments.reference_time_ps,
        reference_face=arguments.reference_face,
    )


if __name__ == "__main__":
    main()