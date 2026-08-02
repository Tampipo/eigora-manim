# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scanning tunneling microscope explainer.

One continuous scene: a tip scans a bumpy 2D sample surface, the camera
zooms into the tallest bump (smallest gap, highest current), then flattens
into the constant-height 1D cross-section scan through that same point --
axes, current trace and all -- reached by zooming rather than a cut.

The tunneling transmission coefficient follows the same WKB rectangular-
barrier formula shown in the barrier.mdx article:

    T ~= 16 (E/V0)(1 - E/V0) * exp(-2 * kappa * d),  kappa = sqrt(2(V0 - E))

(atomic units, hbar = m = 1). Because T decays exponentially in d, the
current is sharply peaked over each bump -- the same exponential
sensitivity that gives STM its atomic-scale resolution.
"""

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    GOLD,
    GREY_B,
    GREY_D,
    LEFT,
    PI,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Axes,
    Create,
    DashedLine,
    Dot3D,
    FadeIn,
    FadeOut,
    Line3D,
    MathTex,
    ParametricFunction,
    Surface,
    ThreeDScene,
    Triangle,
    ValueTracker,
    VGroup,
    always_redraw,
    interpolate_color,
)

V0, E = 5.0, 3.2
KAPPA = np.sqrt(2 * (V0 - E))
T_PREFACTOR = 16 * (E / V0) * (1 - E / V0)


def transmission(d):
    return T_PREFACTOR * np.exp(-2 * KAPPA * max(d, 0.05))


# ── 2D surface + zoom-through ────────────────────────────────────────────────

BUMPS_2D = [
    (-2.0, -1.5, 0.55, 0.9),
    (0.5, 1.2, 0.85, 0.8),
    (2.0, -1.0, 0.4, 0.7),
    (-1.0, 1.8, 0.7, 0.75),
    (2.2, 1.5, 0.5, 0.6),
]

SURF_RANGE = 3.5
TIP_Z = 1.5
ZOOM_POINT = (0.5, 1.2)  # the tallest bump -- smallest gap, highest current


def surface_height_2d(x, y):
    return sum(
        a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, a, w in BUMPS_2D
    )


def gap_2d(x, y):
    return TIP_Z - surface_height_2d(x, y)


T_MAX = transmission(gap_2d(*ZOOM_POINT))  # tallest bump = smallest gap = highest T


class STMSurfaceZoom(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-60 * DEGREES, zoom=0.75)

        surface = Surface(
            lambda u, v: np.array([u, v, surface_height_2d(u, v)]),
            u_range=[-SURF_RANGE, SURF_RANGE],
            v_range=[-SURF_RANGE, SURF_RANGE],
            resolution=(36, 36),
            fill_opacity=0.85,
            checkerboard_colors=False,
            fill_color=GREY_D,
            stroke_color=GOLD,
            stroke_width=0.4,
        )

        self.play(Create(surface), run_time=2.5)

        waypoints = [(2.2, 1.5), (-1.0, 1.8), ZOOM_POINT]
        tx = ValueTracker(waypoints[0][0])
        ty = ValueTracker(waypoints[0][1])

        def tip_point():
            return np.array([tx.get_value(), ty.get_value(), TIP_Z])

        def surface_point():
            x, y = tx.get_value(), ty.get_value()
            return np.array([x, y, surface_height_2d(x, y)])

        tip = always_redraw(lambda: Dot3D(point=tip_point(), radius=0.09, color=WHITE))
        gap_line = always_redraw(
            lambda: Line3D(
                start=tip_point(), end=surface_point(), color=YELLOW, thickness=0.02
            )
        )

        # Electron flux: a few dots looping from the sample up to the tip,
        # brighter wherever the local current (~T(gap)) is higher -- the same
        # exponential sensitivity as the 1D scan above.
        n_particles = 5
        flux_time = ValueTracker(0.0)
        flux_time.add_updater(lambda m, dt: m.increment_value(dt * 0.6))
        self.add(flux_time)

        def make_particle(phase):
            def updater(m):
                x, y = tx.get_value(), ty.get_value()
                z0, z1 = surface_height_2d(x, y), TIP_Z
                t = (flux_time.get_value() + phase) % 1.0
                m.move_to(np.array([x, y, z0 + t * (z1 - z0)]))
                strength = np.clip(transmission(gap_2d(x, y)) / T_MAX, 0.05, 1.0)
                m.set_opacity(np.sin(t * PI) * strength)

            dot = Dot3D(radius=0.035, color=YELLOW)
            dot.add_updater(updater)
            return dot

        flux = VGroup(*[make_particle(i / n_particles) for i in range(n_particles)])

        self.play(FadeIn(tip), Create(gap_line), FadeIn(flux))

        # The tip goes around the surface before settling on the tallest bump.
        for wx, wy in waypoints[1:]:
            self.play(tx.animate.set_value(wx), ty.animate.set_value(wy), run_time=1.8)

        zx, zy = ZOOM_POINT
        zz = surface_height_2d(zx, zy)

        # Swing round to look at the sample edge-on. Centred on x=0 so the
        # sample sits in frame with equal margins either side.
        CAM_ZOOM = 1.43
        CAM_CZ = 0.14
        self.move_camera(
            phi=90 * DEGREES,
            theta=-90 * DEGREES,
            zoom=CAM_ZOOM,
            frame_center=np.array([0.0, zy, CAM_CZ]),
            run_time=5,
        )

        x_min_1d, x_max_1d = -SURF_RANGE, SURF_RANGE

        def cross_section(x):
            return surface_height_2d(x, zy)

        def gap_1d(x):
            return TIP_Z - cross_section(x)

        t_max_1d = max(
            transmission(gap_1d(x)) for x in np.linspace(x_min_1d, x_max_1d, 400)
        )

        # The silhouette of the whole surface is NOT the slice we're scanning
        # (it's the envelope over every y). Draw the actual y = zy slice on the
        # surface first, then strip everything else away, so what's left really
        # is the profile under the tip.
        slice_curve = ParametricFunction(
            lambda t: np.array([t, zy, cross_section(t)]),
            t_range=[x_min_1d, x_max_1d],
            color=GOLD,
            stroke_width=4,
        )
        self.play(Create(slice_curve), run_time=1.5)

        flux_time.clear_updaters()
        self.play(
            FadeOut(surface), FadeOut(tip), FadeOut(gap_line), FadeOut(flux), run_time=1.5
        )

        # ── Build the flat figure at *exactly* the screen coordinates the
        # camera maps this slice to, so swapping one for the other is
        # invisible. Under phi=90, theta=-90 the projection is simply
        #     screen_x = (x - cx) * zoom,  screen_y = (z - cz) * zoom.
        def to_screen(x, z):
            return np.array([(x - 0.0) * CAM_ZOOM, (z - CAM_CZ) * CAM_ZOOM, 0.0])

        surface_axes = Axes(
            x_range=[x_min_1d, x_max_1d, 1],
            y_range=[0, TIP_Z + 0.6, 0.5],
            x_length=(x_max_1d - x_min_1d) * CAM_ZOOM,
            y_length=(TIP_Z + 0.6) * CAM_ZOOM,
            axis_config={"include_tip": False},
        )
        surface_axes.shift(to_screen(0, 0) - surface_axes.c2p(0, 0))

        current_axes = Axes(
            x_range=[x_min_1d, x_max_1d, 1],
            y_range=[0, 1.05, 0.5],
            x_length=(x_max_1d - x_min_1d) * CAM_ZOOM,
            y_length=2.0,
            axis_config={"include_tip": False},
        )
        current_axes.shift(np.array([to_screen(0, 0)[0], -3.3, 0]) - current_axes.c2p(0, 0))

        surface_curve = surface_axes.plot(
            cross_section, x_range=[x_min_1d, x_max_1d], color=GOLD
        )
        surface_fill = surface_axes.get_area(
            surface_curve, x_range=[x_min_1d, x_max_1d], color=GREY_D, opacity=0.7
        )

        x_label_top = MathTex("x", font_size=28).next_to(surface_axes.x_axis, RIGHT, buff=0.15)
        y_label_top = (
            MathTex("z", font_size=28).next_to(surface_axes.y_axis, UP, buff=0.15)
        )
        x_label_bottom = MathTex("x", font_size=28).next_to(current_axes.x_axis, RIGHT, buff=0.15)
        current_label = (
            MathTex(r"T(x)", font_size=28)
            .next_to(current_axes.y_axis, UP, buff=0.15)
        )

        x_tracker = ValueTracker(x_min_1d)

        tip2 = Triangle(fill_opacity=1, color=WHITE, stroke_width=0).scale(0.14).rotate(PI)
        tip2.add_updater(
            lambda m: m.move_to(surface_axes.c2p(x_tracker.get_value(), TIP_Z))
        )

        connector2 = DashedLine(
            surface_axes.c2p(x_min_1d, TIP_Z),
            surface_axes.c2p(x_min_1d, cross_section(x_min_1d)),
            color=GREY_B,
            stroke_width=3,
        )

        def update_connector(m):
            x = x_tracker.get_value()
            frac = np.clip(transmission(gap_1d(x)) / t_max_1d, 0, 1)
            m.become(
                DashedLine(
                    surface_axes.c2p(x, TIP_Z),
                    surface_axes.c2p(x, cross_section(x)),
                    color=interpolate_color(GREY_B, YELLOW, frac),
                    stroke_width=3,
                )
            )

        current_trace = current_axes.plot(
            lambda x: transmission(gap_1d(x)) / t_max_1d,
            x_range=[x_min_1d, x_min_1d + 1e-3],
            color=BLUE,
        )

        def update_trace(m):
            m.become(
                current_axes.plot(
                    lambda x: transmission(gap_1d(x)) / t_max_1d,
                    x_range=[x_min_1d, max(x_min_1d + 1e-3, x_tracker.get_value())],
                    color=BLUE,
                )
            )

        # Everything flat lives fixed in frame so the 3D camera can't touch it.
        # FadeIn (rather than poking .set_opacity) keeps each element's own
        # styling -- notably the grey fill under the profile.
        figure = VGroup(
            surface_axes,
            surface_fill,
            surface_curve,
            x_label_top,
            y_label_top,
            current_axes,
            current_label,
            x_label_bottom,
            tip2,
            connector2,
            current_trace,
        )
        self.add_fixed_in_frame_mobjects(figure)

        # The plotted curve lands exactly on the 3D slice, so this swap reads
        # as the slice simply gaining its axes rather than as a transition.
        self.play(FadeIn(figure), FadeOut(slice_curve), run_time=1.5)

        connector2.add_updater(update_connector)
        current_trace.add_updater(update_trace)

        self.play(
            x_tracker.animate.set_value(x_max_1d),
            run_time=10,
            rate_func=lambda t: t,
        )
        self.wait(1)
