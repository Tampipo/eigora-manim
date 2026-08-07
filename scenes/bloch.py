# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Bloch sphere: every pure spin-1/2 state as a direction in space.

Companion to spin-meas.py, which draws spin states as arrows without ever
saying why it is allowed to. The reason is the parametrisation

    |psi> = cos(theta/2)|u> + e^{i phi} sin(theta/2)|d>

which is onto: every pure state of a two-level system is some (theta, phi), and
every (theta, phi) is a state. So a state *is* a direction.

The one fact worth taking away is the factor of two. |u> and |d> are orthogonal
-- mutually exclusive outcomes -- yet they sit at opposite poles, 180 degrees
apart rather than 90. Angles in Hilbert space are half angles on the sphere,
which is exactly why the amplitude picked up when measuring at theta is
cos(theta/2) and not cos(theta).
"""

import numpy as np

from manim import (
    ThreeDScene,
    ThreeDAxes,
    Sphere,
    Arrow3D,
    Dot3D,
    Circle,
    MathTex,
    Text,
    VGroup,
    ValueTracker,
    always_redraw,
    FadeIn,
    FadeOut,
    Create,
    Write,
    ORIGIN,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    X_AXIS,
    DEGREES,
    PI,
    BLUE,
    GREEN,
    RED,
    YELLOW,
    TEAL,
    GREY_A,
    GREY_B,
    GREY_C,
    WHITE,
)

RADIUS = 2.0


def bloch_point(theta, phi, radius=RADIUS):
    """Cartesian position of the state (theta, phi) on the sphere.

    theta is measured from +z, phi anticlockwise from +x in the xy plane -- the
    usual spherical convention, and the one the ket formula uses.
    """
    return radius * np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


class BlochSphere(ThreeDScene):
    """Builds the sphere, then shows why the half-angle is there."""

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES, zoom=0.9)

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"stroke_color": GREY_C, "stroke_width": 2},
        )
        sphere = Sphere(
            radius=RADIUS,
            resolution=(28, 28),
            fill_opacity=0.10,
            stroke_width=0.4,
            stroke_color=GREY_C,
            checkerboard_colors=False,
            fill_color=BLUE,
        )
        equator = Circle(
            radius=RADIUS, stroke_color=GREY_B, stroke_width=2, stroke_opacity=0.7
        )

        self.play(Create(axes), run_time=1.2)
        self.play(FadeIn(sphere), Create(equator), run_time=1.5)

        # Axis labels live at fixed 3D positions but must keep facing the
        # camera, otherwise they are drawn edge-on and vanish as the view turns.
        labels = VGroup(
            MathTex("x", font_size=30, color=GREY_A).move_to(axes.c2p(3.4, 0, 0)),
            MathTex("y", font_size=30, color=GREY_A).move_to(axes.c2p(0, 3.4, 0)),
            MathTex("z", font_size=30, color=GREY_A).move_to(axes.c2p(0, 0, 3.4)),
        )
        self.add_fixed_orientation_mobjects(*labels)
        self.play(FadeIn(labels), run_time=0.8)

        # --- the poles, and the point of the whole picture ---------------------
        north = Dot3D(bloch_point(0, 0), radius=0.09, color=GREEN)
        south = Dot3D(bloch_point(PI, 0), radius=0.09, color=RED)
        ket_u = MathTex(r"|u\rangle", font_size=34, color=GREEN).move_to(
            bloch_point(0, 0) + UP * 0.45
        )
        ket_d = MathTex(r"|d\rangle", font_size=34, color=RED).move_to(
            bloch_point(PI, 0) + DOWN * 0.45
        )
        self.add_fixed_orientation_mobjects(ket_u, ket_d)
        self.play(FadeIn(north), FadeIn(south), FadeIn(ket_u), FadeIn(ket_d))

        # Captions are pinned to the screen, not to the scene: anything not
        # fixed in frame would tumble with the camera and be unreadable.
        caption = Text(
            "orthogonal states sit at opposite poles — 180° apart, not 90°",
            font_size=24,
            color=GREY_A,
        ).to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(caption)
        self.play(Write(caption), run_time=1.2)
        self.wait(1.5)

        # --- a general state ---------------------------------------------------
        theta = ValueTracker(40 * DEGREES)
        phi = ValueTracker(0.0)

        # always_redraw again: the arrow is rebuilt from the trackers every
        # frame, so animating a tracker animates the geometry.
        state = always_redraw(
            lambda: Arrow3D(
                start=ORIGIN,
                end=bloch_point(theta.get_value(), phi.get_value()),
                color=YELLOW,
                thickness=0.02,
                base_radius=0.06,
            )
        )
        self.add(state)

        formula = MathTex(
            r"|\psi\rangle =",
            r"\cos\!\tfrac{\theta}{2}\,|u\rangle",
            "+",
            r"e^{i\varphi}\sin\!\tfrac{\theta}{2}\,|d\rangle",
            font_size=36,
        )
        formula[1].set_color(GREEN)
        formula[3].set_color(RED)
        formula.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(formula)

        self.play(
            FadeOut(caption),
            Write(formula),
            run_time=1.4,
        )
        self.wait(1.0)

        # Sweep pole to pole: the state passes through the equator, where the
        # two amplitudes are equal and the outcome is a fair coin.
        self.play(theta.animate.set_value(PI), run_time=3.0)
        self.wait(0.6)
        self.play(theta.animate.set_value(90 * DEGREES), run_time=1.6)
        self.wait(0.4)

        # --- the equator: the phase ------------------------------------------
        phase = Text(
            "on the equator only the phase changes — all fair coins",
            font_size=24,
            color=TEAL,
        ).to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(phase)
        self.play(Write(phase), run_time=1.0)

        ket_r = MathTex(r"|r\rangle", font_size=32, color=TEAL).move_to(
            bloch_point(PI / 2, 0) + RIGHT * 0.5
        )
        ket_l = MathTex(r"|l\rangle", font_size=32, color=TEAL).move_to(
            bloch_point(PI / 2, PI) + LEFT * 0.5
        )
        self.add_fixed_orientation_mobjects(ket_r, ket_l)
        self.play(FadeIn(ket_r), FadeIn(ket_l))

        self.play(phi.animate.set_value(2 * PI), run_time=4.0)
        self.wait(0.5)

        # --- back to the measurement scene ------------------------------------
        self.play(FadeOut(phase), run_time=0.5)
        plane = Text(
            "the spin-o-meter lived in this slice: φ = 0, the xz plane",
            font_size=24,
            color=GREY_A,
        ).to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(plane)
        self.play(
            phi.animate.set_value(0.0),
            theta.animate.set_value(40 * DEGREES),
            Write(plane),
            run_time=2.5,
        )
        self.wait(2.0)
