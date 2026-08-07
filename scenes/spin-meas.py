# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Successive measurements of a spin-1/2 along different axes.

Two scenes:

    ConsecutiveMeasurements  one electron crossing a line of apparatuses, each
                             measuring along an axis of its own
    ElectronFlux             the same measurement repeated on an ensemble until
                             the running average settles

Both rest on one relation. For a spin pointing theta away from the axis being
measured,

    |psi> = cos(theta/2)|+n> + sin(theta/2)|-n>

so a single reading is +1 or -1 with probabilities cos^2(theta/2) and
sin^2(theta/2), and the mean over many is cos(theta). Every individual reading
is +-1; only the average is cos(theta).
"""

import numpy as np

from manim import (
    Scene,
    MovingCameraScene,
    FadeOut,
    MathTex,
    VGroup,
    RoundedRectangle,
    Arrow,
    Circle,
    Arc,
    Line,
    DashedLine,
    Text,
    UP,
    DOWN,
    RIGHT,
    LEFT,
    BLACK,
    GREY_A,
    GREY_B,
    GREY_C,
    GREY_D,
    GREY_E,
    YELLOW,
    TAU,
    Dot,
    ORIGIN,
    DEGREES,
    BLUE,
    GREEN,
    RED,
    TEAL,
    FadeIn,
    Flash,
    Indicate,
    Transform,
    Rotate,
    always_redraw,
    Rectangle,
    Succession,
    LaggedStart,
    ValueTracker,
    DecimalNumber,
    linear,
)


# Tilt of the incoming spin away from the first needle: the direction the
# electron arrives in, before anything has measured it. Drives both the drawing
# and the algebra.
SPIN_ANGLE_DEG = 40


class MeasurementDevice(VGroup):
    """A spin measuring apparatus: a dial that reads +1 or -1, and an arrow
    engraved beside it showing the direction along which the spin is measured.

    The parts the animation needs to touch are kept as attributes:

        .arrow    rotate about `.pivot` to change the measured direction
        .reading  Transform to change the displayed value
        .dial     the bezel, e.g. flash it on measurement
        .pivot    centre of rotation for the arrow

    The dial starts blank. The apparatus reads nothing until a measurement is
    made, so there is no value to show before the first one.
    """

    def __init__(self, name="Spin-o-meter", angle_deg=0, **kwargs):
        super().__init__(**kwargs)

        body = RoundedRectangle(
            width=5.0,
            height=3.2,
            corner_radius=0.14,
            stroke_color=GREY_B,
            stroke_width=2,
            fill_color=GREY_E,
            fill_opacity=1,
        )

        # Nameplate across the top, with a rule under it.
        plate = Text(name, font_size=20, color=GREY_A)
        plate.next_to(body.get_top(), DOWN, buff=0.18)
        rule = Line(
            body.get_left() + RIGHT * 0.22,
            body.get_right() + LEFT * 0.22,
            stroke_color=GREY_D,
            stroke_width=1.5,
        ).next_to(plate, DOWN, buff=0.12)

        # Dial on the left, blank until measured.
        dial = Circle(
            radius=0.46, stroke_color=GREY_B, stroke_width=2,
            fill_color=BLACK, fill_opacity=1,
        )
        reading = MathTex(r"\text{--}", font_size=42, color=GREY_C)

        # Needle on the right, upright to start (+z). It sits on a graduated
        # face of its own so a rotation reads as a direction being chosen, and
        # against a fixed scale, so any angle can be read straight off the dial.
        face_radius = 0.92
        face = Circle(radius=face_radius, stroke_color=GREY_D, stroke_width=1.5)

        # Ticks every 15 degrees, longer and brighter at the four cardinal
        # directions -- which are exactly +z, +x, -z, -x.
        ticks = VGroup()
        for i in range(24):
            angle = i * TAU / 24
            cardinal = i % 6 == 0
            length = 0.14 if cardinal else 0.07
            outward = np.array([np.cos(angle), np.sin(angle), 0.0])
            ticks.add(
                Line(
                    face.get_center() + outward * (face_radius - length),
                    face.get_center() + outward * face_radius,
                    stroke_color=GREY_B if cardinal else GREY_D,
                    stroke_width=2 if cardinal else 1,
                )
            )

        # The z and x axes the needle is resolved against, drawn on the face so
        # the decomposition has something to be a decomposition *onto*.
        axis_span = face_radius - 0.16
        axes = VGroup(
            Line(DOWN * axis_span, UP * axis_span, stroke_color=GREY_D, stroke_width=1),
            Line(LEFT * axis_span, RIGHT * axis_span, stroke_color=GREY_D, stroke_width=1),
        )
        z_label = MathTex("z", font_size=22, color=GREY_B).move_to(
            UP * (face_radius + 0.2) + LEFT * 0.16
        )
        x_label = MathTex("x", font_size=22, color=GREY_B).move_to(
            RIGHT * (face_radius + 0.2) + DOWN * 0.16
        )

        arrow = Arrow(
            start=ORIGIN,
            end=UP * (face_radius - 0.1),
            color=YELLOW,
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.22,
        )

        VGroup(dial, face).arrange(RIGHT, buff=0.85)
        reading.move_to(dial.get_center())
        # Everything drawn on the face is built about the origin, then moved onto
        # the face as one piece -- so the axes, labels and needle all share the
        # face's centre exactly.
        VGroup(ticks, axes, z_label, x_label, arrow).shift(face.get_center())
        VGroup(dial, face, ticks, axes, z_label, x_label, reading, arrow).next_to(
            rule, DOWN, buff=0.3
        )

        # Point the needle at its measurement direction. theta is measured from
        # +z toward +x, which is clockwise on screen, hence the minus sign. Done
        # after all the layout, so the pivot is the face's final centre.
        if angle_deg:
            arrow.rotate(-angle_deg * DEGREES, about_point=face.get_center())

        self.body = body
        self.dial = dial
        self.reading = reading
        self.face = face
        self.ticks = ticks
        self.axes = axes
        self.arrow = arrow

        self.add(
            body, plate, rule, dial, face, ticks, axes, z_label, x_label,
            reading, arrow,
        )

    @property
    def pivot(self):
        """Centre of rotation for the needle.

        A property, not an attribute captured in __init__: a coordinate stored
        at construction goes stale the moment anything calls device.move_to(),
        and the needle would then rotate about wherever the face used to be.
        """
        return self.face.get_center()
        
def make_electron(spin_angle, ket=r"|\psi\rangle"):
    """An electron on the beam: a dot, an arrow for the direction its spin
    points, and a ket label underneath. Built at the origin -- shift it into
    place, so the dot lands exactly where you ask rather than the group's
    bounding-box centre landing there.

    Returns the travelling group plus the parts that get changed later.
    """
    dot = Dot(radius=0.09, color=BLUE)
    spin = Arrow(
        start=ORIGIN,
        end=UP * 0.6,
        color=BLUE,
        buff=0,
        stroke_width=3,
        max_tip_length_to_length_ratio=0.3,
    )
    # Minus sign for the same reason as the needle: theta runs from +z toward
    # +x, which is clockwise on screen, while Manim's rotate() is anticlockwise.
    # Get this wrong and the spin leans the wrong way, and every later collapse
    # adds to the tilt instead of cancelling it.
    spin.rotate(-spin_angle, about_point=ORIGIN)
    label = MathTex(ket, font_size=36, color=BLUE).next_to(dot, DOWN, buff=0.3)
    return VGroup(dot, spin, label), dot, spin, label


def arrow_components(device):
    """The needle resolved onto the z and x axes drawn on its face.

    For a needle at theta from +z the two components are cos(theta) and
    sin(theta), closed into a rectangle by dashed guides. The z component is the
    one that carries the physics: it is <sigma_n>, so it shrinks to nothing at
    90 degrees -- where the two outcomes become equally likely -- and turns red
    as it goes negative past the horizontal.

    Rebuilt from the arrow's *current* position on every call, which is what
    lets always_redraw keep it glued to a rotating needle.
    """
    centre = device.pivot
    tip = device.arrow.get_end()
    foot_z = np.array([centre[0], tip[1], 0.0])
    foot_x = np.array([tip[0], centre[1], 0.0])

    z_component = Line(
        centre,
        foot_z,
        stroke_color=GREEN if tip[1] >= centre[1] else RED,
        stroke_width=6,
    )
    x_component = Line(centre, foot_x, stroke_color=TEAL, stroke_width=6)
    guides = VGroup(
        DashedLine(foot_z, tip, stroke_color=GREY_C, stroke_width=1.2, dash_length=0.05),
        DashedLine(foot_x, tip, stroke_color=GREY_C, stroke_width=1.2, dash_length=0.05),
    )
    return VGroup(z_component, x_component, guides)


def theta_overlay(device, needle_deg, spin_deg):
    """The incoming spin drawn on the device's own face, beside the needle.

    This is the link the animation was missing: the electron's arrow and the
    needle are two directions, and theta is the angle between them. Drawing the
    spin onto the same face -- against the same axes, with an arc spanning the
    gap -- is what makes the number in the panel below mean something.

    The overlay arrow is drawn short so that when the two directions coincide
    (theta = 0, a repeat measurement) you can still see both.
    """
    centre = device.pivot
    radius = device.face.width / 2

    def direction(deg):
        # theta runs from +z toward +x: up at 0, right at 90.
        rad = deg * DEGREES
        return np.array([np.sin(rad), np.cos(rad), 0.0])

    spin_arrow = Arrow(
        start=centre,
        end=centre + direction(spin_deg) * radius * 0.62,
        color=BLUE,
        buff=0,
        stroke_width=3,
        max_tip_length_to_length_ratio=0.3,
    )

    parts = VGroup(spin_arrow)

    # Manim angles run anticlockwise from +x, ours run clockwise from +z.
    a_spin = (90 - spin_deg) * DEGREES
    a_needle = (90 - needle_deg) * DEGREES
    if abs(needle_deg - spin_deg) > 1e-6:
        arc = Arc(
            radius=radius * 0.4,
            start_angle=a_spin,
            angle=a_needle - a_spin,
            arc_center=centre,
            stroke_color=GREY_A,
            stroke_width=2,
        )
        mid = (a_spin + a_needle) / 2
        label = MathTex(r"\theta", font_size=24, color=GREY_A).move_to(
            centre + np.array([np.cos(mid), np.sin(mid), 0.0]) * radius * 0.58
        )
        parts.add(arc, label)

    return parts


def decomposition_panel(theta_deg, font_size=34):
    """The incoming spin resolved onto the basis the device measures in.

    A spin pointing theta away from the device's arrow is

        |psi> = cos(theta/2)|u> + sin(theta/2)|d>

    so the Born probabilities are the squares of those two coefficients. The
    half-angle is the whole story: it is why theta = 90 degrees gives 50/50
    rather than certainty, and why <sigma_n> works out to cos(theta).

    Returns VGroup(ket_line, probability_line) so a scene can Indicate or dim
    either branch by index.
    """
    theta = theta_deg * DEGREES
    c, s = np.cos(theta / 2), np.sin(theta / 2)

    ket = MathTex(
        r"|\psi\rangle =",
        rf"{c:.2f}\,|u\rangle",
        "+",
        rf"{s:.2f}\,|d\rangle",
        rf"\quad (\theta = {theta_deg:g}^\circ)",
        font_size=font_size,
    )
    ket[1].set_color(GREEN)
    ket[3].set_color(RED)
    ket[4].set_color(GREY_B)

    probs = MathTex(
        rf"P(+1) = \cos^2(\theta/2) = {c ** 2:.2f}",
        r"\qquad",
        rf"P(-1) = \sin^2(\theta/2) = {s ** 2:.2f}",
        font_size=font_size - 4,
    )
    probs[0].set_color(GREEN)
    probs[2].set_color(RED)

    return VGroup(ket, probs).arrange(DOWN, buff=0.25)


# One row per apparatus the electron passes through, in order. Each is
# (needle direction from +z, spin direction afterwards, what the dial reads,
#  ket afterwards).
#
# theta is deliberately NOT stored: it is the angle between the incoming spin
# and the needle, so it is derived at run time from the state the electron
# actually arrives in. Storing it as well would let the two disagree.
BEATS = [
    (0, 0, "+1", r"|u\rangle"),
    (0, 0, "+1", r"|u\rangle"),
    # The dial reverses, the state does not: measuring along -z on |u> returns
    # -1 with certainty and leaves the spin exactly where it was. This is where
    # "the reading flipped" and "the spin flipped" come apart.
    (180, 0, "-1", r"|u\rangle"),
    # 90 degrees: the first genuinely undetermined outcome. +1 here is one roll
    # of a fair coin, not a prediction.
    (90, 90, "+1", r"|r\rangle"),
]

SPACING = 6.8

# How wide the camera is while looking at a single apparatus. Everything else
# is measured against this: it is the "readable" zoom level, and annotations get
# scaled by frame.width / BASE_WIDTH so they keep that size as we pull back.
BASE_WIDTH = 10.5


class ConsecutiveMeasurements(MovingCameraScene):
    """One electron crossing a line of apparatuses.

    Only the apparatus in use is on screen. Each time the electron clears one,
    the camera pulls back and the next fades in -- so the sequence is revealed
    rather than laid out in advance, and the growing record of readings stays
    visible behind the electron.
    """

    def construct(self):
        devices = VGroup()
        components = VGroup()
        for i, (needle, *_rest) in enumerate(BEATS):
            device = MeasurementDevice(angle_deg=needle)
            device.move_to(RIGHT * i * SPACING)
            # The electron passes behind the panel: each measurement happens
            # inside its apparatus, out of sight.
            device.set_z_index(2)
            devices.add(device)
            components.add(arrow_components(device).set_z_index(3))

        frame = self.camera.frame
        frame.set(width=BASE_WIDTH).move_to(devices[0].get_center() + DOWN * 0.45)

        # Only the first apparatus exists to begin with. The rest are built but
        # unadded, so nothing of the sequence is visible before its reveal.
        self.add(devices[0], components[0])

        electron, dot, spin, label = make_electron(SPIN_ANGLE_DEG * DEGREES)
        electron.set_z_index(1)
        self.place(electron, dot, devices[0].get_center() + LEFT * 4.5)

        self.play(FadeIn(electron, shift=RIGHT * 0.4), run_time=0.6)

        spin_now = SPIN_ANGLE_DEG
        for i, (needle, spin_after, outcome, ket_after) in enumerate(BEATS):
            device = devices[i]
            zoom = frame.width / BASE_WIDTH

            # In, and hidden by the panel.
            self.play(self.travel(electron, dot, device.get_center()), run_time=1.0)

            self.measure(device, needle, spin_now, outcome, zoom)

            # Applied while the electron is out of sight, so it emerges already
            # collapsed -- no animating a collapse nobody could watch.
            spin.rotate((spin_now - spin_after) * DEGREES, about_point=dot.get_center())
            spin_now = spin_after
            label.become(
                MathTex(ket_after, font_size=36, color=GREEN)
                .scale(zoom)
                .move_to(label)
            )

            # Out the far side, parking exactly midway to the next apparatus.
            # A zoom-scaled offset was wrong in both directions: barely clear of
            # the body at the first device, far past it by the last. Half the
            # spacing is the one resting place that reads identically at every
            # zoom level and never overlaps a device.
            self.play(
                self.travel(electron, dot, device.get_center() + RIGHT * SPACING / 2),
                run_time=0.9,
            )
            # A beat to read the reading before anything else moves.
            self.wait(2.0)

            if i + 1 < len(BEATS):
                # Pull back far enough for the next apparatus, and fade it in as
                # we go.
                new_width = (i + 1) * SPACING + 9.0
                ratio = new_width / frame.width
                shown = VGroup(*devices[: i + 2])
                self.play(
                    frame.animate.set(width=new_width).move_to(
                        shown.get_center() + DOWN * 0.45 * (new_width / BASE_WIDTH)
                    ),
                    FadeIn(devices[i + 1]),
                    FadeIn(components[i + 1]),
                    # About the dot, so the electron grows in place rather than
                    # being flung sideways by its own bounding box.
                    electron.animate.scale(ratio, about_point=dot.get_center()),
                    run_time=1.8,
                )

        # Final framing wide enough to actually contain the electron, which sits
        # past the last apparatus and fell outside a devices-only frame.
        everything = VGroup(devices, electron)
        self.play(
            frame.animate.set(width=everything.width + 4).move_to(everything),
            run_time=1.5,
        )
        self.wait(2.0)

    @staticmethod
    def place(electron, dot, target):
        """Put the *dot* at `target`, immediately.

        Never move_to() the electron group: its bounding box also contains the
        spin arrow and the ket label, so the box's centre shifts every time the
        spin turns. Positioning by the group would slide the electron off the
        beam line by a different amount after every collapse. Shifting by the
        dot's own offset is exact and orientation-independent.
        """
        electron.shift(target - dot.get_center())

    @staticmethod
    def travel(electron, dot, target):
        """The same placement, as an animation."""
        return electron.animate.shift(target - dot.get_center())

    def measure(self, device, needle_deg, spin_deg, outcome, zoom):
        """One press of the button: show the decomposition, then collapse it."""
        frame = self.camera.frame
        theta_deg = abs(spin_deg - needle_deg)

        # The incoming spin, drawn on the device's own face beside the needle,
        # with the angle between them marked -- so theta in the panel below is
        # visibly the same angle you can see on the dial.
        overlay = theta_overlay(device, needle_deg, spin_deg).set_z_index(4)

        panel = decomposition_panel(theta_deg).scale(0.85 * zoom)
        # Centred in the frame rather than pinned under its device: as the
        # camera pulls back the device drifts off to one side, and a panel
        # attached to it drifts too and runs off the edge of the screen.
        panel.move_to(frame.get_center() + DOWN * frame.height * 0.30)
        ket, probs = panel

        won, lost = (0, 2) if outcome == "+1" else (2, 0)
        colour = GREEN if outcome == "+1" else RED

        self.play(FadeIn(overlay), FadeIn(panel, shift=UP * 0.2), run_time=0.6)
        self.wait(0.7)

        self.play(
            Flash(
                device.dial,
                color=colour,
                line_length=0.15 * zoom,
                num_lines=12,
                flash_radius=0.6 * zoom,
            ),
            Transform(
                device.reading,
                MathTex(outcome, font_size=42, color=colour).move_to(device.dial),
            ),
            Indicate(probs[won], color=colour, scale_factor=1.15),
            run_time=0.8,
        )
        self.play(
            probs[lost].animate.set_opacity(0.2),
            ket[1 if lost == 0 else 3].animate.set_opacity(0.2),
            run_time=0.35,
        )
        self.wait(0.8)
        self.play(FadeOut(panel), FadeOut(overlay), run_time=0.5)


class DevicePreview(Scene):
    """Layout check for MeasurementDevice alone.

    Plays nothing, so Manim writes a single PNG instead of a video -- which is
    what makes it a fast way to check positions while tweaking the layout.
    """

    def construct(self):
        device = MeasurementDevice()
        self.add(device)


# With the spin prepared along +z and the needle at theta, P(+1) = cos^2(theta/2)
# and <sigma_n> = cos(theta). At 60 degrees those are 0.75 and 0.50 -- round
# enough that a running average visibly lands on them.
FLUX_THETA_DEG = 60
FLUX_COUNT = 90


class ElectronFlux(Scene):
    """The same measurement repeated on an ensemble.

    Every electron is prepared afresh in |u> before it reaches the needle. That
    is not decoration -- it is the whole experiment. Reuse the electron that
    just came out and it is already an eigenstate of sigma_n, so every later
    press returns the same value and the average runs to +-1 instead of
    cos(theta). Preparing, measuring once, and discarding is what makes the
    average mean anything.
    """

    def construct(self):
        theta = FLUX_THETA_DEG * DEGREES
        p_plus = np.cos(theta / 2) ** 2

        # Seeded so the run is reproducible. This particular draw is a typical
        # one rather than a lucky one: it lands at 68/90 = 0.756 against a true
        # 0.750, and is already near it by halfway. A run that ends 1.1 sigma
        # out is perfectly legal but reads as though nothing converged.
        rng = np.random.default_rng(30)
        outcomes = rng.random(FLUX_COUNT) < p_plus

        beam_y = 1.1
        source = self.build_source(LEFT * 5.6 + UP * beam_y)

        device = MeasurementDevice(angle_deg=FLUX_THETA_DEG).scale(0.6)
        device.move_to(LEFT * 1.2 + UP * beam_y)
        device.set_z_index(3)
        overlay = theta_overlay(device, FLUX_THETA_DEG, 0).set_z_index(4)
        # theta_overlay draws at the device's own scale; it was built for a
        # full-size face, so bring it down to match.
        overlay.scale(0.6, about_point=device.pivot)

        plus_lane, minus_lane = UP * 2.5, DOWN * 0.6
        collectors = VGroup(
            self.build_collector(RIGHT * 4.9 + plus_lane, "+1", GREEN),
            self.build_collector(RIGHT * 4.9 + minus_lane, "-1", RED),
        )

        self.add(source, device, overlay, collectors)

        progress = ValueTracker(0)
        readout = self.build_readout(progress, outcomes, theta)
        self.add(readout)

        # Bars are plain rectangles, so redrawing them every frame is cheap.
        bars = always_redraw(lambda: self.build_bars(progress, outcomes))
        self.add(bars)

        flights = []
        for i, plus in enumerate(outcomes):
            electron = Dot(radius=0.07, color=BLUE).move_to(source.get_right())
            electron.set_z_index(2)
            lane = plus_lane if plus else minus_lane
            flights.append(
                Succession(
                    FadeIn(electron, run_time=0.01),
                    electron.animate(run_time=0.55).move_to(device.get_center()),
                    electron.animate(run_time=0.55).move_to(RIGHT * 4.4 + lane),
                    FadeOut(electron, run_time=0.01),
                )
            )

        self.play(
            LaggedStart(*flights, lag_ratio=0.13),
            progress.animate.set_value(FLUX_COUNT),
            run_time=15,
            rate_func=linear,
        )
        self.wait(2.5)

    # -- pieces ---------------------------------------------------------------

    def build_source(self, centre):
        """The preparation stage, as a box that emits a fresh |u> each time."""
        box = RoundedRectangle(
            width=2.0, height=1.5, corner_radius=0.1,
            stroke_color=GREY_B, stroke_width=2,
            fill_color=GREY_E, fill_opacity=1,
        )
        title = Text("prepare", font_size=18, color=GREY_A)
        ket = MathTex(r"|u\rangle", font_size=34, color=GREEN)
        VGroup(title, ket).arrange(DOWN, buff=0.12).move_to(box.get_center())
        return VGroup(box, title, ket).move_to(centre)

    def build_collector(self, centre, label, colour):
        box = RoundedRectangle(
            width=1.1, height=1.1, corner_radius=0.1,
            stroke_color=colour, stroke_width=2,
            fill_color=BLACK, fill_opacity=1,
        )
        text = MathTex(label, font_size=32, color=colour).move_to(box.get_center())
        return VGroup(box, text).move_to(centre)

    def build_bars(self, progress, outcomes):
        landed = int(progress.get_value())
        n_plus = int(outcomes[:landed].sum())
        n_minus = landed - n_plus
        # Sized so a full tally stops short of the frame edge: the bar starts at
        # x = 5.5 and the visible world ends near 7.1.
        unit = 1.4 / max(1, FLUX_COUNT)

        bars = VGroup()
        for count, y, colour in (
            (n_plus, 2.5, GREEN),
            (n_minus, -0.6, RED),
        ):
            width = max(0.001, count * unit)
            bar = Rectangle(
                width=width, height=0.32,
                stroke_width=0, fill_color=colour, fill_opacity=0.85,
            )
            bar.move_to(RIGHT * 5.5 + UP * y, aligned_edge=LEFT)
            bars.add(bar)
        return bars

    def build_readout(self, progress, outcomes, theta):
        """Live tallies.

        DecimalNumber, not MathTex: an updater runs every frame, and MathTex
        would shell out to LaTeX each time. Static labels are MathTex because
        they are built once.
        """
        target_p = np.cos(theta / 2) ** 2
        target_mean = np.cos(theta)

        def value(fn, places=3):
            number = DecimalNumber(0, num_decimal_places=places, font_size=32)
            number.add_updater(
                lambda m: m.set_value(fn(int(progress.get_value())))
            )
            return number

        def fraction(landed):
            return outcomes[:landed].sum() / landed if landed else 0.0

        def mean(landed):
            if not landed:
                return 0.0
            n_plus = int(outcomes[:landed].sum())
            return (2 * n_plus - landed) / landed

        rows = VGroup(
            VGroup(
                MathTex(r"N =", font_size=32, color=GREY_A),
                value(lambda k: k, places=0),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                MathTex(r"\hat{P}(+1) =", font_size=32, color=GREEN),
                value(fraction),
                MathTex(
                    rf"\to \cos^2(\theta/2) = {target_p:.3f}",
                    font_size=30, color=GREY_B,
                ),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                MathTex(r"\langle \sigma_n \rangle =", font_size=32, color=YELLOW),
                value(mean),
                MathTex(
                    rf"\to \cos\theta = {target_mean:.3f}",
                    font_size=30, color=GREY_B,
                ),
            ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        return rows.to_edge(DOWN, buff=0.5)
