#!/usr/bin/env python3

# ROS 2 action server that flies the Swift Pico to a sequence of waypoints.
# The position controller is the same PID controller used for Task 2A; it is
# embedded here so the action server can command the drone directly.

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from waypoint_navigation.action import NavToWaypoint

from swift_msgs.msg import SwiftMsgs
from std_msgs.msg import Header
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from controller_msg.msg import PIDTune


BASE_RC = 1500.0


def _clamp(value, low, high):
    return max(low, min(high, value))


def _slew(current, target, max_step):
    if target > current:
        return min(target, current + max_step)
    return max(target, current - max_step)


class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')

        self.pid_or_lqr_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        # --- action / settling state ---------------------------------
        self.state_lock = threading.Lock()
        # Task 3A saving maneuver: hold at Z = 13 for 1 s, then descend and
        # hold at Z = 14 for 2 s.  `hold_time` is kept as the visit hold.
        self.hold_time = float(self.declare_parameter(
            'hold_time', 1.0).value)
        self.save_hold_time = float(self.declare_parameter(
            'save_hold_time', 2.0).value)
        self.home_hold_time = float(self.declare_parameter(
            'home_hold_time', 1.0).value)
        self.whycode_tolerance = float(self.declare_parameter(
            'whycode_tolerance', 0.4).value)
        self.via_tolerance = float(self.declare_parameter(
            'via_tolerance', 0.5).value)
        self.via_vertical_tolerance = float(self.declare_parameter(
            'via_vertical_tolerance', 1.25).value)
        self.goal_timeout = float(self.declare_parameter(
            'goal_timeout', 35.0).value)
        # Ramp altitude changes instead of commanding a step; a step makes
        # the altitude loop overshoot the Save sphere and delays the hold.
        self.vertical_slew_rate = float(self.declare_parameter(
            'vertical_slew_rate', 0.6).value)
        # Extra damping inside the last few tenths of a Z move: the altitude
        # loop is slow to reverse, so without it the Save descent sails
        # through Z=14 and the 2 s hold restarts.  Damping is only raised once
        # the remaining error is small, so the long descent still runs at
        # normal speed.
        self.vertical_near_radius = float(self.declare_parameter(
            'vertical_near_radius', 0.55).value)
        self.vertical_near_damping = float(self.declare_parameter(
            'vertical_near_damping', 3.0).value)
        self.hold_z_offset = float(self.declare_parameter(
            'hold_z_offset', 0.15).value)

        # --- F6 landing ----------------------------------------------
        self.f6_radius = float(self.declare_parameter(
            'f6_radius', 0.7).value)
        self.ground_z = float(self.declare_parameter(
            'ground_z', 14.95).value)
        self.landing_descent_rate = float(self.declare_parameter(
            'landing_descent_rate', 0.5).value)
        self.landing_timeout = float(self.declare_parameter(
            'landing_timeout', 20.0).value)
        self.contact_clearance = float(self.declare_parameter(
            'contact_clearance', 0.35).value)
        self.contact_velocity = float(self.declare_parameter(
            'contact_velocity', 0.30).value)
        self.landing_overshoot = float(self.declare_parameter(
            'landing_overshoot', 0.50).value)
        self.landed = False

        # --- controller configuration --------------------------------
        self.sample_time = 1.0 / 30.0
        self.airborne_z_start = float(self.declare_parameter(
            'airborne_z_start', 14.6).value)
        self.airborne_z_full = float(self.declare_parameter(
            'airborne_z_full', 14.2).value)
        self.marker_timeout = float(self.declare_parameter(
            'marker_timeout', 0.5).value)

        self.max_tilt = float(self.declare_parameter(
            'max_tilt_counts', 100.0).value)
        self.throttle_min = int(self.declare_parameter(
            'throttle_min', 1400).value)
        self.throttle_max = int(self.declare_parameter(
            'throttle_max', 1600).value)
        self.hover_throttle = float(self.declare_parameter(
            'hover_throttle', 1500).value)
        self.altitude_ceiling_margin = float(self.declare_parameter(
            'altitude_ceiling_margin', 1.6).value)

        self.max_tilt_step = float(self.declare_parameter(
            'max_tilt_step', 5.0).value)
        self.max_throttle_step = float(self.declare_parameter(
            'max_throttle_step', 15.0).value)

        self.integral_limit = float(self.declare_parameter(
            'integral_limit', 8.0).value)

        # Horizontal gain scheduling: relax damping during the long transit
        # and increase it close to the waypoint so the drone brakes hard and
        # does not overshoot.  Only hover goals use the near schedule: a via
        # is flown straight through, so braking for it would waste several
        # seconds at every corner of the visibility graph.
        self.brake_radius = float(self.declare_parameter(
            'brake_radius', 2.5).value)
        self.kd_far_scale = float(self.declare_parameter(
            'kd_far_scale', 0.7).value)
        self.kd_near_scale = float(self.declare_parameter(
            'kd_near_scale', 3.0).value)
        self.kp_near_scale = float(self.declare_parameter(
            'kp_near_scale', 3.0).value)
        # Vias are flown through: brake later and less so the corner still
        # slows the drone enough to turn, but it never stops at the point.
        self.via_brake_radius = float(self.declare_parameter(
            'via_brake_radius', 1.4).value)
        self.via_kd_near_scale = float(self.declare_parameter(
            'via_kd_near_scale', 2.5).value)
        self.via_kp_near_scale = float(self.declare_parameter(
            'via_kp_near_scale', 2.0).value)
        self.pos_tau = float(self.declare_parameter(
            'position_filter_tau', 0.08).value)
        self.vel_tau = float(self.declare_parameter(
            'velocity_filter_tau', 0.10).value)

        self.x_axis = str(self.declare_parameter('x_axis', 'pitch').value).lower()
        self.y_axis = str(self.declare_parameter('y_axis', 'roll').value).lower()
        if {self.x_axis, self.y_axis} != {'pitch', 'roll'}:
            raise ValueError("x_axis and y_axis must be 'pitch' and 'roll'")

        self.pitch_sign = float(self.declare_parameter('pitch_sign', 1.0).value)
        self.roll_sign = float(self.declare_parameter('roll_sign', 1.0).value)
        self.throttle_sign = float(self.declare_parameter(
            'throttle_sign', -1.0).value)

        self.gains = {
            'pitch': self._read_gains('pitch_gains', [25.0, 1.5, 45.0]),
            'roll': self._read_gains('roll_gains', [25.0, 1.5, 45.0]),
            'throttle': self._read_gains(
                'throttle_gains', [35.0, 8.0, 55.0]),
        }

        # --- yaw configuration ---------------------------------------
        self.yaw = 0.0
        self.desired_yaw = float(self.declare_parameter(
            'desired_yaw', -0.5).value)
        # Heading hold is off by default.  The RC yaw channel is rate based and
        # the only heading feedback comes from the flat marker, which is a poor
        # orientation estimate while the drone is tilted hard in a corner: it
        # made the controller chase a phantom yaw and spin the drone, which
        # then rotated the body axes and sent it flying past the waypoint.
        # Nothing in Task 3A needs a particular heading, so the yaw channel is
        # left centred and the drone keeps whatever heading it has.
        self.yaw_kp = float(self.declare_parameter('yaw_kp', 0.0).value)
        self.yaw_kd = float(self.declare_parameter('yaw_kd', 0.0).value)
        self.yaw_deadband = float(self.declare_parameter(
            'yaw_deadband', 6.0).value)
        self.yaw_rate = 0.0
        self.last_yaw_time = 0.0

        # --- state ---------------------------------------------------
        self.curr_state = [0.0, 0.0, 0.0]
        self.drone_position = [0.0, 0.0, 0.0]
        self.filtered_state = None
        self.velocity = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.desired_state = [0.0, 0.0, 13.0]
        # True while the active goal is a hover hold; via goals set this False
        # so the controller does not brake for a point it only flies past.
        self.goal_hold = True
        self.last_marker_time = 0.0
        self.last_pid_time = 0.0
        self.dtime = 0

        self.command_roll = BASE_RC
        self.command_pitch = BASE_RC
        self.command_throttle = self.hover_throttle
        self.yaw_command = BASE_RC

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = int(BASE_RC)
        self.cmd.rc_pitch = int(BASE_RC)
        self.cmd.rc_yaw = int(BASE_RC)
        self.cmd.rc_throttle = int(self.hover_throttle)
        self.cmd.rc_aux4 = 2000

        self.pos_error = Error()

        # --- ROS interfaces -----------------------------------------
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        self.create_subscription(
            MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
        self.create_subscription(
            PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(
            PIDTune, '/roll_pid', self.roll_set_pid, 1)
        self.create_subscription(
            PIDTune, '/throttle_pid', self.altitude_set_pid, 1)

        self.action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group,
        )

        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        time.sleep(0.5)
        self.arm()

        self.timer = self.create_timer(
            self.sample_time, self.pid,
            callback_group=self.pid_or_lqr_callback_group)

        self.get_logger().info('Waypoint action server ready')

    # ------------------------------------------------------------------
    # Gains / heartbeat / arming
    # ------------------------------------------------------------------
    def _read_gains(self, name, default):
        values = list(self.declare_parameter(name, default).value)
        if len(values) != 3:
            raise ValueError('%s must contain [Kp, Ki, Kd]' % name)
        return [float(v) for v in values]

    def pitch_set_pid(self, tune):
        self.gains['pitch'] = [float(tune.kp), float(tune.ki), float(tune.kd)]

    def roll_set_pid(self, tune):
        self.gains['roll'] = [float(tune.kp), float(tune.ki), float(tune.kd)]

    def altitude_set_pid(self, tune):
        self.gains['throttle'] = [float(tune.kp), float(tune.ki), float(tune.kd)]

    def publish_heartbeat(self):
        if not rclpy.ok():
            return
        hb = Header()
        hb.stamp = self.get_clock().now().to_msg()
        hb.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(hb)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = int(BASE_RC)
        self.cmd.rc_yaw = int(BASE_RC)
        self.cmd.rc_pitch = int(BASE_RC)
        self.cmd.rc_throttle = int(self.hover_throttle)
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    # ------------------------------------------------------------------
    # WhyCode feedback
    # ------------------------------------------------------------------
    def whycode_callback(self, msg):
        if not msg.markers:
            return

        marker = msg.markers[0]
        position = marker.position.position
        pos = [float(position.x), float(position.y), float(position.z)]
        if not all(math.isfinite(v) for v in pos):
            return

        now = time.monotonic()

        if self.filtered_state is None or self.last_marker_time == 0.0:
            self.curr_state = list(pos)
            self.drone_position = list(pos)
            self.filtered_state = list(pos)
            self.velocity = [0.0, 0.0, 0.0]
            self.last_marker_time = now
        else:
            dt = now - self.last_marker_time
            if dt > 1e-6:
                dt = min(dt, 0.5)
                alpha_pos = 1.0 - math.exp(-dt / max(self.pos_tau, 1e-3))
                alpha_vel = 1.0 - math.exp(-dt / max(self.vel_tau, 1e-3))
                for i in range(3):
                    raw_velocity = (pos[i] - self.curr_state[i]) / dt
                    self.velocity[i] += alpha_vel * (
                        raw_velocity - self.velocity[i])
                    self.filtered_state[i] += alpha_pos * (
                        pos[i] - self.filtered_state[i])
                self.curr_state = list(pos)
                self.drone_position = list(pos)
                self.last_marker_time = now

        # Marker orientation is reported in radians; yaw control works in deg.
        yaw = math.degrees(marker.rotation.z)
        if self.last_yaw_time > 0.0:
            dyaw = yaw - self.yaw
            while dyaw > 180.0:
                dyaw -= 360.0
            while dyaw < -180.0:
                dyaw += 360.0
            dt_yaw = max(now - self.last_yaw_time, 1e-3)
            self.yaw_rate += 0.5 * ((dyaw / dt_yaw) - self.yaw_rate)
        self.yaw = yaw
        self.last_yaw_time = now
        self.dtime = msg.header.stamp.sec

    # ------------------------------------------------------------------
    # PID position controller (+ yaw hold)
    # ------------------------------------------------------------------
    def pid(self):
        if not rclpy.ok():
            return
        if self.landed:
            # The landing routine already disarmed the motors.  Do not let the
            # periodic controller spin them back up.
            return
        now = time.monotonic()

        if self.last_marker_time == 0.0 or \
                (now - self.last_marker_time) > self.marker_timeout:
            self.reset_control()
            self.publish_targets(BASE_RC, BASE_RC, self.hover_throttle)
            return

        if self.last_pid_time == 0.0:
            dt = self.sample_time
        else:
            dt = _clamp(now - self.last_pid_time, 1e-3, 0.2)
        self.last_pid_time = now

        with self.state_lock:
            desired = list(self.desired_state)
            hold_goal = self.goal_hold

        z = self.filtered_state[2]

        span = self.airborne_z_start - self.airborne_z_full
        if span <= 1e-6:
            gate = 1.0 if z <= self.airborne_z_start else 0.0
        else:
            gate = (self.airborne_z_start - z) / span
        gate = _clamp(gate, 0.0, 1.0)

        axis_for_index = (self.x_axis, self.y_axis, 'throttle')
        error = [desired[i] - self.filtered_state[i] for i in range(3)]
        output = [0.0, 0.0, 0.0]

        e_h = math.hypot(error[0], error[1])
        if hold_goal:
            radius = self.brake_radius
            kd_near = self.kd_near_scale
            kp_near = self.kp_near_scale
        else:
            radius = self.via_brake_radius
            kd_near = self.via_kd_near_scale
            kp_near = self.via_kp_near_scale
        blend = _clamp(
            (radius - e_h) / max(radius, 1e-6), 0.0, 1.0)
        kd_scale = self.kd_far_scale + (
            kd_near - self.kd_far_scale) * blend
        kp_scale = 1.0 + (kp_near - 1.0) * blend

        for i in range(3):
            kp, ki, kd = self.gains[axis_for_index[i]]
            is_horizontal = i < 2
            if is_horizontal:
                kp = kp * kp_scale
                kd = kd * kd_scale
            elif (hold_goal
                  and abs(error[i]) < self.vertical_near_radius):
                kd = kd * self.vertical_near_damping

            u = kp * error[i] + ki * self.error_sum[i] - kd * self.velocity[i]

            integrate = True
            if is_horizontal:
                if gate <= 0.0:
                    integrate = False
                elif u >= self.max_tilt and error[i] > 0.0:
                    integrate = False
                elif u <= -self.max_tilt and error[i] < 0.0:
                    integrate = False
            else:
                throttle_preview = self.hover_throttle + self.throttle_sign * u
                rate_sign = self.throttle_sign * error[i]
                if throttle_preview >= self.throttle_max and rate_sign > 0.0:
                    integrate = False
                elif throttle_preview <= self.throttle_min and rate_sign < 0.0:
                    integrate = False

            if integrate:
                self.error_sum[i] += error[i] * dt * (
                    gate if is_horizontal else 1.0)
                self.error_sum[i] = _clamp(
                    self.error_sum[i], -self.integral_limit, self.integral_limit)

            u = kp * error[i] + ki * self.error_sum[i] - kd * self.velocity[i]

            if is_horizontal:
                # Keep the raw horizontal command here and limit the combined
                # tilt further down.  Clamping pitch and roll independently
                # biases a saturated diagonal command towards 45 degrees,
                # which makes long diagonal transits drift sideways.
                u = gate * u
            output[i] = u

        horizontal = math.hypot(output[0], output[1])
        if horizontal > self.max_tilt:
            scale = self.max_tilt / horizontal
            output[0] *= scale
            output[1] *= scale

        pitch_input = output[0] if self.x_axis == 'pitch' else output[1]
        roll_input = output[0] if self.x_axis == 'roll' else output[1]

        pitch_target = BASE_RC + self.pitch_sign * pitch_input
        roll_target = BASE_RC + self.roll_sign * roll_input
        throttle_target = self.hover_throttle + self.throttle_sign * output[2]

        if z < (desired[2] - self.altitude_ceiling_margin):
            throttle_target = self.throttle_min
            self.error_sum[2] = min(self.error_sum[2], 0.0)

        self.update_yaw_command()

        self.publish_targets(roll_target, pitch_target, throttle_target)

        self.pos_error.pitch_error = float(
            error[0] if self.x_axis == 'pitch' else error[1])
        self.pos_error.roll_error = float(
            error[0] if self.x_axis == 'roll' else error[1])
        self.pos_error.throttle_error = float(error[2])
        self.pos_error.yaw_error = float(self.desired_yaw - self.yaw)
        self.pos_error_pub.publish(self.pos_error)

    def update_yaw_command(self):
        if self.yaw_kp <= 0.0 and self.yaw_kd <= 0.0:
            # Heading hold disabled: leave the rate channel centred.
            self.yaw_command = BASE_RC
            return
        # The RC yaw channel is rate based (1499/1501 -> +/- max yaw rate), so
        # a proportional term is applied only outside a small deadband.
        yaw_error = self.desired_yaw - self.yaw
        while yaw_error > 180.0:
            yaw_error -= 360.0
        while yaw_error < -180.0:
            yaw_error += 360.0

        rate_ff = -self.yaw_kd * self.yaw_rate
        control = self.yaw_kp * yaw_error + rate_ff

        if abs(yaw_error) <= self.yaw_deadband:
            self.yaw_command = BASE_RC
        elif control > 0.0:
            self.yaw_command = BASE_RC + 1
        else:
            self.yaw_command = BASE_RC - 1

    def reset_control(self):
        self.error_sum = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.filtered_state = None
        self.last_pid_time = 0.0

    def publish_targets(self, roll, pitch, throttle):
        if not rclpy.ok():
            return
        roll = _clamp(roll, BASE_RC - self.max_tilt, BASE_RC + self.max_tilt)
        pitch = _clamp(pitch, BASE_RC - self.max_tilt, BASE_RC + self.max_tilt)
        throttle = _clamp(throttle, self.throttle_min, self.throttle_max)

        self.command_roll = _slew(self.command_roll, roll, self.max_tilt_step)
        self.command_pitch = _slew(self.command_pitch, pitch, self.max_tilt_step)
        self.command_throttle = _slew(
            self.command_throttle, throttle, self.max_throttle_step)

        self.cmd.rc_roll = int(round(self.command_roll))
        self.cmd.rc_pitch = int(round(self.command_pitch))
        self.cmd.rc_throttle = int(round(self.command_throttle))
        self.cmd.rc_yaw = int(self.yaw_command)
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    # ------------------------------------------------------------------
    # Action server
    # ------------------------------------------------------------------
    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):

        target = [
            float(goal_handle.request.waypoint.position.x),
            float(goal_handle.request.waypoint.position.y),
            float(goal_handle.request.waypoint.position.z),
        ]
        hover = bool(goal_handle.request.hover)
        near_home = math.hypot(target[0], target[1]) <= self.f6_radius

        if self.landed:
            # A new goal after a completed landing means the caller wants to
            # fly again; re-arm before accepting it.
            self.landed = False
            self.arm()

        if not hover:
            required_hold = 0.0
        elif near_home:
            required_hold = self.home_hold_time
        elif target[2] >= 13.5:
            required_hold = self.save_hold_time
        else:
            required_hold = self.hold_time

        tolerance = (
            self.whycode_tolerance if required_hold > 0.0
            else self.via_tolerance)
        self.get_logger().info(
            'Executing goal to waypoint: %s hover=%s hold=%.1fs%s' % (
                target, hover, required_hold,
                ' (HOME: land after hold)' if near_home and hover else ''))

        with self.state_lock:
            current_z = (self.filtered_state[2]
                         if self.filtered_state is not None
                         else target[2])
            self.desired_state = [target[0], target[1], current_z]
            self.goal_hold = hover
            # Keep the altitude integrator across goals: it carries the
            # hover-throttle trim, and rebuilding it from zero at every hold
            # left the drone about 0.4 units off the hold point.
            self.error_sum[0] = 0.0
            self.error_sum[1] = 0.0
            self.velocity = [0.0, 0.0, 0.0]

        start_time = time.monotonic()
        last_ramp_time = start_time
        settle_start = None
        settle_time = 0.0
        z_locked = False

        feedback_msg = NavToWaypoint.Feedback()

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                # Task 3A: stop the goal and hold the drone where it is until
                # the next goal arrives.
                current = list(
                    self.filtered_state if self.filtered_state is not None
                    else self.drone_position)
                with self.state_lock:
                    self.desired_state = list(current)
                    self.error_sum[0] = 0.0
                    self.error_sum[1] = 0.0
                self.get_logger().info(
                    'Goal canceled; holding at (%.3f, %.3f, %.3f)' % tuple(
                        current))
                goal_handle.canceled()
                result = NavToWaypoint.Result()
                result.hov_time = int(round(time.monotonic() - start_time))
                return result

            now = time.monotonic()
            dt_goal = _clamp(now - last_ramp_time, 1e-3, 0.2)
            last_ramp_time = now
            if not z_locked:
                with self.state_lock:
                    desired = list(self.desired_state)
                    z_error = target[2] - desired[2]
                    step = self.vertical_slew_rate * dt_goal
                    if abs(z_error) <= step:
                        desired[2] = target[2]
                    else:
                        desired[2] += math.copysign(step, z_error)
                    self.desired_state = desired

            pos = list(
                self.filtered_state if self.filtered_state is not None
                else self.drone_position)

            dx = pos[0] - target[0]
            dy = pos[1] - target[1]
            dz = pos[2] - target[2]
            if required_hold > 0.0:
                # Holds are judged in the full 0.4 sphere.
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                in_sphere = distance <= tolerance
            else:
                # A via is only a point on the route, flown past without a
                # stop.  Its job is lateral: requiring the altitude to have
                # finished its ramp as well used to add several seconds to
                # every leg after a Save.
                distance = math.hypot(dx, dy)
                in_sphere = (distance <= tolerance
                             and abs(dz) <= self.via_vertical_tolerance)

            if in_sphere:
                if settle_start is None:
                    settle_start = now
                settle_time = now - settle_start
                if required_hold > 0.0 and not z_locked:
                    # Stop the altitude move just inside the sphere instead
                    # of letting the slow altitude loop sail through the far
                    # side.  Keep a small offset from the centre so the hold
                    # is not sitting on the sphere boundary.
                    approach = target[2] - pos[2]
                    if abs(approach) < 1e-3:
                        approach = target[2] - self.desired_state[2]
                    if abs(approach) < 1e-3:
                        approach = 1.0
                    locked_z = (target[2]
                                - math.copysign(self.hold_z_offset, approach))
                    with self.state_lock:
                        desired = list(self.desired_state)
                        desired[2] = locked_z
                        self.desired_state = desired
                    z_locked = True
            else:
                if settle_start is not None and settle_time > 0.15:
                    self.get_logger().warn(
                        'Hold reset at %s: pos=(%.2f, %.2f, %.2f) '
                        'distance=%.2f' % (
                            target, pos[0], pos[1], pos[2], distance))
                settle_start = None
                settle_time = 0.0

            feedback_msg.current_waypoint.header.stamp.sec = int(settle_time)
            feedback_msg.current_waypoint.header.stamp.nanosec = int(
                (settle_time % 1.0) * 1e9)
            feedback_msg.current_waypoint.header.frame_id = 'whycode'
            feedback_msg.current_waypoint.pose.position.x = pos[0]
            feedback_msg.current_waypoint.pose.position.y = pos[1]
            feedback_msg.current_waypoint.pose.position.z = pos[2]
            feedback_msg.current_waypoint.pose.orientation.w = 1.0
            goal_handle.publish_feedback(feedback_msg)

            if in_sphere and settle_time >= required_hold:
                break

            if (now - start_time) > self.goal_timeout:
                self.get_logger().warn(
                    'Timed out waiting to settle at %s' % target)
                goal_handle.abort()
                result = NavToWaypoint.Result()
                result.hov_time = int(round(now - start_time))
                return result

            time.sleep(self.sample_time)

        if near_home and hover:
            return self.land_at_f6(
                goal_handle, target, start_time, feedback_msg)

        goal_handle.succeed()

        result = NavToWaypoint.Result()
        result.hov_time = int(round(time.monotonic() - start_time))
        self.get_logger().info(
            'Waypoint %s reached, settle time %d s' % (target, result.hov_time))
        return result

    def land_at_f6(self, goal_handle, target, start_time, feedback_msg):
        """Descend slowly from the F6 hover and settle without bouncing."""
        self.get_logger().info(
            'F6 hover complete; descending at %.2f units/s'
            % self.landing_descent_rate)

        landing_start = time.monotonic()
        last_time = landing_start
        contact_start = None

        with self.state_lock:
            self.desired_state = [
                target[0], target[1], self.drone_position[2]]
            self.error_sum = [0.0, 0.0, 0.0]

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                current = list(self.drone_position)
                with self.state_lock:
                    self.desired_state = list(current)
                goal_handle.canceled()
                result = NavToWaypoint.Result()
                result.hov_time = int(round(time.monotonic() - start_time))
                return result

            now = time.monotonic()
            dt = _clamp(now - last_time, 1e-3, 0.2)
            last_time = now

            with self.state_lock:
                desired = list(self.desired_state)
                desired[2] = min(
                    self.ground_z + self.landing_overshoot,
                    desired[2] + self.landing_descent_rate * dt)
                self.desired_state = desired

            pos = list(self.drone_position)
            vertical_speed = abs(self.velocity[2])

            on_ground = (
                pos[2] >= self.ground_z - self.contact_clearance
                and vertical_speed <= self.contact_velocity)
            if on_ground:
                if contact_start is None:
                    contact_start = now
            else:
                contact_start = None

            feedback_msg.current_waypoint.header.stamp.sec = int(
                now - start_time)
            feedback_msg.current_waypoint.header.frame_id = 'whycode'
            feedback_msg.current_waypoint.pose.position.x = pos[0]
            feedback_msg.current_waypoint.pose.position.y = pos[1]
            feedback_msg.current_waypoint.pose.position.z = pos[2]
            feedback_msg.current_waypoint.pose.orientation.w = 1.0
            goal_handle.publish_feedback(feedback_msg)

            if contact_start is not None and (now - contact_start) >= 0.4:
                break
            if (now - landing_start) > self.landing_timeout:
                self.get_logger().warn(
                    'Landing timeout at z=%.3f; disarming' % pos[2])
                break

            time.sleep(self.sample_time)

        with self.state_lock:
            self.desired_state = [target[0], target[1], self.ground_z]
        self.landed = True
        # Let any controller callback that was already in flight finish, then
        # send the disarm command last so the motors cannot spin back up.
        time.sleep(2.0 * self.sample_time)
        self.disarm()
        self.get_logger().info('Landed at F6; motors disarmed')

        goal_handle.succeed()
        result = NavToWaypoint.Result()
        result.hov_time = int(round(time.monotonic() - start_time))
        return result


def main(args=None):
    rclpy.init(args=args)

    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

