#!/usr/bin/env python3

"""Position-hold PID controller for the Swift Pico.

Publishes /drone_command and /pos_error, reads the marker from /whycode_node/markers,
and accepts live gain updates on /pitch_pid, /roll_pid and /throttle_pid.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from controller_msg.msg import PIDTune
from error_msg.msg import Error
from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')

        # Position is [x, y, z] in WhyCode units; z is measured from the camera.
        self.current_state = [0.0, 0.0, 0.0]
        self.desired_state = [1.5, -1.5, 13.0]
        self.filtered_state = None
        self.velocity = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.derivative = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.last_output = [0.0, 0.0, 0.0]
        self.last_marker_time = 0.0
        self.have_error = False

        # Gains are ordered [pitch/x, roll/y, throttle/z] and are the tuned final values.
        self.Kp = [60.0, 60.0, 50.0]
        self.Ki = [4.0, 4.0, 10.0]
        self.Kd = [150.0, 150.0, 115.0]

        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]
        self.sum_limit = 3.0
        self.pos_alpha = 0.4
        self.deriv_alpha = 0.2
        self.airborne_z = 14.6

        self.sample_time = 0.033

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000

        self.pos_error = Error()

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)

        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        time.sleep(0.5)
        self.arm()

        self.create_timer(self.sample_time, self.pid)

    def publish_heartbeat(self):
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
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def whycode_callback(self, msg):
        if not msg.markers:
            return
        position = msg.markers[0].position.position
        self.current_state[0] = position.x
        self.current_state[1] = position.y
        self.current_state[2] = position.z
        self.last_marker_time = time.monotonic()

    def pitch_set_pid(self, tune):
        self.Kp[0] = tune.kp
        self.Ki[0] = tune.ki
        self.Kd[0] = tune.kd

    def roll_set_pid(self, tune):
        self.Kp[1] = tune.kp
        self.Ki[1] = tune.ki
        self.Kd[1] = tune.kd

    def altitude_set_pid(self, tune):
        self.Kp[2] = tune.kp
        self.Ki[2] = tune.ki
        self.Kd[2] = tune.kd

    def pid(self):
        now = time.monotonic()
        if self.last_marker_time == 0.0 or now - self.last_marker_time > 0.5:
            self.error_sum = [0.0, 0.0, 0.0]
            self.derivative = [0.0, 0.0, 0.0]
            self.filtered_state = None
            self.have_error = False
            self.hold_neutral()
            return

        # Low-pass the marker and derive a damped velocity estimate.
        if self.filtered_state is None:
            self.filtered_state = list(self.current_state)
        for i in range(3):
            self.filtered_state[i] += self.pos_alpha * (
                self.current_state[i] - self.filtered_state[i])

        error = [0.0, 0.0, 0.0]
        raw_derivative = [0.0, 0.0, 0.0]
        for i in range(3):
            error[i] = self.desired_state[i] - self.filtered_state[i]
            if self.have_error:
                raw_derivative[i] = (error[i] - self.prev_error[i]) / self.sample_time
                self.derivative[i] += self.deriv_alpha * (
                    raw_derivative[i] - self.derivative[i])
        self.have_error = True

        output = [0.0, 0.0, 0.0]
        for i in range(3):
            if abs(self.last_output[i]) < 400.0 or error[i] * self.last_output[i] < 0.0:
                self.error_sum[i] += error[i] * self.sample_time
            self.error_sum[i] = max(-self.sum_limit, min(self.sum_limit, self.error_sum[i]))
            output[i] = (self.Kp[i] * error[i]
                         + self.Ki[i] * self.error_sum[i]
                         + self.Kd[i] * self.derivative[i])
            self.prev_error[i] = error[i]
            self.last_output[i] = output[i]

        # Hold x/y until the drone has cleared the floor, then blend into full control.
        if self.filtered_state[2] > self.airborne_z:
            output[0] = 0.0
            output[1] = 0.0

        self.cmd.rc_pitch = self.clamp(1500.0 + output[0], 0)
        self.cmd.rc_roll = self.clamp(1500.0 + output[1], 1)
        self.cmd.rc_throttle = self.clamp(1500.0 - output[2], 2)
        self.cmd.rc_yaw = 1500

        self.publish_command()
        self.pos_error.pitch_error = float(error[0])
        self.pos_error.roll_error = float(error[1])
        self.pos_error.throttle_error = float(error[2])
        self.pos_error.yaw_error = 0.0
        self.pos_error_pub.publish(self.pos_error)

    def clamp(self, value, axis):
        return int(max(self.min_values[axis], min(self.max_values[axis], value)))

    def publish_command(self):
        self.command_pub.publish(self.cmd)

    def hold_neutral(self):
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_aux4 = 2000
        self.publish_command()


def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()

    try:
        rclpy.spin(swift_pico)
    except KeyboardInterrupt:
        swift_pico.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        swift_pico.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
