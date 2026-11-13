#!/usr/bin/env python3

# Task 3A action client.
#
# The client fetches the critical route, flies it goal by goal, and watches a
# 55-second window that starts at take-off.  When the third critical Save is
# complete, or the window expires, it cancels the goal in flight, asks the
# re-plan service for a stable route from the drone's actual position, and
# then flies that route to F6, where the server lands.

import math
import time
from dataclasses import dataclass

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node

from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints, PlanStableRoute

VISIT_Z = 13.0
SAVE_Z = 14.0


@dataclass
class GoalSpec:
    x: float
    y: float
    z: float
    hover: bool
    kind: str          # via | visit | save | home
    marker: str
    phase: str = 'critical'


class WayPointClient(Node):

    def __init__(self):
        super().__init__('waypoint_client')

        self.critical_window = float(self.declare_parameter(
            'critical_window', 55.0).value)
        self.goal_log_interval = 2.0

        self.action_client = ActionClient(
            self, NavToWaypoint, 'waypoint_navigation')
        self.route_client = self.create_client(GetWaypoints, 'waypoints')
        self.stable_client = self.create_client(
            PlanStableRoute, 'plan_stable_route')

        # Mission state.
        self.phase = 'idle'          # idle | critical | switching | stable | done
        self.goals = []
        self.goal_index = 0
        self.takeoff_time = None
        self.critical_saves = 0
        self.last_feedback = None
        self.last_feedback_time = 0.0

        self.goal_waiting = False
        self.active_handle = None
        self.active_spec = None
        self.retry_counts = {}
        self.goal_serial = 0

        self.cancel_sent = False
        self.cancel_accepted = False
        self.switch_time = 0.0

        self.replan_requested = False
        self.replan_attempts = 0
        self.last_replan_failure = 0.0

        self.last_not_ready_log = 0.0
        self.last_status_log = 0.0

        self.control_timer = None
        self.get_logger().info('Waypoint client created')

    # ------------------------------------------------------------------
    # Mission start
    # ------------------------------------------------------------------
    def start_mission(self):
        while not self.route_client.service_is_ready() and rclpy.ok():
            self.get_logger().info(
                'waypoints service not available, waiting again...')
            rclpy.spin_once(self, timeout_sec=0.5)

        if not rclpy.ok():
            return False

        request = GetWaypoints.Request()
        request.get_waypoints = True
        future = self.route_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None or not response.waypoints.poses:
            self.get_logger().error(
                'GetWaypoints returned no critical route; cannot start')
            return False

        self.get_logger().info(
            'Critical route received: %d poses, markers=%s'
            % (len(response.waypoints.poses), list(response.marker_ids)))

        self.goals = self._build_goals(response, 'critical')
        if not self.goals:
            self.get_logger().error('Critical route had no usable poses')
            return False

        self.phase = 'critical'
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            'Waiting for the action server before take-off; the %d s '
            'critical window starts at lift-off' % int(self.critical_window))
        return True

    def _build_goals(self, response, phase):
        goals = []
        marker_ids = list(response.marker_ids)

        for index, pose in enumerate(response.waypoints.poses):
            marker = marker_ids[index] if index < len(marker_ids) else ''
            x = float(pose.position.x)
            y = float(pose.position.y)

            if marker.startswith('via_'):
                goals.append(
                    GoalSpec(x, y, VISIT_Z, False, 'via', marker, phase))
            elif marker == 'F6':
                goals.append(
                    GoalSpec(x, y, VISIT_Z, True, 'home', marker, phase))
            else:
                goals.append(
                    GoalSpec(x, y, VISIT_Z, True, 'visit', marker, phase))
                goals.append(
                    GoalSpec(x, y, SAVE_Z, True, 'save', marker, phase))

        for index, goal in enumerate(goals, start=1):
            self.get_logger().info(
                'Goal %d/%d: %s %s hover=%s (%.2f, %.2f, %.2f)' % (
                    index, len(goals), goal.kind, goal.marker,
                    goal.hover, goal.x, goal.y, goal.z))
        return goals

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def control_loop(self):
        if not rclpy.ok():
            return

        now = time.monotonic()

        if self.phase == 'critical':
            if self.takeoff_time is not None and (
                    now - self.takeoff_time) >= self.critical_window:
                self.get_logger().info(
                    'Critical window expired after %.1f s; starting Switch'
                    % (now - self.takeoff_time))
                self._begin_switch()
                return

        if self.phase in ('critical', 'stable'):
            if self.goal_waiting:
                return
            if self.goal_index >= len(self.goals):
                if self.phase == 'critical':
                    self._begin_switch()
                else:
                    self.phase = 'done'
                    self.get_logger().info('Stable route complete')
                return
            if not self.action_client.server_is_ready():
                if now - self.last_not_ready_log >= self.goal_log_interval:
                    self.last_not_ready_log = now
                    self.get_logger().info(
                        'action server not available, waiting...')
                return
            self._send_goal(self.goals[self.goal_index])
            return

        if self.phase == 'switching':
            self._control_switch(now)
            return

    def _send_goal(self, spec):
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = float(spec.x)
        goal_msg.waypoint.position.y = float(spec.y)
        goal_msg.waypoint.position.z = float(spec.z)
        goal_msg.waypoint.orientation.w = 1.0
        goal_msg.hover = bool(spec.hover)

        self.goal_waiting = True
        self.active_spec = spec
        self.active_handle = None
        self.goal_serial += 1
        serial = self.goal_serial

        if self.takeoff_time is None and self.phase == 'critical':
            # The action server lifts off as soon as the goal is accepted.  We
            # start the clock here, after waiting for the server, and never at
            # client launch.
            self.takeoff_time = time.monotonic()
            self.get_logger().info(
                'Critical window started (%.0f s) at first goal send'
                % self.critical_window)

        self.get_logger().info(
            'Sending %s %s goal (%d/%d): [%.2f, %.2f, %.2f] hover=%s' % (
                self.phase, spec.kind, self.goal_index + 1, len(self.goals),
                spec.x, spec.y, spec.z, spec.hover))

        future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        future.add_done_callback(
            lambda result, serial=serial, spec=spec:
            self.goal_response_callback(result, spec, serial))

    def goal_response_callback(self, future, spec, serial):
        if serial != self.goal_serial:
            self.get_logger().warn('Ignoring a response from a stale goal')
            return

        try:
            goal_handle = future.result()
        except Exception as error:
            goal_handle = None
            self.get_logger().error('Goal send failed: %s' % error)

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(
                'Goal rejected by the action server: %s' % spec.kind)
            self.goal_waiting = False
            self.active_handle = None
            self._retry_or_advance('rejected', spec)
            return

        self.active_handle = goal_handle

        self.get_logger().info('Goal accepted by the action server')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, serial=serial, spec=spec:
            self.get_result_callback(result, spec, serial))

    def get_result_callback(self, future, spec, serial):
        if serial != self.goal_serial:
            self.get_logger().warn('Ignoring a result from a stale goal')
            return

        self.goal_waiting = False
        self.active_handle = None
        self.active_spec = None

        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error('Goal result failed: %s' % error)
            self._retry_or_advance('exception', spec)
            return

        status = response.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                'Goal succeeded: %s %s (hov_time=%d s)' % (
                    spec.kind, spec.marker, response.result.hov_time))

            if spec.kind == 'save' and spec.phase == 'critical':
                self.critical_saves += 1
                self.get_logger().info(
                    'Critical survivors saved: %d/3' % self.critical_saves)

            if spec.kind == 'home':
                self.phase = 'done'
                self.get_logger().info(
                    'Mission complete: drone landed at F6')
                return

            if self.phase != spec.phase:
                self.get_logger().info(
                    'Goal finished during the Switch; not advancing the route')
                return

            self.goal_index += 1
            if self.phase == 'critical' and self.critical_saves >= 3:
                self.get_logger().info(
                    'All 3 critical survivors saved; starting Switch')
                self._begin_switch()
            return

        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Active goal canceled; switch can proceed')
            return

        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(
                'Goal aborted: %s %s' % (spec.kind, spec.marker))
            self._retry_or_advance('aborted', spec)
            return

        self.get_logger().warn('Goal finished with status %d' % status)
        self._retry_or_advance('status_%d' % status, spec)

    def _retry_or_advance(self, reason, spec):
        key = (spec.phase, self.goal_index, spec.kind, spec.marker)
        count = self.retry_counts.get(key, 0)
        if count < 2:
            self.retry_counts[key] = count + 1
            self.get_logger().warn(
                'Retrying the same goal (%s, attempt %d/2)'
                % (reason, count + 1))
            return

        self.get_logger().error(
            'Giving up on goal %s %s after retries; continuing'
            % (spec.kind, spec.marker))
        self.goal_index += 1

    # ------------------------------------------------------------------
    # Switch / re-plan
    # ------------------------------------------------------------------
    def _begin_switch(self):
        if self.phase != 'critical':
            return
        self.phase = 'switching'
        self.switch_time = time.monotonic()
        self.cancel_sent = False
        self.cancel_accepted = False
        self.get_logger().info(
            'SWITCH: canceling the active goal and waiting for it to finish')

        if not self.goal_waiting:
            return

        if self.active_handle is not None:
            self._send_cancel()

    def _send_cancel(self):
        self.cancel_sent = True
        self.get_logger().info('Requesting cancellation of the active goal')
        future = self.active_handle.cancel_goal_async()
        future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future):
        try:
            response = future.result()
            self.cancel_accepted = bool(response.goals_canceling)
        except Exception as error:
            self.cancel_accepted = False
            self.get_logger().warn('Cancel request failed: %s' % error)

        if self.cancel_accepted:
            self.get_logger().info('Cancel accepted; waiting for the goal result')
        else:
            self.get_logger().warn('Cancel was not accepted by the server')

    def _control_switch(self, now):
        if self.goal_waiting:
            if not self.cancel_sent and self.active_handle is not None:
                self._send_cancel()

            if now - self.switch_time >= 5.0:
                self.get_logger().warn(
                    'Cancel did not finish within 5 s; re-planning anyway')
                self.goal_waiting = False
                self.active_handle = None
            return

        if not self.replan_requested:
            if self.last_replan_failure and (
                    now - self.last_replan_failure) < 1.0:
                return
            self._request_replan()

    def _request_replan(self):
        request = PlanStableRoute.Request()
        if self.last_feedback is not None:
            x, y, z = self.last_feedback
        else:
            x, y, z = 0.0, 0.0, VISIT_Z
            self.get_logger().warn(
                'No feedback position available; planning from F6')

        request.current_position.position.x = float(x)
        request.current_position.position.y = float(y)
        request.current_position.position.z = float(z)
        request.current_position.orientation.w = 1.0

        self.replan_requested = True
        self.replan_attempts += 1
        self.get_logger().info(
            'Requesting stable route from Switch position (%.3f, %.3f, %.3f)'
            % (x, y, z))

        future = self.stable_client.call_async(request)
        future.add_done_callback(self.replan_callback)

    def replan_callback(self, future):
        self.replan_requested = False

        try:
            response = future.result()
        except Exception as error:
            response = None
            self.get_logger().error('Stable route call failed: %s' % error)

        if response is None or not response.success or \
                not response.waypoints.poses:
            message = response.message if response is not None else 'no response'
            self.last_replan_failure = time.monotonic()
            self.get_logger().error(
                'Stable route planning failed (%d/%d): %s' % (
                    self.replan_attempts, 5, message))
            if self.replan_attempts >= 5:
                self.get_logger().error(
                    'Could not obtain a stable route; mission stopped')
                self.phase = 'failed'
            return

        self.get_logger().info(
            'Stable route received: %d poses, markers=%s' % (
                len(response.waypoints.poses), list(response.marker_ids)))
        stable_goals = self._build_goals(response, 'stable')
        if not stable_goals:
            self.last_replan_failure = time.monotonic()
            self.get_logger().error('Stable route had no usable poses')
            return

        self.goals = stable_goals
        self.goal_index = 0
        self.goal_waiting = False
        self.active_handle = None
        self.active_spec = None
        self.phase = 'stable'
        self.get_logger().info(
            'Switch complete: flying the stable route and then landing at F6')

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        position = feedback.current_waypoint.pose.position
        self.last_feedback = (
            float(position.x), float(position.y), float(position.z))
        self.last_feedback_time = time.monotonic()

        now = time.monotonic()
        if now - self.last_status_log >= self.goal_log_interval:
            self.last_status_log = now
            self.get_logger().info(
                'Current whycode position: %.2f, %.2f, %.2f (hold %.1fs)' % (
                    position.x, position.y, position.z,
                    float(feedback.current_waypoint.header.stamp.sec)
                    + float(feedback.current_waypoint.header.stamp.nanosec)
                    * 1e-9))


def main(args=None):
    rclpy.init(args=args)

    waypoint_client = WayPointClient()
    started = waypoint_client.start_mission()

    if started:
        try:
            rclpy.spin(waypoint_client)
        except KeyboardInterrupt:
            waypoint_client.get_logger().info(
                'KeyboardInterrupt, shutting down.\n')
        finally:
            waypoint_client.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    else:
        waypoint_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
