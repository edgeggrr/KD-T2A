#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given drone.
This node publishes and subsribes the following topics:

		PUBLICATIONS			SUBSCRIPTIONS
		/drone_command			/whycode_node/markers
		/pos_error				/throttle_pid
								/pitch_pid
								/roll_pid

Rather than using different variables, use list. eg : self.desired_state = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_desired_state = 1, self.y_desired_state = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.
'''

# Importing the required libraries


from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from std_msgs.msg import Header
import time
import rclpy
from rclpy.node import Node


class Swift_Pico(Node):
	def __init__(self):
		super().__init__('pico_controller')  # initializing ros node with name pico_controller

		# This corresponds to your current position of drone. This value must be updated in your whycode callback
		# [x,y,z]
		self.current_state = [0.0, 0.0, 0.0]




		# This corresponds to the setpoint you want the drone to reach or hold
		# [x_desired_state, y_desired_state, z_desired_state]
		self.desired_state = [1.5, -1.5, 13.0]  # whycode marker at the position of the drone given in the scene. Make the whycode marker associated with position_to_hold drone renderable and make changes accordingly




		# Declaring a cmd of message type swift_msgs and initializing values
		self.cmd = SwiftMsgs()
		self.cmd.rc_roll = 1100
		self.cmd.rc_pitch = 1100
		self.cmd.rc_yaw = 1100
		self.cmd.rc_throttle = 1100

		#initial setting of Kp, Kd and ki for [pitch, roll, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
		#after tuning and computing corresponding PID parameters, change the parameters

		self.Kp = [0.0, 0.0, 0.0]
		self.Ki = [0.0, 0.0, 0.0]
		self.Kd = [0.0, 0.0, 0.0]

		#-----------------------Add other required variables for pid here ----------------------------------------------
		# Every list below is indexed [pitch, roll, throttle] == [x-axis, y-axis, z-axis]

		# Previous cycle's error per axis, used to compute the derivative term
		self.prev_error = [0.0, 0.0, 0.0]

		# Accumulated error per axis, used for the integral term
		self.error_sum = [0.0, 0.0, 0.0]

		# Anti-windup: cap how large the integral can grow on each axis. Without
		# this, an error that persists while the drone is stalled/out of frame
		# would keep summing and then kick the drone when control resumes.
		self.error_sum_limit = [5000.0, 5000.0, 5000.0]

		# Hard limits for the RC channel values
		self.max_values = [2000, 2000, 2000]
		self.min_values = [1000, 1000, 1000]

		# Latest per-axis error, published on /pos_error for the tuner GUI and rosbag
		self.pos_error = Error()

		# Only start controlling once a WhyCode marker has actually been seen, so
		# we never integrate against the placeholder (0, 0, 0) position.
		self.marker_received = False








		# Hint : Add variables for storing previous errors in each axis, like self.prev_error = [0,0,0] where corresponds to [pitch, roll, throttle]		#		 Add variables for limiting the values like self.max_values = [2000,2000,2000] corresponding to [roll, pitch, throttle]
		#													self.min_values = [1000,1000,1000] corresponding to [pitch, roll, throttle]
		#																	You can change the upper limit and lower limit accordingly. 
		#----------------------------------------------------------------------------------------------------------

		# # This is the sample time in which you need to run pid. Choose any time which you seem fit.
	
		self.sample_time = 0.033  # in seconds

		# Publishing /drone_command, /pos_error
		self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
		self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
		self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

		#------------------------Add other ROS 2 Publishers here-----------------------------------------------------


		# Subscribing to /whycode_node/markers, /throttle_pid, /pitch_pid, roll_pid
		self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
		self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)

		#------------------------Add other ROS Subscribers here-----------------------------------------------------
		self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
		self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)

		self.publish_heartbeat()
		self.create_timer(0.1, self.publish_heartbeat)

		time.sleep(0.5)  # let the heartbeat reach the rosbag before arming
		self.arm()  # ARMING THE DRONE

		# Creating a timer to run the pid function periodically, refer ROS 2 tutorials on how to create a publisher subscriber(Python)
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
		self.command_pub.publish(self.cmd)  # Publishing /drone_command


	# Whycode callback function
	# The function gets executed each time when /whycode_node publishes /whycode_node/markers
	def whycode_callback(self, msg):
		if not msg.markers:
			return
		self.current_state[0] = msg.markers[0].position.position.x
		#--------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------
		self.current_state[1] = msg.markers[0].position.position.y
		self.current_state[2] = msg.markers[0].position.position.z
		#---------------------------------------------------------------------------------------------------------------

		# From here on we have a real position to control against
		self.marker_received = True


	# Callback function for /throttle_pid
	# This function gets executed each time when /drone_pid_tuner publishes /throttle_pid
	def altitude_set_pid(self, alt):
		self.Kp[2] = alt.kp * 0.03  # This is just for an example. You can change the ratio/fraction value accordingly
		self.Ki[2] = alt.ki * 0.008
		self.Kd[2] = alt.kd * 0.6

	#----------------------------Define callback function like altitide_set_pid to tune pitch, roll--------------
	def pitch_set_pid(self, pitch):
		self.Kp[0] = pitch.kp * 0.03
		self.Ki[0] = pitch.ki * 0.008
		self.Kd[0] = pitch.kd * 0.6

	def roll_set_pid(self, roll):
		self.Kp[1] = roll.kp * 0.03
		self.Ki[1] = roll.ki * 0.008
		self.Kd[1] = roll.kd * 0.6

	#----------------------------------------------------------------------------------------------------------------------


	def pid(self):
	#-----------------------------Write the PID algorithm here--------------------------------------------------------------

	# Steps:
	# 	1. Compute error in each axis. eg: error[0] = self.current_state[0] - self.desired_state[0] ,where error[0] corresponds to error in x...
	#	2. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer "Understanding PID.pdf" to understand PID equation.
	#	3. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
	#	4. Reduce or add this computed output value on the avg value ie 1500. For eg: self.cmd.rcRoll = 1500 + self.out_roll. LOOK OUT FOR SIGN (+ or -). EXPERIMENT AND FIND THE CORRECT SIGN
	#	5. Don't run the pid continously. Run the pid only at the a sample time. self.sampletime defined above is for this purpose. THIS IS VERY IMPORTANT.
	#	6. Limit the output value and the final command value between the maximum(2000) and minimum(1000)range before publishing. For eg : if self.cmd.rcPitch > self.max_values[1]:
	#																														self.cmd.rcPitch = self.max_values[1]
	#	7. Update previous errors.eg: self.prev_error[1] = error[1] where index 1 corresponds to that of pitch (eg)
	#	8. Add error_sum

		# Nothing to control against until the marker has been seen
		if not self.marker_received:
			return

		error = [0.0, 0.0, 0.0]
		derivative = [0.0, 0.0, 0.0]
		output = [0.0, 0.0, 0.0]

		# 1. Error per axis, indexed [pitch(x), roll(y), throttle(z)]
		error[0] = self.current_state[0] - self.desired_state[0]
		error[1] = self.current_state[1] - self.desired_state[1]
		error[2] = self.current_state[2] - self.desired_state[2]

		for i in range(3):
			# 8. Integral term, with anti-windup clamping
			self.error_sum[i] += error[i]
			if self.error_sum[i] > self.error_sum_limit[i]:
				self.error_sum[i] = self.error_sum_limit[i]
			elif self.error_sum[i] < -self.error_sum_limit[i]:
				self.error_sum[i] = -self.error_sum_limit[i]

			# 2. Change in error over one sample period (derivative term)
			derivative[i] = (error[i] - self.prev_error[i]) / self.sample_time

			# 3. PID output for this axis
			output[i] = (self.Kp[i] * error[i]
						 + self.Ki[i] * self.error_sum[i]
						 + self.Kd[i] * derivative[i])

		# 4. Add the correction onto the neutral midpoint (1500) and
		# 6. clamp the final command into the valid RC range.
		#    NOTE: if an axis runs away instead of settling, flip that axis's
		#    sign here (+ -> -) and re-tune.
		self.cmd.rc_pitch = int(min(max(1500 + output[0], self.min_values[0]), self.max_values[0]))
		self.cmd.rc_roll = int(min(max(1500 + output[1], self.min_values[1]), self.max_values[1]))
		self.cmd.rc_throttle = int(min(max(1500 + output[2], self.min_values[2]), self.max_values[2]))

		# 7. Store this cycle's error for the next derivative calculation
		self.prev_error[0] = error[0]
		self.prev_error[1] = error[1]
		self.prev_error[2] = error[2]

	#------------------------------------------------------------------------------------------------------------------------
		self.command_pub.publish(self.cmd)
		# calculate throttle error, pitch error and roll error, then publish it accordingly
		self.pos_error.pitch_error = error[0]
		self.pos_error.roll_error = error[1]
		self.pos_error.throttle_error = error[2]
		self.pos_error_pub.publish(self.pos_error)



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
