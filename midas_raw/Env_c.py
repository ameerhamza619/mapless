#!/usr/bin/env python3

# Python libraries for ROS
import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import numpy as np
import math
from math import pi

import gym
from gym import spaces

from respawnGoal import Respawn
from tf.transformations import euler_from_quaternion
from collections import deque

import PIL
import ros_numpy

import torch
import utils
import cv2

from torchvision.transforms import Compose
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas.midas_net_custom import MidasNet_small

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model_path = '../midas_v_np/weights/midas_v21_small-70d6b9c8.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
net_w, net_h = 256, 256
resize_mode="upper_bound"
normalization = NormalizeImage(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = Compose(
    [
        Resize(
            net_w ,
            net_h ,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_mode,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
)

model.eval()

model.to(device)


class Env(gym.Env):
    def __init__(self, is_training):
        super(Env, self).__init__()
        rospy.init_node('baseline')
        # rospy.Rate(5)

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        # self.depth_img = rospy.Publisher('/camera1/depth', Image, queue_size=1)

        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry, queue_size=1)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty) 
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        
        self.bot_position = Pose()
        self.goal_position_x = 0
        self.goal_position_y = 0
        self.heading = 0
        self.initGoal = True
        self.respawn_goal = Respawn()
        self.steps = 0
        self.diagonal_distance = round(math.sqrt(3.8**2 + 3.8**2),2)
        # self.diagonal_distance = round(math.sqrt(4.8**2 + 4.8**2),2)

        self.past_distance = 0.

        if is_training:
            self.threshold_arrive = 0.2
            self.min_range = 0.2
        else:
            self.threshold_arrive = 0.4
            self.min_range = 0.15

        self.action_space = spaces.Box(low=np.array([-1.0, 0.05]), high=np.array([1.0, 0.2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(192, 256, 1) ,dtype=np.uint8)

        self.distance_rate = 0
        self.episode = 0

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_position_x - self.bot_position.x, self.goal_position_y - self.bot_position.y),2)
        self.past_distance = goal_distance
        return goal_distance

    def getOdometry(self, odom):
        self.bot_position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_position_y - self.bot_position.y, self.goal_position_x - self.bot_position.x)

        # Difference of angle between robot and goal
        heading = goal_angle - yaw

        # Keeping angle between -pi and pi
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)    

    def getState(self, scan, image):
        scan_range = []
        done = False
        status = {'collide': False, 'goal': False, 'limit': False}

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if self.min_range > min(scan_range) > 0:
            done = True
            status['collide'] = True

        self.current_distance = round(math.hypot(self.goal_position_x - self.bot_position.x, self.goal_position_y - self.bot_position.y),2)
        if self.current_distance <= self.threshold_arrive:
            # Reached desired goal
            status['goal'] = True
            # done = True
            self.steps = 0

        if self.steps >= 200:
            done = True
            status['limit'] = True

        img = utils.read_image(image)
        img_input = transform({"image": img})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            prediction = model.forward(sample)
            prediction = prediction.squeeze().cpu().numpy()
            prediction = (255 * (prediction - prediction.min()) / (prediction.max() - prediction.min())).astype(np.uint8)
            # cv2.imwrite(f'output/{self.steps}.png', prediction)
            # self.depth_img.publish(ros_numpy.msgify(Image, prediction, encoding='mono8'))

            obs = prediction.reshape(192, 256, 1)

        return obs, self.current_distance, self.heading, done, status

    def setReward(self, state, status, action):
    

        distance_rate = round(self.past_distance - self.current_distance,2)
        self.distance_rate = distance_rate
        angular_penalty = round(1 - math.cos(action[0]),2)

        reward = (100 * distance_rate) - 0.1 - angular_penalty + (action[1] * 10)
        self.past_distance = self.current_distance

        if status['collide']:
            # rospy.loginfo("Collision!!")
            print("Collision!!")
            reward = -120 #50
            self.pub_cmd_vel.publish(Twist())
            self.goal_position_x, self.goal_position_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()

        if status["goal"]:
            # rospy.loginfo("Goal!!")
            print("Goal!!")
            reward = 100
            self.pub_cmd_vel.publish(Twist())
            self.goal_position_x, self.goal_position_y = self.respawn_goal.getPosition(True, delete=True)
            rospy.sleep(1)
            self.goal_distance = self.getGoalDistace()

        if status['limit']:
            reward = -120 #50 
            self.pub_cmd_vel.publish(Twist())
            # rospy.loginfo("Timesteps exceeded!!")
            print("Timesteps exceeded!!")
            self.goal_position_x, self.goal_position_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()

        return reward

    def step(self, action):

        self.steps += 1
        ang_vel = action[0]
        vel_cmd = Twist()
        vel_cmd.linear.x = action[1] #0.25
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        rospy.sleep(0.05)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                rospy.loginfo("Unable to read laser data : step")
                pass

        image = None
        while image is None:
            try:
                image = rospy.wait_for_message('camera1/image_raw', Image, timeout=5)
                # bridge = CvBridge()
                try:
                    # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
                    # image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
                    image = ros_numpy.numpify(image)
                except Exception as e:
                    raise e
            except:
                pass

        state, distance, heading, done, status = self.getState(data, image)
        reward = self.setReward(state, status, action)

        status["Episode"] = self.episode

        # return np.asarray(state), reward, done, {"Episode": self.episode}
        return np.asarray(state), reward, done, status

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            rospy.loginfo("gazebo/reset_simulation service call failed")
        
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                rospy.loginfo("Unable to read laser data : reset")
                pass

        image = None
        while image is None:
            try:
                image = rospy.wait_for_message('camera1/image_raw', Image, timeout=5)
                # bridge = CvBridge()
                try:
                    # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
                    # image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
                    image = ros_numpy.numpify(image)
                except Exception as e:
                    raise e
            except:
                pass

        if self.initGoal:
            self.goal_position_x, self.goal_position_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, distance, heading, done, status = self.getState(data, image)


        self.steps = 0
        self.episode += 1


        # rospy.loginfo("Env reset!")
        return np.asarray(state)
