
import rospy
from sensor_msgs.msg import LaserScan, Image
from matplotlib import cm
import cv2
import ros_numpy

rospy.init_node("Testing123")

try:
    image = rospy.wait_for_message('camera1/image_raw', Image, timeout=5)
    cv2.imwrite("image_save.png", image)
except:
    print("Image not saved!!")

print("Image Saved!!")
