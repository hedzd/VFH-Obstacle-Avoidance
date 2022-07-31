#!/usr/bin/python3

import math
from numpy.core.shape_base import block
import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from rospy.timer import sleep
from copy import deepcopy
import sys
import threading
import random
import datetime
import tf
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid
from math import radians, sqrt, pow, pi
import matplotlib.pyplot as plt
from math import pi


ROTATION , MOVE = range(2)

class VFHController():
    def __init__(self, *args, **kwargs):
        rospy.init_node('velocity_controller', anonymous=False)
        print(rospy.get_rostime())
        rospy.on_shutdown(self.shutdown)

        self.new_velocity_sub = rospy.Publisher('/change', Twist, queue_size=1)
        self.r = rospy.Rate(20)
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.base_frame = ''
        self.odom_frame = '/odom'
        self.tf_listener = tf.TransformListener()
        rospy.sleep(2)

        self.angular_speed = 0.3
        self.linear_speed = 0.25
        
        self.start_time = datetime.datetime.now()
        self.tf_listener.waitForTransform(self.odom_frame, '/base_footprint', rospy.Time(), rospy.Duration(1.0))
        self.base_frame = '/base_footprint'
                
        self.grid_sub = rospy.Subscriber('/window_data' , OccupancyGrid , self.callback_grid , queue_size=1)
        

        # 0.0175 Radian = 1 Degree
        self.angle_increment = 0.087
        # 6.28 Radian = 360 Degrees
        self.number_of_sectors = int(6.28/self.angle_increment) + 1
        
        self.smoothing_factors = list(range(1,4)) + [4] + list(reversed(range(1,4))) 

        goal_x = rospy.get_param("/vfh_algo/goal_x") 
        goal_y = rospy.get_param("/vfh_algo/goal_y") 

        print("goal_x", goal_x)
        print("goal_y", goal_y)
        
        self.target_point = (goal_x,goal_y)
        self.state = MOVE
        self.end_rotation = rospy.get_rostime()
        
        self.smax = 12
        self.angular_tolerance = 0.2
        self.threshold = 8000
        
        tw_msg = Twist()
        tw_msg.linear.x = self.linear_speed
        self.cmd_vel.publish(tw_msg)
    
    
    def calculate_magnitude(self, window_map):
        M = 1
        
        N, _ = window_map.shape
        vcp = (N - 1) / 2
        distances = np.zeros(window_map.shape)
        d_max = 1.4142135 * vcp
        
        for i in range(N):
            for j in range(N):
                # calculate di,j
                distances[i,j] = ((i - vcp) ** 2 + (j - vcp) ** 2) ** 0.5
        
        #return mi,j
        return (window_map ** 2) * (1 - 0.25 * distances)

   
    def callback_grid(self , msg:OccupancyGrid):
        if self.state == ROTATION or self.end_rotation >= msg.info.map_load_time:
            return
        
        self.state = ROTATION
        
        tw_msg = Twist()
        self.cmd_vel.publish(tw_msg)
        rospy.sleep(1)
        
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height

        # mapp = np.array(list(msg.data)).reshape(width,height)
        mapp = self.calculate_magnitude(np.array(list(msg.data)).reshape(width,height))
             
        histogram = np.zeros((self.number_of_sectors,))
        
        # bottom to top
        y = (-height // 2) * resolution
        # left to right
        for i in range(height):
            x = (-width // 2) * resolution
            for j in range(width):
                sector = int(self.scale(np.arctan2(y,x)) / self.angle_increment)
                histogram[sector] += mapp[i,j]
                x += resolution
            y += resolution
                
                
        histcopy = deepcopy(histogram)
        l = 2
    
        for i in range(len(histogram)):
            val = 0
            for j in range(-l,l):
                val += histcopy[(i+j) % len(histogram) if i+j >= 0 else i+j] * self.smoothing_factors[l+j]
            val /= 2 * (l+1) + 1
            histogram[i] = val
            
        pos , rotation = self.get_heading()
        robot_heading_sector = int(self.scale(rotation) / self.angle_increment)
        ktarget =len(histogram) + robot_heading_sector
        
        i = ktarget - 1
        j = ktarget + 1
        
        goal_sector = None
        
        kn,kf = None,None
        
        
        histogram = np.array(histogram.tolist() + histogram.tolist() + histogram.tolist())
        valleys = histogram <= self.threshold
        if not valleys[ktarget]:
            while i >= 0 or j < len(histogram):
                res = []
                if i >=0 and valleys[i]:
                    ind = i
                    kn = i
                    valley_size = 1
                    while ind >= 0 and valleys[ind]:
                        valley_size += 1
                        ind -= 1
                    if valley_size >= self.smax:
                        kf = kn - self.smax
                    else:
                        kf = ind + 1
                        
                    assert kf is not None
                    goal_sector = (kf + kn) // 2
                    res.append((kf , kn , ktarget , goal_sector))
                i -= 1
                
                if j < len(histogram) and valleys[j]:
                    ind = j
                    kn = j
                    valley_size = 1
                    while ind < len(histogram) and valleys[ind]:
                        valley_size += 1
                        ind += 1
                    if valley_size >= self.smax:
                        kf = kn + self.smax
                    else:
                        kf = ind - 1
                    
                    assert kf is not None
                    goal_sector = (kf + kn) // 2
                    res.append((kf , kn , ktarget , goal_sector))
                    
                j += 1
                assert len(res) <= 2
                if len(res) == 2:
                    (kf , kn , ktarget , goal_sector) = res[0] if abs(res[0][0] - res[0][1]) > abs(res[1][0] - res[1][1]) else res[1]
                    break
                elif len(res) == 1:
                    (kf , kn , ktarget , goal_sector) = res[0]
                    break
                    
        else:
            print("target in valley")
            res = []
            ind = i
            kn = i
            valley_size = 1
            while ind >= 0 and valleys[ind]:
                valley_size += 1
                ind -= 1
            if valley_size >= self.smax:
                kf = kn - self.smax
            else:
                kf = ind + 1
                
            assert kf is not None
            goal_sector = (kf + kn) // 2
            res.append((kf , kn , ktarget , goal_sector))
            
            ind = j
            kn = j
            valley_size = 1
            while ind < len(histogram) and valleys[ind]:
                valley_size += 1
                ind += 1
            if valley_size >= self.smax:
                kf = kn + self.smax
            else:
                kf = ind - 1
            
            assert kf is not None
            goal_sector = (kf + kn) // 2
            res.append((kf , kn , ktarget , goal_sector))
            goal_angles = [self.normalize(g[3] * self.angle_increment) for g in res]
            (kf , kn , ktarget , goal_sector) = res[0] if abs(res[0][0] - res[0][1]) > abs(res[1][0] - res[1][1]) else res[1]
            
               
        if goal_sector is None:
            self.threshold = min(histogram) + 1000
            goal_sector = robot_heading_sector
            
       
        goal_angle = self.normalize(goal_sector * self.angle_increment) 
        
        
        tw_msg = Twist()
        last_angle = rotation
        first_angle = rotation
        turn_angle = 0
        tw_msg.angular.z = self.angular_speed * ( goal_angle - rotation) / abs(goal_angle - rotation)
        
        while abs(turn_angle) < abs(first_angle - goal_angle) and not rospy.is_shutdown():
            self.cmd_vel.publish(tw_msg)
            self.r.sleep()
            position , rotation = self.get_heading()
            delta_angle = self.normalize(rotation - last_angle)
            turn_angle += delta_angle
            last_angle = rotation
            
        tw_msg = Twist()
        self.cmd_vel.publish(tw_msg)
        rospy.sleep(2)
        tw_msg = Twist()
        tw_msg.linear.x = self.linear_speed
        self.cmd_vel.publish(tw_msg)
        
        self.state = MOVE
        self.end_rotation = rospy.get_rostime()
        
        
    def get_heading(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.odom_frame, self.base_frame, rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), self.quat_to_angle(Quaternion(*rot)))


    def normalize(self , angle):
        res = angle
        while res > pi:
            res -= 2.0 * pi
        while res < -pi:
            res += 2.0 * pi
        return res
    
    def scale(self,angle):
        res = angle
        if angle < 0:
            return angle + (2 * pi)
        return angle

    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
    def quat_to_angle(self, quat):
        return tf.transformations.euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))[2]  

if __name__ == '__main__':
    robot = VFHController()  
    rospy.spin()