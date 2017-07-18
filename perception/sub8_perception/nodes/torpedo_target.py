#!/usr/bin/env python
from sub8_vision_tools import VisionNode
import cv2
import numpy as np
import rospy
import tf
from visualization_msgs.msg import Marker
from mil_ros_tools import numpy_pair_to_pose, numpy_to_vector3, numpy_to_point, numpy_to_quaternion, numpy_to_colorRGBA
from mil_vision_tools import CircleFinder, Threshold, auto_canny
from geometry_msgs.msg import Point
import scipy
from tf.transformations import quaternion_from_euler, quaternion_matrix
from mil_vision_tools import ImageMux
from sub8_vision_tools import MultiObservation


# Ensure opencv3 is used
assert cv2.__version__[0] == '3'

class Target(object):
    MULTI_OBSERVATION_MODEL = None
    NEAR_TOLERANCE_PIXELS = 20
    top_idx = 0
    COLORS=np.array([[1.0, 0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0],
                     [1.0, 1.0, 0.0, 1.0]], dtype=float)
    NAMES=['RED', 'GREEN', 'BLUE', 'YELLOW']

    def __init__(self, points, transform, time):
        self.last_points = points
        self.points = np.array([points])
        self.tfs = [transform]
        self.times = [time]
        self.id = Target.top_idx
        if self.id >= 4:
            self.name = 'UNKNOWN {}'.format(self.id)
            self.color = [1.0, 1.0, 1.0, 1.0]
        else:
            self.name = self.NAMES[self.id]
            self.color = self.COLORS[self.id]
        Target.top_idx += 1

    @classmethod
    def set_camera_model(cls, model):
        cls.MULTI_OBSERVATION_MODEL = MultiObservation(model)

    @classmethod
    def set_params(cls):
        pass

    def get_response(self):
        if self.pose is None:
            return VisionNode.make_response(found=False)
        return VisionNode.make_response(pose=self.pose, frame='/map', stamp=self.time)

    def add_observation(self, points, transform, time):
        self.last_points = points
        if np.linalg.norm(transform[0] - self.tfs[-1][0]) > 0.05:
            self.points = np.vstack((self.points, [points]))
            self.tfs.append(transform)
            self.times.append(time)

    def near(self, centroid):
        dist = np.linalg.norm(centroid - self.last_points[4])
        return dist <= self.NEAR_TOLERANCE_PIXELS

    def estimate_pose(self):
        pts3D = []
        for i in xrange(self.points.shape[1]-1):
            pts3D.append(self.MULTI_OBSERVATION_MODEL.lst_sqr_intersection(self.points[:, i, :], self.tfs))
        position = self.MULTI_OBSERVATION_MODEL.lst_sqr_intersection(self.points[:, -1, :], self.tfs)
        pts3D = np.array(pts3D)
        A = np.c_[pts3D[:, 0], pts3D[:, 1], np.ones(pts3D.shape[0])]
        coeff, resid, _,_ = np.linalg.lstsq(A, pts3D[:, 2])
        yaw = np.arctan2(coeff[1], coeff[0])
        quat = quaternion_from_euler(0, 0, yaw)
        return pts3D, (position, quat)


class TorpedoTargetFinder(VisionNode):
    '''
    Node to find the circular targets of the "Battle a Squid" / torpedo task in RoboSub2017.

    Uses basic OpenCV edge detection and heuristic checks to identify likely target edges.
    solvePnP is applied to these edges with their known demensions to estimate 3D pose for the targets.

    A rough overview of the algorithm:
    * denoise camera image using bluring or erosion/dilation
    * run an edge/corner detection alogrithm to create a edge mask image
    * process detected contours to identify up to 4 pottential targets
      * attempt to fine pairs of 2 concentric elipses representing the inner and outer edge of the target
      * filter false positives

    TODO:
    - roslaunch / YAMLize: put in perception.launch, constants in yaml
    - allow for multiple targets in one frame. Track them based on:
       - 3D position/pose in map?: good for moving Sub, might create duplicates if bad initial estimate
       - instantanuous centroid: should be able to shift over time....
    - KF them poses?
       - ehhhhhhhh, could use
    '''
    def __init__(self):
        rospy.set_param('~image_topic', '/camera/front/left/image_rect_color')
        rospy.set_param('~slop', 0.1)
        self.targets = []
        self.canny_low = 30
        self.canny_ratio = 1.0
        self.pose = None

        self.tf_listener = tf.TransformListener()
        self.o = CircleFinder(1.0)

        thresh = {'HSV':[[24,90,60],[70,255,255]]}
        rospy.set_param('~thresh', thresh)
        self.threshold = Threshold.from_param('~thresh')
        super(TorpedoTargetFinder, self).__init__()
        Target.set_camera_model(self.cam)
        if self.debug_ros:
            scale = 1.0
            self.debug_image = ImageMux(size=(scale*self._camera_info.height, scale * self._camera_info.width), shape=(2, 1), text_color=(0, 0, 255),
                                        labels=['FOUND', 'THRESH'])

        self.enabled = True

    def request3D(self, target):
        '''
        Service callback for missions.
        '''
        if len(self.targets) == 0:
            return self.make_repsonse(found=False)
        return self.targets[0].get_response()

    def get_target_corners(self, img):
        blur = cv2.blur(img, (5, 5))
        thresh = self.threshold.threshold(blur)
        edges = auto_canny(thresh, 0.8)
        _, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blank = img.copy()#np.zeros(img.shape, dtype=img.dtype)
        targets = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            if self.o.verify_contour(cnt) < 0.4:
                if hierarchy[0][idx][3] != -1 and hierarchy[0][idx][2] == -1:  # Only get outer edges
                    cv2.drawContours(blank, [cnt], 0, (0, 0, 255), 3)
                    corners = self.o.get_corners(cnt, debug_image=blank)
                    targets.append(corners)
        self.debug_image[0] = blank
        self.debug_image[1] = thresh
        return targets

    def points_marker(self, pts, pose, id=0):
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'torpedo_points'
        marker.type = Marker.POINTS
        marker.id = id
        marker.color = numpy_to_colorRGBA(self.targets[id].color)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.frame_locked = True
        for pt in pts:
            marker.points.append(Point(pt[0], pt[1], pt[2]))
        self.send_marker(marker)
        marker.id = id + 100
        marker.type = Marker.ARROW
        marker.pose.position = numpy_to_point(pose[0])
        marker.pose.orientation = numpy_to_quaternion(pose[1])
        marker.points = []
        marker.scale.z = 0.05
        marker.scale.x = 0.5
        marker.scale.y = 0.05
        self.send_marker(marker)

    def get_target_pose(self, pts3D):
        '''
        Returns pose tutple (position, orientation) from
        approximate plane of  corner points pts3D

        @param pts3d 3D points
        '''
        centroid = np.mean(pts3D, axis=0)
        pts3D = np.subtract(pts3D, centroid)
        A = np.c_[pts3D[:, 0], pts3D[:, 1], np.ones(pts3D.shape[0])]
        coeff, resid, _,_ = np.linalg.lstsq(A, pts3D[:, 2])
        yaw = np.arctan2(coeff[1], coeff[0])
        quat = quaternion_from_euler(0, 0, yaw)
        return (centroid, quat), resid[0]

    def sane_pose_est(self, points3d):
        z = np.mean(points3d[:, 2])
        #return z > 1.0
        distances = [np.linalg.norm(points3d[3] - points3d[2]),
                     np.linalg.norm(points3d[2] - points3d[1]),
                     np.linalg.norm(points3d[1] - points3d[0]),
                     np.linalg.norm(points3d[3] - points3d[0])]
        for d in distances:
            if (d < 0.1 or d > 0.65):
                print 'insane in the membrane',d, np.mean(points3d[:, 2])
                return False
        return True

    def add_target(self, target, transform, time):
        M = cv2.moments(target)
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        target = np.vstack((target, centroid))
        for idx in xrange(len(self.targets)):
            if self.targets[idx].near(target[-1]):
                self.targets[idx].add_observation(target, transform, time)
                if len(self.targets[idx].points) > 10:
                    points, pose = self.targets[idx].estimate_pose()
                    self.points_marker(points, pose, idx)
                return
        if len(self.targets) >= 4:
            print 'too many'
            return
        self.targets.append(Target(target, transform, time))

    def img_cb(self, img):
        try:
            self.tf_listener.waitForTransform('/map', self.cam.tfFrame(), self.last_image_time, rospy.Duration(0.2))
            tf_trans, tf_quat = self.tf_listener.lookupTransform('/map', self.cam.tfFrame(), self.last_image_time)
            tf_rot = quaternion_matrix(tf_quat)[:3, :3]
            tf_cam_map = (np.array(tf_trans), np.array(tf_rot))
        except tf.Exception as e:
            rospy.logwarn('TF error {}'.format(e))
            return
        targets = self.get_target_corners(img)
        for target in targets:
            self.add_target(target, tf_cam_map, self.last_image_time)
        self.send_debug_image(self.debug_image.image)

if __name__ == '__main__':
    rospy.init_node('torpedo_target')
    TorpedoTargetFinder()
    rospy.spin()
