#!/usr/bin/env python
import rospy
import numpy as np
from sub8_msgs.srv import VisionRequest, VisionRequest2D, VisionRequestResponse, VisionRequest2DResponse
from mil_ros_tools import Image_Publisher, Image_Subscriber, StereoImageSubscriber,\
                          numpy_to_point, numpy_to_quaternion, numpy_to_pose2D, numpy_to_vector3
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_matrix
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import SetBool
from image_geometry import PinholeCameraModel


__author__ = "Kevin Allen"


class VisionNode(object):
    '''
    Class to boostrap your vision node. Creates all the services and subscribers
    needed to write a vision node. Also has some helpful static functions to use in
    your node. In general, class variables / functions starting with an underscore
    (ex: _img_cb) are for internal use and should not be modified by classes
    using this abstraction.
    '''
    def __init__(self):
        self._initialized = False
        self.enabled = False
        self.debug_ros = rospy.get_param('~debug_ros', True)
        if self.debug_ros:
            self._marker_pub = rospy.Publisher('~marker', Marker, queue_size=5)
            self._debug_image_pub = Image_Publisher('~debug_image')
        self._enable_service = rospy.Service('~enable', SetBool, self._enable_cb)
        self._pose_2D_service = rospy.Service('~2D', VisionRequest2D, self._request2D)
        self._pose_3D_service = rospy.Service('~pose', VisionRequest, self._request3D)
        self._init_cams()
        self._initialized = True

    def _init_cams(self):
        image_topic = rospy.get_param('~image_topic')
        self.cam = None
        self.last_image_time = None
        self._image_sub = Image_Subscriber(image_topic, self._img_cb)
        self._camera_info = self._image_sub.wait_for_camera_info()
        self.cam = PinholeCameraModel()
        self.cam.fromCameraInfo(self._camera_info)

    def send_debug_image(self, img):
        if self.debug_ros:
            self._debug_image_pub.publish(img)

    def send_marker(self, marker):
        if self.debug_ros:
            self._marker_pub.publish(marker)

    @staticmethod
    def make_response(position=None, orientation=None, pose=None, found=True, frame='/map',
                      stamp=None, covariance=None):
        res = VisionRequestResponse(found=found)
        if pose is not None:
            position = pose[0]
            orientation = pose[1]
        if position is not None:
            position = np.array(position)
            res.pose.pose.position = numpy_to_point(np.array(position))
        if orientation is not None:
            orientation = np.array(orientation)
            if orientation.shape == (3, 3):
                e = np.eye(4)
                e[:3, :3] = orientation
                orientation = quaternion_from_matrix(e)
            elif orientation.shape == (4,4):
                orientation = quaternion_from_matrix(orientation)
            res.pose.pose.orientation = numpy_to_quaternion(orientation)
        if stamp is None:
            res.pose.header.stamp = rospy.Time.now()
        else:
            res.pose.header.stamp = stamp
        if covariance is not None:
            res.covariance_diagonal = numpy_to_vector3(covariance)
        res.pose.header.frame_id = frame
        return res

    @staticmethod
    def make_response2D(pose=None, found=True, stamp=None, frame=None, info=CameraInfo(), max_x=0, max_y=0):
        res = VisionRequest2DResponse(found=found, max_x=max_x, max_y=max_y, camera_info=info)
        if pose is not None:
            res.pose = numpy_to_pose2D(numpy.array(pose))
        if stamp is None:
            stamp = rospy.Time.now()
        res.header.stamp = stamp
        if frame is not None:
            res.header.frame_id = frame
        return res

    def _img_cb(self, img):
        self.last_image_time = self._image_sub.last_image_time
        if not self.enabled:
            return
        self.img_cb(img)

    def _request2D(self, req):
        if not self.enabled:
            return VisionRequest2DResponse(found=False)
        return self.request2D(req.target_name)

    def _request3D(self, req):
        if not self.enabled:
            return VisionRequest2DResponse(found=False)
        return self.request3D(req.target_name)

    def _enable_cb(self, req):
        if not self._initialized:
            return {'success': False, 'message': 'not _initialized'}
        self.enabled = req.data
        self.enable_cb(req.data)
        return {'success': True}

    def img_cb(self, img):
        pass

    def request2D(self, target):
        rospy.logwarn('Please override request2D, returning found=False')
        return VisionRequest2DResponse(found=False)

    def request3D(self, target):
        rospy.logwarn('Please override request3D, returning found=False')
        return VisionRequestResponse(found=False)

    def enable_cb(self, enabled):
        '''
        Overide me!
        @param enabled Boolean, if node should begin processing images
        '''
        pass


class StereoVisionNode(VisionNode):
    def _init_cams(self):
        left_topic = rospy.get_param('~image_topic/left', '/camera/front/left/image_rect_color')
        right_topic = rospy.get_param('~image_topic/right', '/camera/front/right/image_rect_color')
        slop = rospy.get_param('~slop', None)
        self._image_sub = StereoImageSubscriber(left_topic, right_topic, callback=self._img_cb, slop=slop)
        self._camera_info_left, self._camera_info_right = self._image_sub.wait_for_camera_info()
        self.cam_left, self.cam_right = PinholeCameraModel(), PinholeCameraModel()
        self.cam_left.fromCameraInfo(self._camera_info_left)
        self.cam_right.fromCameraInfo(self._camera_info_right)
        self.last_image_time_left = None
        self.last_image_time_right = None

    def _img_cb(self, left, right):
        self.last_image_time_left = self._image_sub.last_image_time_left
        self.last_image_time_right = self._image_sub.last_image_time_right
        if not self.enabled:
            return
        self.img_cb(left, right)


class ExampleVisionNode(VisionNode):
    '''
    Example of the minimum needed to have a vision node using
    the template, and a test for the template.
    '''
    def __init__(self):
        super(ExampleVisionNode, self).__init__()
        self.enabled = True  # Auto start node

    def request2D(self, target):
        return self.make_response2D(pose=(500, 500, 0))

    def request3D(self, target):
        return self.make_response(position=(10, 10, -2), frame='/odom')

    def img_cb(self, img):
        print 'I got an image!'

if __name__ == '__main__':
    rospy.init_node('example_vision')
    ExampleVisionNode()
    rospy.spin()
