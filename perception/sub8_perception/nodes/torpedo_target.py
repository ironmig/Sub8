#!/usr/bin/env python
from sub8_vision_tools import StereoVisionNode
import cv2
import numpy as np
import rospy
import tf
from visualization_msgs.msg import Marker
from mil_ros_tools import numpy_pair_to_pose, numpy_to_vector3, numpy_to_point, numpy_to_quaternion, numpy_to_colorRGBA
from mil_vision_tools import CircleFinder, Threshold, auto_canny
from geometry_msgs.msg import Point
import scipy
from tf.transformations import quaternion_from_euler
from mil_vision_tools import ImageMux


# Ensure opencv3 is used
assert cv2.__version__[0] == '3'

class Target(object):
    DIAMETER = 0.3048
    POSITION_TOLERANCE = 0.2
    top_idx = 0
    COLORS=np.array([[1.0, 0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0],
                     [1.0, 1.0, 0.0, 1.0]], dtype=float)
    NAMES=['RED', 'GREEN', 'BLUE', 'YELLOW']

    def __init__(self, pose=None, time=None, error=None):
        self.id = Target.top_idx
        Target.top_idx += 1
        self.pose = pose
        self.time = time
        self.error = error

    def get_response(self):
        if self.pose is None:
            return VisionNode.make_response(found=False)
        return VisionNode.make_response(pose=self.pose, frame='/map', stamp=self.time)

    def update_pose(self, pose, error):
        '''
        If pose is near current pose, update pose and return True
        If pose if far from current pose, do nothing and return False
        '''
        dist = np.linalg.norm(pose[0] - self.pose[0])
        print 'DIST', dist
        if dist < self.POSITION_TOLERANCE:
            if error <= self.error:
                self.pose = pose
                self.error = error
            return True
        return False

    def get_marker(self):
        marker = Marker()
        marker.header.frame_id  = '/map'
        marker.ns = 'torpedo_target'
        marker.type = Marker.ARROW
        pitch_90 = tf.transformations.euler_matrix(0.0, np.degrees(90), 0.0)[:3, :3]
        orientation = self.pose[1]
        marker.pose = numpy_pair_to_pose(self.pose[0], orientation)
        marker.color = numpy_to_colorRGBA(Target.COLORS[self.id])
        marker.scale = numpy_to_vector3((0.5, 0.1, 0.1))
        #marker.scale = numpy_to_vector3(np.array((0.5, 0.5, 0.5)))
        #marker.scale = numpy_to_vector3(np.array((0.1, self.DIAMETER, self.DIAMETER)))
        return marker


class TorpedoTargetFinder(StereoVisionNode):
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
        rospy.set_param('~image_topic/left', '/camera/front/left/image_rect_color')
        rospy.set_param('~image_topic/right', '/camera/front/right/image_rect_color')
        rospy.set_param('~slop', 0.1)
        self.targets = []
        self.canny_low = 30
        self.canny_ratio = 1.0
        self.pose = None

        self.tf_listener = tf.TransformListener()
        self.o = CircleFinder(Target.DIAMETER)

        thresh = {'LAB':[[0,99,139],[255,127,219]]}
        rospy.set_param('~thresh', thresh)
        self.threshold = Threshold.from_param('~thresh')
        super(TorpedoTargetFinder, self).__init__()
        self.left_dist, self.right_dist, self.left_proj, self.right_proj = self.get_stereo_projection_matricies()
        if self.debug_ros:
            scale = 1.0
            self.debug_image = ImageMux(size=(scale*self._camera_info_left.height, scale * self._camera_info_right.width), shape=(2, 2), text_color=(0, 0, 255),
                                        labels=['LEFT FOUND', 'RIGHT FOUND', 'THRESH', 'THRESH'])

        self.enabled = True

    def request3D(self, target):
        '''
        Service callback for missions.
        '''
        if len(self.targets) == 0:
            return self.make_repsonse(found=False)
        return self.targets[0].get_response()

    def get_stereo_projection_matricies(self):
        try:
            self.tf_listener.waitForTransform(self.cam_left.tfFrame(), self.cam_right.tfFrame(), rospy.Time(), rospy.Duration(0.5))
        except tf.Exception as e:
            print 'TF Exception {}'.format(e)
            return None
        tf_trans, tf_quat = self.tf_listener.lookupTransform(self.cam_left.tfFrame(), self.cam_right.tfFrame(), rospy.Time())
        left_rect, right_rect, left_proj, right_proj, _,_,_ = cv2.stereoRectify(self.cam_left.projectionMatrix()[:3, :3],
                np.zeros(5), self.cam_right.projectionMatrix()[:3, :3], np.zeros(5),
          (self._camera_info_left.width, self._camera_info_right.height), np.array([0,0,0]), tf_trans)
        self.tf_trans = tf_trans
        return left_rect, right_rect, left_proj, right_proj

    def get_target_corners(self, img, camera):
        print 'camera'
        blur = cv2.blur(img, (5, 5))
        thresh = self.threshold.threshold(blur)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        edges = auto_canny(thresh, 0.8)
        _, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blank = img.copy()#np.zeros(img.shape, dtype=img.dtype)
        targets = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            if self.o.verify_contour(cnt) < 0.4:
                print 'heir ',idx, hierarchy[0][idx]
                if hierarchy[0][idx][3] != -1 and hierarchy[0][idx][2] == -1:  # Only get outer edges
                    cv2.drawContours(blank, [cnt], 0, (0, 0, 255), 3)
                    corners = self.o.get_corners(cnt, debug_image=blank)
                    print 'appending', idx
                    targets.append(corners)
        if self.debug_ros:
            if camera == 'left':
                self.debug_image[0, 0] = blank
                self.debug_image[1, 0] = thresh
            if camera=='right':
                self.debug_image[0, 1] = blank
                self.debug_image[1, 1] = thresh
        return targets

    def points_marker(self, pts, pose, id=0):
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'torpedo_points'
        marker.type = Marker.POINTS
        marker.id = id
        marker.color = numpy_to_colorRGBA(Target.COLORS[id])
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
        print 'RESID', resid
        yaw = np.arctan2(coeff[1], coeff[0])
        quat = quaternion_from_euler(0, 0, yaw)
        return (centroid, quat), resid[0]

    def get_3D_points(self, left_pts, right_pts, id=0):
        print '---'
        #DBG
        diff = left_pts - right_pts
        print 'DIFF ', diff, 'STD', np.std(diff[:, 0])

        # Convert to float points for undistortion
        left_pts = np.array(left_pts, dtype=float).reshape(4, 1, 2)
        right_pts = np.array(right_pts, dtype=float).reshape(4, 1, 2)

        # Undistort points for pose estimation
        left_pts = cv2.undistortPoints(left_pts, self.cam_left.projectionMatrix()[:3, :3], np.zeros(5),
                                       R=self.left_dist, P=self.left_proj)
        right_pts = cv2.undistortPoints(right_pts, self.cam_right.projectionMatrix()[:3, :3], np.zeros(5),
                                        R=self.right_dist, P=self.right_proj)

        # Convert points to their transpose (required for triangulation)
        left_pts = left_pts.reshape(4, 2).T
        right_pts = right_pts.reshape(4, 2).T

        #print 'Left:'
        #print self.cam_left.projectionMatrix()
        #print self.left_proj
        #print np.array(self._camera_info_left.R).reshape(3, 3)
        #print self.left_dist
        #print 'Right:'
        #print self.cam_right.projectionMatrix()
        #print self.right_proj
        #print np.array(self._camera_info_right.R).reshape(3, 3)
        #print self.right_dist

        # Triangulate points
        pts4D = cv2.triangulatePoints(self.left_proj.copy(), self.right_proj.copy(), left_pts.copy(), right_pts.copy())
        pts3D = cv2.convertPointsFromHomogeneous(pts4D.T)
        pts3D = -pts3D.reshape(4, 3)
        pts3D = np.add(pts3D, self.tf_trans)
        if not self.sane_pose_est(pts3D):
            return None

        # Transform pose estimate to map frame
        try:
            self.tf_listener.waitForTransform('/map', self.cam_left.tfFrame(), self.last_image_time_left,
                                              rospy.Duration(0.5))
            tf_trans, tf_quat = self.tf_listener.lookupTransform('/map', self.cam_left.tfFrame(),
                                                                 self.last_image_time_left)
            pts3D_map = np.zeros((4, 3), dtype=float)
            tf_mat = tf.transformations.quaternion_matrix(tf_quat)[:3, :3]
            for idx, pt in enumerate(pts3D):
                pts3D_map[idx] = tf_trans + tf_mat.dot(pt)
            pose, resid = self.get_target_pose(pts3D_map)
            error = resid
            #if resid > 1e-7:
            #    return None
            return pose, pts3D_map, error
        except tf.Exception as e:
            print 'TF Exception {}'.format(e)
            return None

    def match_contours(self, left_contours, right_contours):
        '''
        Attempt to find corosponding target contours in left and right cameras.
        @param left_contours
        @param right_contours
        '''
        #print 'LEFT={} RIGHT={}'.format(len(left_contours), len(right_contours))
        matches = []
        for left_cnt in left_contours:
            left_M = cv2.moments(left_cnt)
            centroid_left = np.array([left_M['m10'] / left_M['m00'], left_M['m01'] / left_M['m00']])
            for right_cnt in right_contours:
                right_M = cv2.moments(right_cnt)
                centroid_right = np.array([right_M['m10'] / right_M['m00'], right_M['m01'] / right_M['m00']])
                r_diff = abs(np.linalg.norm(centroid_left - left_cnt[0]) - np.linalg.norm(centroid_right - right_cnt[0]))
                print r_diff
                if abs(centroid_left[1] - centroid_right[1]) < 10 and r_diff < 1.0:
                    matched = True
                    #for m in matches:
                    #    if np.linalg.norm(m[0] - left_cnt) < 20 or np.linalg.norm(m[1] - right_cnt) < 20:
                    #        matched = False
                    if matched:
                        matches.append([left_cnt, right_cnt])
        return matches

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

    def img_cb(self, left, right):
        left_targets = self.get_target_corners(left, 'left')
        right_targets = self.get_target_corners(right, 'right')
        matches = self.match_contours(left_targets, right_targets)
        print 'matches ', len(matches)
        #print 'Left Targets={}, Right Targets={}, Matches={}'.format(len(left_targets), len(right_targets), len(matches))
        #print '{} MATCHES'.format(len(matches))
        for i, m in enumerate(matches):
            #print 'getting 3d points'
            ret = self.get_3D_points(m[0], m[1], id=i)
            if ret is not None:
                pose, pts3D, error = ret
                found = False
                for idx in xrange(len(self.targets)):
                    if self.targets[idx].update_pose(pose, error):
                        t_id = self.targets[idx].id
                        found = True
                        break
                if not found:
                    self.targets.append(Target(pose, rospy.Time.now(), error))
                    t_id = self.targets[-1].id
                print 'ID', Target.NAMES[t_id]
                self.points_marker(pts3D, pose, id=t_id)
        self.send_debug_image(self.debug_image.image)

if __name__ == '__main__':
    rospy.init_node('torpedo_target')
    TorpedoTargetFinder()
    rospy.spin()
