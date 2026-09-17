#!/usr/bin/env python3
"""
D-PoSE ROS2 Webcam Demo

This script demonstrates real-time 3D human pose estimation using a webcam
and publishes the results to ROS2 topics. It also provides TF transforms
for visualization in RViz.

Requirements:
    - ROS2 (Humble or later)
    - CUDA-capable GPU
    - Webcam connected to the system
    - Trained D-PoSE model checkpoint
    - ArUco marker for camera calibration (optional)

Usage:
    python3 ros_demo_webcam.py [--camera-id 0] [--cfg configs/dpose_conf.yaml]

Author: D-PoSE Team
"""

import os
import sys
import time
import argparse

# --- D-PoSE dependency check ------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DPOSE_DIR = os.path.join(_SCRIPT_DIR, 'D-PoSE')
if not os.path.isdir(_DPOSE_DIR):
    print("ERROR: D-PoSE directory not found at:", _DPOSE_DIR)
    print()
    print("Run the setup script to clone and configure D-PoSE:")
    print("  bash setup.sh")
    print()
    print("Then activate the environment before running this script:")
    print("  source /opt/ros/humble/setup.bash")
    print("  source D-PoSE/venv/bin/activate")
    print("  python3 ros_demo_webcam.py")
    sys.exit(1)
sys.path.insert(0, _DPOSE_DIR)
os.chdir(_DPOSE_DIR)  # D-PoSE internals use relative paths (data/ckpt/...) so run from its directory
# ---------------------------------------------------------------------------

import torch
import numpy as np
import cv2
import math
from loguru import logger

# The script file shares its name with the installed ROS package, so Python
# finds this .py file before the package when searching sys.path.  Remove the
# script's own directory temporarily so the ROS-generated package wins.
_script_dir_real = os.path.realpath(_SCRIPT_DIR)
sys.path = [p for p in sys.path if os.path.realpath(p or '.') != _script_dir_real]

# ROS2 imports
import rclpy
from rclpy.node import Node
from magician_body_pose_estimation.msg import Skeletons, Skeleton, Joint3D
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64
import tf2_ros
# import tf_transformations  # ROS apt package (ros-*-tf-transformations), not on PyPI — re-enable once installed

# D-PoSE specific imports
from train.core.tester import Tester
from train.utils.one_euro_filter import OneEuroFilter
from multi_person_tracker import MPT
from multi_person_tracker import Sort
from aruco.aruco_create import detect_aruco_from_image
from aruco.marker_pose_table import MarkerPoseTable, install_exit_hooks

def getCaptureDeviceFromPath(videoFilePath, videoWidth, videoHeight, videoFramerate=30):
    if videoFilePath == 'webcam' or videoFilePath == '/dev/video0':
        cap = cv2.VideoCapture(0)
    elif videoFilePath.startswith('/dev/video'):
        idx = int(videoFilePath.replace('/dev/video', ''))
        cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(videoFilePath)
    if hasattr(cap, 'set'):
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
    return cap


# Set environment variables for OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def rotmat_to_quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
        
    Returns:
        np.ndarray: Quaternion as [w, x, y, z]
    """
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix"
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])


def degrees_to_radians(deg):
    """Convert degrees to radians."""
    return math.radians(deg)


class PoseEstimationNode(Node):
    """
    ROS2 node for real-time 3D human pose estimation using D-PoSE.
    
    This node captures frames from a webcam, performs 3D human pose estimation,
    and publishes the results as skeleton messages and TF transforms.
    """

    def __init__(self, args):
        """
        Initialize the pose estimation node.
        
        Args:
            args: Command line arguments containing configuration
        """
        super().__init__('dpose_webcam_node')
        
        self.args = args
        self.get_logger().info('Initializing D-PoSE Webcam Node...')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize publishers and broadcasters
        self.skeleton_publisher = self.create_publisher(Skeletons, 'humans', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Initialize tracking and filtering
        self.tracker = Sort()
        self._last_no_detection_log = 0.0  # throttle "no human" warnings to once per 2 s
        self._last_stats_log = 0.0
        self._detection_was_active = False   # tracks state changes (detected → lost)
        self._frames_with_detection = 0
        self._frames_total = 0
        self._first_frame_logged = False
        self.bbox_one_euro_filter = OneEuroFilter(
            np.zeros(4),
            np.zeros(4),
            min_cutoff=0.004,
            beta=0.4,
        )
        
        # Initialize pose estimation model
        self._initialize_model()
        
        # Initialize camera
        self._initialize_camera()
        
        # Initialize person tracker
        self._initialize_tracker()
        
        # ArUco detection state
        self.first_rvec = None
        self.first_tvec = None
        self._initialize_marker_table()
        
        self.get_logger().info('D-PoSE Webcam Node initialized successfully!')

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.args.output_folder
        os.makedirs(log_dir, exist_ok=True)
        
        logger.add(
            os.path.join(log_dir, 'dpose_demo.log'),
            level='INFO',
            colorize=False,
        )
        logger.info(f'D-PoSE Demo options: \n {self.args}')

    def _initialize_model(self):
        """Initialize the D-PoSE model."""
        try:
            self.get_logger().info('Loading D-PoSE model...')
            self.tester = Tester(self.args)
            self.get_logger().info('D-PoSE model loaded successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to load D-PoSE model: {e}')
            raise

    def _try_open_camera(self):
        """Attempt a single camera open and test-grab a frame; return True on success."""
        self.cap = getCaptureDeviceFromPath(
            self.args.input, self.args.width, self.args.height, self.args.fps
        )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # Re-apply resolution after FOURCC — setting MJPG can reset camera to a default resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        if not self.cap.isOpened():
            return False
        ret, _ = self.cap.read()
        return ret

    def _initialize_camera(self):
        """Initialize camera capture, retrying indefinitely when --insist-camera is set."""
        retry_delay = 3.0
        attempt = 0
        while True:
            attempt += 1
            try:
                self.get_logger().info(
                    f'Initializing camera {self.args.input} (attempt {attempt})...'
                )
                if self._try_open_camera():
                    actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.get_logger().info(
                        f'Camera initialized: {int(actual_width)}x{int(actual_height)} @ {actual_fps} FPS'
                    )
                    return
                raise RuntimeError(f"Cannot open camera {self.args.input}")
            except Exception as e:
                self.get_logger().error(f'Failed to initialize camera: {e}')
                if not getattr(self.args, 'insist_camera', False):
                    raise
                self.get_logger().info(
                    f'--insist-camera: retrying in {retry_delay:.0f}s...'
                )
                time.sleep(retry_delay)

    def _initialize_tracker(self):
        """Initialize the multi-person tracker."""
        try:
            self.get_logger().info('Initializing person tracker...')
            
            # Set PyTorch precision for better performance
            torch.set_float32_matmul_precision('medium')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.get_logger().info(f'Using device: {device}')

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cc_major, cc_minor = torch.cuda.get_device_capability(0)
                self.get_logger().info(
                    f'GPU: {gpu_name} | compute capability: {cc_major}.{cc_minor} | '
                    f'CUDA runtime: {torch.version.cuda} | PyTorch: {torch.__version__}'
                )
            else:
                self.get_logger().warning('CUDA not available, using CPU (will be slower)')

            _hardcoded_detector = 'yolo'
            if self.args.detector != _hardcoded_detector:
                self.get_logger().warning(
                    f'--detector={self.args.detector} was requested but MPT is hardcoded '
                    f'to detector_type="{_hardcoded_detector}". '
                    f'The --detector flag is currently ignored.'
                )

            self.mot = MPT(
                device=device,
                batch_size=4,
                display=False,
                detector_type=_hardcoded_detector,
                output_format='dict',
                yolo_img_size=self.args.yolo_img_size
            )
            self.get_logger().info(f'YOLO input image size: {self.args.yolo_img_size}')
            
            self.frame_number = 0
            self.get_logger().info('Person tracker initialized successfully!')
            if not self.args.render:
                self.get_logger().warning(
                    '--render was not passed: 3D mesh overlay is DISABLED. '
                    "Pass --render to see the rendered skeleton in the 'front' OpenCV window."
                )
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize tracker: {e}')
            raise

    def process_frame(self, frame):
        """
        Process a single frame for pose estimation.
        
        Args:
            frame (np.ndarray): Input frame in RGB format
            
        Returns:
            tuple: (detections, hmr_output) or (None, None) if no persons detected
        """
        # Detect persons in the frame
        with torch.amp.autocast('cuda'), torch.no_grad():
            input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0

            if not self._first_frame_logged:
                self.get_logger().info(
                    f'First frame — input tensor: shape={list(input_tensor.shape)}, '
                    f'dtype={input_tensor.dtype}, '
                    f'value range=[{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}]'
                )
                self._first_frame_logged = True

            detection = self.mot.detector(input_tensor.cuda())

            # Process detections
            if detection:
                boxes = torch.cat([pred['boxes'] for pred in detection], dim=0)
                scores = torch.cat([pred['scores'] for pred in detection], dim=0)

                # Check for person-class labels when the detector provides them (e.g. MaskRCNN)
                if all('labels' in pred for pred in detection):
                    labels = torch.cat([pred['labels'] for pred in detection], dim=0)
                    person_boxes = (labels == 1).sum().item()  # COCO class 1 = person
                    if person_boxes == 0 and len(labels) > 0:
                        now = time.time()
                        if now - self._last_no_detection_log >= 2.0:
                            unique_classes = labels.unique().tolist()
                            self.get_logger().warning(
                                f'Detector found {len(labels)} box(es) but NONE are person-class (1). '
                                f'Detected COCO classes: {unique_classes}. '
                                'The model may be mis-configured or using wrong weights.'
                            )
                            self._last_no_detection_log = now

                # Apply confidence threshold
                mask = scores > self.args.detection_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask].unsqueeze(1)

                if filtered_boxes.numel() > 0:
                    dets = torch.cat([filtered_boxes, filtered_scores], dim=1).cpu().detach().numpy()
                else:
                    dets = np.empty((0, 5))
                    now = time.time()
                    if now - self._last_no_detection_log >= 2.0:
                        if scores.numel() > 0:
                            score_info = (
                                f'Score range: {scores.min().item():.3f}–{scores.max().item():.3f} '
                                f'(mean {scores.mean().item():.3f}). '
                                'Consider lowering --detection-threshold.'
                            )
                        else:
                            score_info = 'Detector returned prediction dicts with no boxes inside.'
                        self.get_logger().warning(
                            f'No person detected above confidence threshold '
                            f'({self.args.detection_threshold}). '
                            f'Raw detector returned {len(boxes)} box(es) — all below threshold. '
                            + score_info
                        )
                        self._last_no_detection_log = now
            else:
                dets = np.empty((0, 5))
                now = time.time()
                if now - self._last_no_detection_log >= 2.0:
                    self.get_logger().warning(
                        'No person detected: detector returned no boxes for this frame.'
                    )
                    self._last_no_detection_log = now
            
            # Update tracker
            if dets.shape[0] > 0:
                track_bbs_ids = self.tracker.update(dets)
            else:
                track_bbs_ids = np.empty((0, 5))
            
            # Prepare detections for pose estimation
            detections = [dets]
            detection = self.mot.prepare_output_detections(detections)
            
            self._frames_total += 1

            if len(detection[0]) > 0:
                self._frames_with_detection += 1
                if not self._detection_was_active:
                    self.get_logger().info(
                        f'Human detected (frame {self._frames_total}). '
                        f'Detection rate so far: '
                        f'{self._frames_with_detection}/{self._frames_total} frames.'
                    )
                    self._detection_was_active = True
                hmr_output = self.tester.run_on_single_image_tensor(frame, detection, render=self.args.render)
                return track_bbs_ids, hmr_output
            else:
                if self._detection_was_active:
                    self.get_logger().warning(
                        f'Human tracking lost (frame {self._frames_total}). '
                        f'Detection rate so far: '
                        f'{self._frames_with_detection}/{self._frames_total} frames.'
                    )
                    self._detection_was_active = False
                now = time.time()
                if now - self._last_no_detection_log >= 2.0:
                    self.get_logger().warning(
                        'No human detected after tracking — skipping pose estimation. '
                        'Check lighting, camera angle, or lower --detection-threshold '
                        f'(currently {self.args.detection_threshold}).'
                    )
                    self._last_no_detection_log = now
                if now - self._last_stats_log >= 30.0:
                    self.get_logger().info(
                        f'Detection rate (last 30 s window): '
                        f'{self._frames_with_detection}/{self._frames_total} frames had a human.'
                    )
                    self._last_stats_log = now
                display_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    display_bgr,
                    'No human detected',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow('front', display_bgr)

        return None, None

    def publish_skeletons(self, track_bbs_ids, hmr_output):
        """
        Publish skeleton data to ROS topic.
        
        Args:
            track_bbs_ids: Tracking IDs and bounding boxes
            hmr_output: HMR model output containing 3D joints
        """
        skeletons_msg = Skeletons()
        skeletons_msg.humans = []
        
        # Extract 3D joints and camera translation
        hmr_joints = hmr_output['joints3d'][:, 0:22, :].cpu().numpy()
        camera_translation = hmr_output['pred_cam_t'].cpu().numpy() * 0.5
        
        current_time = self.get_clock().now().to_msg()
        
        marker_R = None
        marker_t = None
        if self.args.use_aruco:
            if self.first_rvec is None or self.first_tvec is None:
                self.get_logger().warning(
                    'No ArUco marker pose yet, not publishing skeletons',
                    throttle_duration_sec=2.0,
                )
                return
            marker_R = cv2.Rodrigues(self.first_rvec)[0]
            marker_t = self.first_tvec.reshape(3)
        
        for i in range(len(track_bbs_ids)):
            human = Skeleton()
            human.id = int(track_bbs_ids[i][-1])
            human.joints = []
            
            # Get joints for this person in the camera frame
            # (OpenCV axes: x right, y down, z forward)
            joints = hmr_joints[i] + camera_translation[i]

            # Express joints w.r.t. the ArUco marker: p_marker = R^T (p_camera - t)
            if marker_R is not None:
                joints = (joints - marker_t) @ marker_R

            # Convert joints to ROS message format
            for j, joint in enumerate(joints):
                joint3d = Joint3D()
                joint3d.x = float(joint[0])
                joint3d.y = float(joint[1])
                joint3d.z = float(joint[2])
                human.joints.append(joint3d)
                

                # Publish TF transform for the root joint (pelvis)
                # if j == 0:
                #     self._publish_joint_transform(
                #         current_time, human.id, j, joint3d
                #     )

            self._publish_human_camera_transform(current_time, human.id, human)
            skeletons_msg.humans.append(human)
        
        # Publish the skeleton message
        self.skeleton_publisher.publish(skeletons_msg)

    def _publish_joint_transform(self, timestamp, human_id, joint_id, joint):
        """
        Publish TF transform for a joint.
        
        Args:
            timestamp: ROS timestamp
            human_id: Human ID
            joint_id: Joint ID
            joint: Joint3D message
        """
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'Camera'
        t.child_frame_id = f'human_{human_id}_joint_{joint_id}'
        
        t.transform.translation.x = joint.x
        t.transform.translation.y = joint.y
        t.transform.translation.z = joint.z
        
        # Set orientation (assuming no rotation for joints)
        t.transform.rotation.x = -1.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)

    def _publish_human_camera_transform(self, timestamp, human_id, human):
        """
        Publish TF transform for a joint.
        
        Args:
            timestamp: ROS timestamp
            human_id: Human ID
            joint_id: Joint ID
            joint: Joint3D message
        """
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'Aruco_marker' if self.args.use_aruco else 'Camera'
        # t.child_frame_id = f'human_{human_id}_joint_{joint_id}'
        t.child_frame_id = f'human_{human_id}'
        
        t.transform.translation.x = human.joints[0].x
        t.transform.translation.y = human.joints[0].y
        t.transform.translation.z = human.joints[0].z
        
        pelvis = np.array([human.joints[0].x, human.joints[0].y, human.joints[0].z])
        left_hip = np.array([human.joints[1].x, human.joints[1].y, human.joints[1].z])
        right_hip = np.array([human.joints[2].x, human.joints[2].y, human.joints[2].z])
        neck = np.array([human.joints[12].x, human.joints[12].y, human.joints[12].z])

        z_axis = neck-pelvis
        z_axis /= np.linalg.norm(z_axis)

        x_axis = right_hip - left_hip
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))

        # q = tf_transformations.quaternion_from_matrix(  # needs ros-*-tf-transformations (apt), returns (x,y,z,w)
        #     np.vstack((np.column_stack((R, [0,0,0])), [0,0,0,1])))
        q = rotmat_to_quat(R)  # returns (w, x, y, z)

        t.transform.rotation.w = float(q[0])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])
        
        self.tf_broadcaster.sendTransform(t)

    def _initialize_marker_table(self):
        """Set up the averaged marker poses and the slider subscription."""
        table_file = self.args.marker_table_file or os.path.join(
            self.args.output_folder, 'aruco_marker_table.json')
        self.marker_table = MarkerPoseTable(
            static_camera=self.args.static_camera,
            static_robot=self.args.static_robot,
            bin_size=self.args.slider_bin,
            min_samples=self.args.marker_min_samples,
            spread_threshold=self.args.marker_spread_threshold,
            recheck_interval=self.args.marker_recheck_interval,
            alarm_pct=self.args.marker_alarm_pct,
            line_fit=self.args.marker_line_fit,
            fit_residual=self.args.marker_fit_residual,
            table_file=table_file,
            logger=self.get_logger(),
        )

        if not (self.args.use_aruco and self.args.static_camera):
            return

        install_exit_hooks(self.marker_table)
        if self.args.static_robot:
            self.get_logger().info('Static robot: averaging all marker observations '
                                   'into a single slider position')
        else:
            self.slider_subscription = self.create_subscription(
                Float64, self.args.slider_topic,
                lambda msg: self.marker_table.set_slider(float(msg.data)), 10)
            self.get_logger().info(
                f'Averaging marker poses per slider position from {self.args.slider_topic}')

    # TODO: Move this to the robot side (so no fixed envirnoment components are present here)
    def publish_aruco_transforms(self, timestamp):
        """
        Publish ArUco marker transforms.
        
        Args:
            timestamp: ROS timestamp
        """
        # Publish base ArUco transform
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.args.aruco_parent_frame
        t.child_frame_id = 'Aruco_marker'
        
        t.transform.translation.x = self.args.aruco_xyz[0]
        t.transform.translation.y = self.args.aruco_xyz[1]
        t.transform.translation.z = self.args.aruco_xyz[2]
        
        # Fixed-axis roll (X), pitch (Y), yaw (Z): R = Rz(yaw) Ry(pitch) Rx(roll)
        roll, pitch, yaw = (degrees_to_radians(a) for a in self.args.aruco_rpy)
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        q = rotmat_to_quat(Rz @ Ry @ Rx)  # returns (w, x, y, z)
        t.transform.rotation.w = float(q[0])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])
        
        self.tf_broadcaster.sendTransform(t)
        
        # Publish camera transform relative to ArUco marker
        if self.first_rvec is not None and self.first_tvec is not None:
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = 'Aruco_marker'
            t.child_frame_id = 'Camera'
            
            # Convert rotation vector to rotation matrix and invert
            R = cv2.Rodrigues(self.first_rvec)[0]
            R_inv = R.T
            
            # Invert translation
            tvec = self.first_tvec.reshape(3)
            t_inv = -np.dot(R_inv, tvec)
            
            t.transform.translation.x = float(t_inv[0])
            t.transform.translation.y = float(t_inv[1])
            t.transform.translation.z = float(t_inv[2])
            
            # Convert rotation matrix to quaternion
            quat = rotmat_to_quat(R_inv)
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            t.transform.rotation.w = float(quat[0])
            
            self.tf_broadcaster.sendTransform(t)

    def run(self):
        """
        Main processing loop.
        """
        self.get_logger().info('Starting pose estimation loop...')
        self.get_logger().info('Press "q" in the OpenCV window to quit')
        
        try:
            while rclpy.ok():
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error('Failed to capture frame from camera')
                    break
                
                self.frame_number += 1
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[1] != self.args.width or frame.shape[0] != self.args.height:
                    self.get_logger().warning(
                        f'Frame size mismatch: camera delivered {frame.shape[1]}x{frame.shape[0]} '
                        f'but {self.args.width}x{self.args.height} was requested — resizing. '
                        'This may reduce detection accuracy.'
                    )
                    frame = cv2.resize(frame, (self.args.width, self.args.height))

                # Detect ArUco markers (for camera calibration)
                if self.args.use_aruco:
                    if self.marker_table.should_detect():
                        rvec, tvec = detect_aruco_from_image(
                            frame,
                            marker_id=self.args.aruco_marker_id,
                            fx=self.args.fx, fy=self.args.fy,
                            cx=self.args.cx, cy=self.args.cy,
                            dist_coeffs=self.args.dist_coeffs,
                            marker_length=self.args.aruco_marker_length,
                            # frame was already converted to RGB above
                            input_is_bgr=False,
                        )
                        if rvec is not None and tvec is not None:
                            self.marker_table.add(rvec, tvec)
                    estimate = self.marker_table.estimate()
                    if estimate is not None:
                        self.first_rvec, self.first_tvec = estimate
                
                # Process frame for pose estimation
                track_bbs_ids, hmr_output = self.process_frame(frame)
                
                current_time = self.get_clock().now().to_msg()
                
                if track_bbs_ids is not None and hmr_output is not None:
                    # Publish skeleton data
                    self.publish_skeletons(track_bbs_ids, hmr_output)
                    
                    # Publish ArUco transforms
                    if self.args.use_aruco:
                        self.publish_aruco_transforms(current_time)
                
                # Pump the OpenCV event loop so the 'front' window (managed by D-PoSE
                # tester) actually refreshes on screen.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info('Quit requested by user')
                    break
                rclpy.spin_once(self, timeout_sec=0.001)
                
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted by user')
        except Exception as e:
            self.get_logger().error(f'Error in processing loop: {e}')
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info('Cleaning up resources...')
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if hasattr(self, 'tester') and hasattr(self.tester, 'model'):
            del self.tester.model
        
        self.marker_table.save()
        
        logger.info('================= END =================')
        self.get_logger().info('Cleanup completed')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='D-PoSE ROS2 Webcam Demo for real-time 3D human pose estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--cfg', type=str,
        default=os.path.join(_DPOSE_DIR, 'configs', 'dpose_conf.yaml'),
        help='Config file that defines model hyperparameters'
    )
    parser.add_argument(
        '--ckpt', type=str,
        default=os.path.join(_DPOSE_DIR, 'data', 'ckpt', 'paper_arxiv.ckpt'),
        help='Path to model checkpoint file'
    )
    
    # Camera configuration
    parser.add_argument(
        '--input', type=str, default='/dev/video0',
        help='Input device or file (e.g. /dev/video0, /dev/video8, webcam, or a video file path)'
    )
    parser.add_argument(
        '--width', type=int, default=1920,
        help='Camera capture width in pixels'
    )
    parser.add_argument(
        '--height', type=int, default=1080,
        help='Camera capture height in pixels'
    )
    parser.add_argument(
        '--fps', type=int, default=15,
        help='Camera capture frame rate'
    )
    
    # Processing configuration
    parser.add_argument(
        '--detection-threshold', type=float, default=0.7,
        help='Confidence threshold for person detection (0.0-1.0)'
    )
    parser.add_argument(
        '--tracker-batch-size', type=int, default=1,
        help='Batch size for object detector used for bbox tracking'
    )
    
    # Display and output options
    parser.add_argument(
        '--display', action='store_true',
        help='Show real-time video window (press "q" to quit)'
    )
    parser.add_argument(
        '--output-folder', type=str, default='./logs',
        help='Output folder for log files'
    )
    
    # ArUco marker options
    parser.add_argument(
        '--use-aruco', action=argparse.BooleanOptionalAction, default=True,
        help='Enable ArUco marker detection for camera calibration. Pass '
             '--no-use-aruco to disable and publish in the raw camera frame'
    )
    parser.add_argument(
        '--aruco-marker-id', type=int, default=None,
        help='Only use this ArUco marker ID (DICT_6X6_250) as the skeleton '
             'reference, e.g. 1. Other markers are ignored. Default: any marker'
    )
    parser.add_argument(
        '--aruco-marker-length', type=float, default=0.15,
        help='Printed ArUco marker side length in meters (black border edge to edge). '
             'Scales the estimated camera translation'
    )
    parser.add_argument(
        '--aruco-parent-frame', type=str, default='wood_panel',
        help='TF parent frame the ArUco marker is mounted on'
    )
    parser.add_argument(
        '--aruco-xyz', type=float, nargs=3, default=[0.1055, 1.405, -0.1025],
        metavar=('X', 'Y', 'Z'),
        help='ArUco marker position in the parent frame, in meters'
    )
    parser.add_argument(
        '--aruco-rpy', type=float, nargs=3, default=[90.0, 0.0, 180.0],
        metavar=('ROLL', 'PITCH', 'YAW'),
        help='ArUco marker orientation in the parent frame, in degrees '
             '(fixed-axis roll about X, then pitch about Y, then yaw about Z). '
             'Marker axes: X right, Y up along the printed marker, Z out of the marker'
    )
    
    parser.add_argument(
        '--static-camera', action='store_true',
        help='The camera never moves, so the marker pose only changes when the robot '
             'does. Marker observations are then averaged per slider position, which '
             'removes detection noise, survives occlusion and lets detection be skipped '
             'once a position has converged'
    )
    parser.add_argument(
        '--static-robot', action='store_true',
        help='The robot never moves along its slider, so all observations belong to a '
             'single position and the slider topic is not needed (implies a slider of 0)'
    )
    parser.add_argument(
        '--slider-topic', type=str, default='/slider/position_y',
        help='std_msgs/Float64 topic carrying the robot slider position in meters'
    )
    parser.add_argument(
        '--slider-bin', type=float, default=0.01,
        help='Slider positions this far apart (meters) share one averaged marker pose'
    )
    parser.add_argument(
        '--marker-table-file', type=str, default='',
        help='Where the averaged marker poses are saved on shutdown and reloaded from '
             'on start. Default: <output-folder>/aruco_marker_table.json'
    )
    parser.add_argument(
        '--marker-min-samples', type=int, default=30,
        help='Observations a slider position needs before its averaged marker pose is '
             'trusted enough to skip detection'
    )
    parser.add_argument(
        '--marker-spread-threshold', type=float, default=0.01,
        help='Maximum spread (standard deviation, meters) of the averaged marker '
             'position for it to count as converged'
    )
    parser.add_argument(
        '--marker-recheck-interval', type=float, default=5.0,
        help='Once converged, how often (seconds) the marker is detected again to '
             'catch a bumped camera'
    )
    parser.add_argument(
        '--no-marker-line-fit', dest='marker_line_fit', action='store_false',
        help='Do not fit a straight line through the learned slider positions. By '
             'default the marker pose at a slider position that was never visited is '
             'interpolated from that line, which never overrides a position that has '
             'been measured directly'
    )
    parser.add_argument(
        '--marker-fit-residual', type=float, default=0.02,
        help='Largest residual (meters) the rail line fit may have before '
             'interpolation is refused, e.g. because the slider is not metric or the '
             'marker moved on its mount'
    )
    parser.add_argument(
        '--marker-alarm-pct', type=float, default=5.0,
        help='After loading poses from disk, warn while live observations disagree by '
             'more than this percentage of the marker distance (possible tampering '
             'while the node was down)'
    )
    
    # Camera intrinsics (used for ArUco pose estimation, at --width x --height)
    parser.add_argument('--fx', type=float, default=800.0, help='Camera focal length x in pixels')
    parser.add_argument('--fy', type=float, default=800.0, help='Camera focal length y in pixels')
    parser.add_argument('--cx', type=float, default=320.0, help='Camera principal point x in pixels')
    parser.add_argument('--cy', type=float, default=240.0, help='Camera principal point y in pixels')
    parser.add_argument(
        '--dist-coeffs', type=float, nargs='+', default=None,
        metavar='K',
        help='Lens distortion coefficients in OpenCV order (k1 k2 p1 p2 [k3 ...]). '
             'Default: no distortion'
    )

    # Camera retry option
    parser.add_argument(
        '--insist-camera', action='store_true',
        help='Retry camera initialization indefinitely (3 s delay between attempts) until it succeeds'
    )
    
    # Detector configuration (for compatibility)
    parser.add_argument(
        '--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
        help='Object detector to use for person detection'
    )
    parser.add_argument(
        '--yolo-img-size', type=int, default=256,
        help='Input image size for YOLO detector'
    )
    
    #Render 3d Mesh
    parser.add_argument(
        '--render', action='store_true',
        help='Render the 3D mesh on the OpenCV window'
    )
    
    # Deprecated/unused options (kept for compatibility)
    parser.add_argument('--image-folder', type=str, default='demo_images', help=argparse.SUPPRESS)
    parser.add_argument('--eval-dataset', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dataframe-path', type=str, default='data/ssp_3d_test.npz', help=argparse.SUPPRESS)
    parser.add_argument('--data-split', type=str, default='test', help=argparse.SUPPRESS)
    
    return parser.parse_args()


def validate_requirements():
    """Validate that all requirements are met."""
    errors = []
    
    # Check for CUDA
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. This demo requires a CUDA-capable GPU for optimal performance.")
    
    # Check for required files (resolved relative to the D-PoSE subdirectory)
    required_files = [
        os.path.join(_DPOSE_DIR, 'configs', 'dpose_conf.yaml'),
        os.path.join(_DPOSE_DIR, 'data', 'ckpt', 'paper_arxiv.ckpt'),
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            errors.append(f"Required file not found: {file_path}")
    
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"   • {error}")
        print("\n💡 Please make sure you have:")
        print("   1. Set up the environment as described in README.md")
        print("   2. Downloaded the required model checkpoint")
        print("   3. A CUDA-capable GPU available")
        return False
    
    print("✅ All requirements validated successfully!")
    return True


@torch.no_grad()
def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate requirements
    if not validate_requirements():
        return 1
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create and run the node
        node = PoseEstimationNode(args)
        node.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Shutdown ROS2
        if rclpy.ok():
            rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
