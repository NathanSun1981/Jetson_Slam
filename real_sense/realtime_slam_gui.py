import argparse
import json
import shutil
import sys
from enum import IntEnum
from os import makedirs
from os.path import abspath, exists, join
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import cv2
import numpy as np
import time
import threading
# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import open3d as o3d
import signal
import readchar
import paramiko

from config import ConfigParser

from common import (extract_rgbd_frames, extract_trianglemesh,
                    get_default_dataset, load_intrinsic, load_rgbd_file_names,
                    save_poses, get_profiles)

sys.path.append(abspath(__file__))


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

class ReconstructionWindow:

    def __init__(self, config, font_id):
        self.config = config

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in fixed props
        self.fixed_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        ### Depth scale slider
        scale_label = gui.Label('Depth scale')
        self.scale_slider = gui.Slider(gui.Slider.INT)
        self.scale_slider.set_limits(1000, 5000)
        self.scale_slider.int_value = int(config.depth_scale)
        self.fixed_prop_grid.add_child(scale_label)
        self.fixed_prop_grid.add_child(self.scale_slider)

        voxel_size_label = gui.Label('Voxel size')
        self.voxel_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_size_slider.set_limits(0.003, 0.01)
        self.voxel_size_slider.double_value = config.voxel_size
        self.fixed_prop_grid.add_child(voxel_size_label)
        self.fixed_prop_grid.add_child(self.voxel_size_slider)

        trunc_multiplier_label = gui.Label('Trunc multiplier')
        self.trunc_multiplier_slider = gui.Slider(gui.Slider.DOUBLE)
        self.trunc_multiplier_slider.set_limits(1.0, 20.0)
        self.trunc_multiplier_slider.double_value = config.trunc_voxel_multiplier
        self.fixed_prop_grid.add_child(trunc_multiplier_label)
        self.fixed_prop_grid.add_child(self.trunc_multiplier_slider)

        est_block_count_label = gui.Label('Est. blocks')
        self.est_block_count_slider = gui.Slider(gui.Slider.INT)
        self.est_block_count_slider.set_limits(40000, 100000)
        self.est_block_count_slider.int_value = config.block_count
        self.fixed_prop_grid.add_child(est_block_count_label)
        self.fixed_prop_grid.add_child(self.est_block_count_slider)

        est_point_count_label = gui.Label('Est. points')
        self.est_point_count_slider = gui.Slider(gui.Slider.INT)
        self.est_point_count_slider.set_limits(500000, 8000000)
        self.est_point_count_slider.int_value = config.est_point_count
        self.fixed_prop_grid.add_child(est_point_count_label)
        self.fixed_prop_grid.add_child(self.est_point_count_slider)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Reconstruction interval
        interval_label = gui.Label('Recon. interval')
        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(1, 500)
        self.interval_slider.int_value = 50
        self.adjustable_prop_grid.add_child(interval_label)
        self.adjustable_prop_grid.add_child(self.interval_slider)

        ### Depth max slider
        max_label = gui.Label('Depth max')
        self.max_slider = gui.Slider(gui.Slider.DOUBLE)
        self.max_slider.set_limits(3.0, 6.0)
        self.max_slider.double_value = config.depth_max
        self.adjustable_prop_grid.add_child(max_label)
        self.adjustable_prop_grid.add_child(self.max_slider)

        ### Depth diff slider
        diff_label = gui.Label('Depth diff')
        self.diff_slider = gui.Slider(gui.Slider.DOUBLE)
        self.diff_slider.set_limits(0.07, 0.5)
        self.diff_slider.double_value = config.odometry_distance_thr
        self.adjustable_prop_grid.add_child(diff_label)
        self.adjustable_prop_grid.add_child(self.diff_slider)

        ### Update surface?
        update_label = gui.Label('Update surface?')
        self.update_box = gui.Checkbox('')
        self.update_box.checked = True
        self.adjustable_prop_grid.add_child(update_label)
        self.adjustable_prop_grid.add_child(self.update_box)

        ### Ray cast color?
        raycast_label = gui.Label('Raycast color?')
        self.raycast_box = gui.Checkbox('')
        self.raycast_box.checked = True
        self.adjustable_prop_grid.add_child(raycast_label)
        self.adjustable_prop_grid.add_child(self.raycast_box)

        set_enabled(self.fixed_prop_grid, True)

        ## Application control
        b = gui.ToggleSwitch('Pause/Start')
        b.set_on_clicked(self._on_switch)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.raycast_color_image = gui.ImageWidget()
        self.raycast_depth_image = gui.ImageWidget()
        tab2.add_child(self.raycast_color_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.raycast_depth_image)
        tabs.add_tab('Raycast images', tab2)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Starting settings'))
        self.panel.add_child(self.fixed_prop_grid)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = False
        self.is_running = False
        self.is_surface_updated = False

        self.idx = 0
        self.poses = []

        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.pipeline = rs.pipeline()

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running
        """if not self.is_running:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_close) """


    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        device = o3d.core.Device("CPU:0")

        max_points = self.est_point_count_slider.int_value

        """ pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32)) """

        pcd_placeholder = o3d.t.geometry.PointCloud(device)

        pcd_placeholder.point["positions"] = o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32))
        pcd_placeholder.point["colors"] =  o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32))

        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        path_bag = join(self.config.output_folder, "realsense.bag")
        if self.config.record_rosbag:
            if exists(path_bag):
                user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
                if user_input.lower() == 'n':
                    exit()

        # Create a pipeline
        #Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        rs_config = rs.config()
        # note: using 640 x 480 depth resolution produces smooth depth boundaries
        #       using rs.format.bgr8 for color image format for OpenCV based image visualization
        """ print('Using the default profiles: \n  color:{}, depth:{}'.format(
            color_profiles[30], depth_profiles[1]))
        w, h, fps, fmt = depth_profiles[1]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = color_profiles[30]
        config.enable_stream(rs.stream.color, w, h, fmt, fps) """
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        if self.config.record_rosbag:
            rs_config.enable_record_to_file(path_bag)

        # Start streaming
        profile = self.pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        # Using preset HighAccuracy for recording
        if self.config.record_rosbag:
            depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_scale = depth_sensor.get_depth_scale()
        # We will not display the background of objects more than
        #  clipping_distance_in_meters meters away

        self.clipping_distance = self.config.clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        T_frame_to_model = o3c.Tensor(np.identity(4))


        self.model = o3d.t.pipelines.slam.Model(
            self.voxel_size_slider.double_value, 16,
            self.est_block_count_slider.int_value, T_frame_to_model,
            o3c.Device(self.config.device))
        self.is_started = True

        set_enabled(self.fixed_prop_grid, False)
        set_enabled(self.adjustable_prop_grid, True)

    def _on_close(self):
        self.is_done = True

        if self.is_started:
            print('Saving model to {}...'.format(self.config.path_npz))
            self.model.voxel_grid.save(self.config.path_npz)
            print('Finished.')

            mesh_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.ply'
            print('Extracting and saving mesh to {}...'.format(mesh_fname))
            mesh = extract_trianglemesh(self.model.voxel_grid, self.config,
                                        mesh_fname)
            print('Finished.')

            pc_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.pcd'

            if self.config.engine == 'legacy':
                pc = self.model.extract_pointcloud().transform(self.flip_transform)
            else:
                pc = self.model.extract_pointcloud().transform(self.flip_transform).to_legacy()
            print('Extracting and saving point cloud to {}...'.format(pc_fname))
            o3d.io.write_point_cloud(pc_fname, pc)
            print('Finished.')

            if self.config.upload:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.config.ip, username=self.config.username, password=self.config.password)
                sftp = ssh.open_sftp()
                sftp.put(pc_fname, '/mnt/output.ply')
                sftp.close()
                ssh.close()       

            log_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.log'
            print('Saving trajectory to {}...'.format(log_fname))
            save_poses(log_fname, self.poses)
            print('Finished.')

            self.pipeline.stop()
            self.is_started = False
            self.is_running = False

        return True

    def init_render(self, depth_ref, color_ref):
        self.input_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     self.config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(color_ref.to_legacy())

        self.raycast_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     self.config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.raycast_color_image.update_image(color_ref.to_legacy())
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def update_render(self, input_depth, input_color, raycast_depth,
                      raycast_color, pcd, frustum):
        self.input_depth_image.update_image(
            input_depth.colorize_depth(
                float(self.scale_slider.int_value), self.config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(input_color.to_legacy())

        self.raycast_depth_image.update_image(
            raycast_depth.colorize_depth(
                float(self.scale_slider.int_value), self.config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.raycast_color_image.update_image(
            (raycast_color).to(o3c.uint8, False, 255.0).to_legacy())

        if self.is_scene_updated:
            if pcd is not None and pcd.point['positions'].shape[0] > 0:
                self.widget3d.scene.scene.update_geometry(
                    'points', pcd, rendering.Scene.UPDATE_POINTS_FLAG |
                    rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)

    # Major loop
    def update_main(self):

        device = o3d.core.Device(self.config.device)

        T_frame_to_model = o3c.Tensor(np.identity(4))

        fps_interval_len = 30
        self.idx = 0
        pcd = None

        print("start update scene")
       
        while not self.is_done:
            if not self.is_started or not self.is_running:
                time.sleep(0.05)
                continue    
            #get start configration
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()


            #write instrinsic in the first frame.
            if self.idx == 0:
                print("get first frame and save intrinsic ")
                profile = color_frame.profile.as_video_stream_profile().intrinsics
                intrinsic_json = json.dumps(
                {           
                    'width': profile.width,
                    'height': profile.height,
                    'intrinsic_matrix': [
                        profile.fx, 
                        0, 
                        0, 
                        0, 
                        profile.fy, 
                        0, 
                        profile.ppx, 
                        profile.ppy, 
                        1]
                },
                indent = 4
                )

                with open(self.config.path_intrinsic, "w") as outfile:
                    json_obj = json.dump(
                        json.loads(intrinsic_json),
                        outfile,
                        indent = 4)
                    

                # Validate that both frames are valid
            

            depth_array = np.array(aligned_depth_frame.get_data())
            color_array = np.asarray(color_frame.get_data())

            color_ref = o3d.t.geometry.Image(color_array)
            depth_ref = o3d.t.geometry.Image(depth_array)
            depth = depth_ref.to(device)
            color = color_ref.to(device) 

            intrinsic = load_intrinsic(self.config)  
            start = time.time()

            if self.idx == 0:
                print("process first frame")
                input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                                        intrinsic, device)
                raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                        depth_ref.columns, intrinsic,
                                                        device)
                raycast_frame.set_data_from_image('depth', depth)
                raycast_frame.set_data_from_image('color', color)      
                gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.init_render(depth_ref, color_ref))  
 


            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)

            if self.idx > 0:
                print("get {} frame ".format(self.idx))
                result = self.model.track_frame_to_model(
                    input_frame,
                    raycast_frame,
                    float(self.scale_slider.int_value),
                    self.max_slider.double_value,
                )
                T_frame_to_model = T_frame_to_model @ result.transformation

            self.poses.append(T_frame_to_model.cpu().numpy())
            self.model.update_frame_pose(self.idx, T_frame_to_model)
            self.model.integrate(input_frame,
                                 float(self.scale_slider.int_value),
                                 self.max_slider.double_value,
                                 self.trunc_multiplier_slider.double_value)
            self.model.synthesize_model_frame(
                raycast_frame, float(self.scale_slider.int_value),
                self.config.depth_min, self.max_slider.double_value,
                self.trunc_multiplier_slider.double_value,
                self.raycast_box.checked)

            if (self.idx % self.interval_slider.int_value == 0 and
                    self.update_box.checked):
                print("update scene")
                pcd = self.model.voxel_grid.extract_point_cloud(
                    3.0, self.est_point_count_slider.int_value).to(
                        o3d.core.Device('CPU:0'))
                self.is_scene_updated = True
            else:
                self.is_scene_updated = False

            frustum = o3d.geometry.LineSet.create_camera_visualization(
                color.columns, color.rows, intrinsic.numpy(),
                np.linalg.inv(T_frame_to_model.cpu().numpy()), 0.2)
            frustum.paint_uniform_color([0.961, 0.475, 0.000])

            # Output FPS
            if (self.idx % fps_interval_len == 0):
                end = time.time()
                elapsed = end - start
                start = time.time()
                self.output_fps.text = 'FPS: {:.3f}'.format(fps_interval_len /
                                                            elapsed)

            # Output info
            info = 'Frame {}\n\n'.format(self.idx)
            info += 'Transformation:\n{}\n'.format(
                np.array2string(T_frame_to_model.numpy(),
                                precision=3,
                                max_line_width=40,
                                suppress_small=True))
            info += 'Active voxel blocks: {}/{}\n'.format(
                self.model.voxel_grid.hashmap().size(),
                self.model.voxel_grid.hashmap().capacity())
            info += 'Surface points: {}/{}\n'.format(
                0 if pcd is None else pcd.point['positions'].shape[0],
                self.est_point_count_slider.int_value)

            self.output_info.text = info

            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render(
                    input_frame.get_data_as_image('depth'),
                    input_frame.get_data_as_image('color'),
                    raycast_frame.get_data_as_image('depth'),
                    raycast_frame.get_data_as_image('color'), pcd, frustum))

            self.idx += 1


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)


if __name__ == "__main__":
  
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a ' 
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--output_folder',
            help='set output folder',
            default='./dataset/realsense/')
    parser.add('--record_rosbag',
            help='Recording rgbd stream into realsense.bag')   

    parser.add('--path_npz',
        help='path to the npz file that stores voxel block grid.',
        default='output.npz') 

    parser.add('--path_ply',
        help='path to the npz file that stores voxel block grid.',
        default='output.ply') 

    parser.add('--upload',
        help='path to the npz file that stores voxel block grid.',
        default=False) 

    parser.add('--display',
        help='if show reconstructed point cloud.',
        default=False) 
    
    parser.add('--ip',
        help='ip address to upload result.',
        default='192.168.1.35')

    parser.add('--port',
        help='FTP port to upload result.',
        default='22')

    parser.add('--username',
        help='username to ligin server.',
        default='nathans')
    
    parser.add('--password',
        help='password to login server.',
        default='nathans')

    config = parser.get_config()

    """ if sum(o is not False for o in vars(config).values()) != 2:
        parser.print_help()
        #exit() """

    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = ReconstructionWindow(config, mono)
    app.run()