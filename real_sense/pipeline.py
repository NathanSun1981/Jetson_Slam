import argparse
import json
import shutil
import sys
from enum import IntEnum
from os import makedirs
from os.path import abspath, exists, join

import cv2
import numpy as np
import time
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

try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass


exit_flag = False


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def handler(signum, frame):
    global exit_flag
    res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        exit_flag = True
 

def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()


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

    signal.signal(signal.SIGINT, handler)

    

     
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

    args = parser.get_config()

    if sum(o is not False for o in vars(args).values()) != 2:
        parser.print_help()
        #exit()

    path_output = args.output_folder

    path_bag = join(args.output_folder, "realsense.bag")
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    color_profiles, depth_profiles = get_profiles()


    if args.record_rosbag:
        # note: using 640 x 480 depth resolution produces smooth depth boundaries
        #       using rs.format.bgr8 for color image format for OpenCV based image visualization
        print('Using the default profiles: \n  color:{}, depth:{}'.format(
            color_profiles[30], depth_profiles[1]))
        w, h, fps, fmt = depth_profiles[1]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = color_profiles[30]
        config.enable_stream(rs.stream.color, w, h, fmt, fps)
        config.enable_record_to_file(path_bag)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    if args.record_rosbag:
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = o3d.core.Device(args.device)
    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(args.voxel_size, 16,
                                            args.block_count, T_frame_to_model,
                                            device)


    # Streaming loop
    frame_count = 0
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

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

            with open(args.path_intrinsic, "w") as outfile:
                json_obj = json.dump(
                    json.loads(intrinsic_json),
                    outfile,
                    indent = 4)
                

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_array = np.array(aligned_depth_frame.get_data())
            color_array = np.asarray(color_frame.get_data())

            color_ref = o3d.t.geometry.Image(color_array)
            depth_ref = o3d.t.geometry.Image(depth_array)
            depth = depth_ref.to(device)
            color = color_ref.to(device) 
            
            start = time.time()
            #do slam
            if frame_count == 0:
                intrinsic = load_intrinsic(args)
                
                poses = []

                input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                                        intrinsic, device)
                raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                        depth_ref.columns, intrinsic,
                                                        device)
                raycast_frame.set_data_from_image('depth', depth)
                raycast_frame.set_data_from_image('color', color)        
 
            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color) 

            try:
                if frame_count > 0:
                    
                    result = model.track_frame_to_model(input_frame, raycast_frame,
                                                        args.depth_scale,
                                                        args.depth_max,
                                                        args.odometry_distance_thr)
                    T_frame_to_model = T_frame_to_model @ result.transformation

                poses.append(T_frame_to_model.cpu().numpy())
                model.update_frame_pose(frame_count, T_frame_to_model)
                model.integrate(input_frame, args.depth_scale, args.depth_max,
                                args.trunc_voxel_multiplier)
                model.synthesize_model_frame(raycast_frame, args.depth_scale,
                                            args.depth_min, args.depth_max,
                                            args.trunc_voxel_multiplier, False)
                stop = time.time()
                print('{:04d} slam takes {:.4}s'.format(frame_count, stop - start))    
                
            except:
                print("Unexpected error: ", sys.exc_info()[1])

            frame_count += 1   


            #visulization
            # Render images
            """ depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_array, alpha=0.09), cv2.COLORMAP_JET)
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', depth_colormap)
            key = cv2.waitKey(1) """

            # if 'esc' button pressed, escape loop and exit program
            #if key == 27 or exit_flag:
            if exit_flag:
                #cv2.destroyAllWindows()
                pc = model.extract_pointcloud().transform(flip_transform)
                print('Saving to {} and {}...'.format(args.path_npz, args.path_ply))
                model.voxel_grid.save(args.path_npz)
                o3d.t.io.write_point_cloud(args.path_ply, pc)
                save_poses('output.log', poses)
                print('Saving finished')
                #o3d.visualization.draw([pc])

                if args.upload:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect('162.208.132.238', username='nathans', password='nathans')
                    sftp = ssh.open_sftp()
                    sftp.put(args.path_ply, '/mnt/output.ply')
                    sftp.close()
                    ssh.close()

                break

    finally:
        pipeline.stop()