#!/usr/bin/python
import asyncore
import getopt
import json
import pickle
import socket
import struct
import sys
from enum import IntEnum

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from pyparsing import And


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
mc_ip_address = '224.0.0.1'
port = 1024
remote_port = 1025
chunk_size = 4096
#rs.log_to_console(rs.log_severity.debug)

flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)

 
    return out

def getDataStructure(pipeline, depth_filter, depth_scale, clipping_distance_in_meters):
     # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    
    frames = pipeline.wait_for_frames()
    # take owner ship of the frame for further processing
    frames.keep()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)


    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        return None, None

    #post_process can be done on client side
    #depth2 = depth_filter.process(aligned_depth_frame)
    aligned_depth_frame.keep()
    # represent the frame as a numpy array
    depthData = aligned_depth_frame.get_data()
    colorData = color_frame.get_data()
    depthMat = np.asanyarray(depthData)
    colorMat = np.asanyarray(colorData)
    ts = frames.get_timestamp()


    #get point cloud using open3d
    depth_image = o3d.geometry.Image(
        np.array(aligned_depth_frame.get_data()))
    color_image = o3d.geometry.Image(np.asarray(color_frame.get_data()))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(color_frame))

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

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0 / depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity=False)
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    temp.transform(flip_transform)
    points = np.asarray(temp.points).astype(np.float32)
    colors = np.asarray(temp.colors).astype(np.float32)
    pc = np.concatenate([points, colors], axis=-1)

    return depthMat, colorMat, pc, intrinsic_json, ts


def openPipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    pipeline = rs.pipeline()
    pipeline_profile = pipeline.start(cfg)
    sensor = pipeline_profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    depth_scale = sensor.get_depth_scale()
    clipping_distance_in_meters = 3# 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    #print(depth_scale)

    return pipeline, depth_scale, clipping_distance

class DevNullHandler(asyncore.dispatcher_with_send):

    def handle_read(self):
        print(self.recv(1024))

    def handle_close(self):
        self.close()
           
		
class EtherSenseServer(asyncore.dispatcher):
    def __init__(self, address):
        asyncore.dispatcher.__init__(self)
        print("Launching Realsense Camera Server")
        try:
            self.pipeline, self.depth_scale, self.clipping_distance = openPipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        #print('sending acknowledgement to', address)
        
	# reduce the resolution of the depth image using post processing
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        self.frame_data = ''
        print(address)
        self.connect(address)
        self.packet_id = 0      

    def handle_connect(self):
        print("connection received")

    def writable(self):
        return True

    def update_frame(self):
        depth, color, pc, intrinsic_json, timestamp = getDataStructure(self.pipeline, self.decimate_filter, self.depth_scale, self.clipping_distance)
        if depth is not None and color is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)
            img = np.hstack((color, depth_colormap))
            #print(color.shape, depth_colormap.shape)
	    # convert the depth image to a string for broadcast


            data = pickle.dumps(color)
            depth_data = pickle.dumps(depth)

	    # capture the lenght of the data portion of the message	
            length = struct.pack('<I', len(data))
            depth_length = struct.pack('<I', len(depth_data))

            #print(len(intrinsic_json.encode('utf-8')))
            len_instrinsic = struct.pack('<I', len(intrinsic_json.encode('utf-8')))

            instrinsic = intrinsic_json.encode('utf-8')

	    # include the current timestamp for the frame
            ts = struct.pack('<d', timestamp)
            
	    # for the message for transmission
        if self.packet_id == 0:
            print('send the very first frame with instrinsic')
            self.frame_data = b''.join([ts, length, depth_length, len_instrinsic, instrinsic, data, depth_data])
        else:
            self.frame_data = b''.join([ts, length, depth_length, data, depth_data])

        self.packet_id += 1  
            

    def handle_write(self):
	# first time the handle_write is called
        print("handle_write")
        if not hasattr(self, 'frame_data'):
            self.update_frame()
	# the frame has been sent in it entirety so get the latest frame
        if len(self.frame_data) == 0:
            self.update_frame()
        else:
	    # send the remainder of the frame_data until there is no data remaining for transmition
            remaining_size = self.send(self.frame_data)
            self.frame_data = self.frame_data[remaining_size:]
	

    def handle_close(self):
        self.close()
            

class MulticastServer(asyncore.dispatcher):
    def __init__(self, host = mc_ip_address, port=1024):
        asyncore.dispatcher.__init__(self)
        server_address = ('', port)
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind(server_address) 	

    def handle_read(self):
        data, addr = self.socket.recvfrom(42)
        print('Recived Multicast message %s bytes from %s' % (data, addr))
	# Once the server recives the multicast signal, open the frame server
        EtherSenseServer(addr)
        #self.socket.close()
        print(sys.stderr, data)

    def writable(self): 
        return False # don't want write notifies

    def handle_close(self):
        self.close()

    def handle_accept(self):
        channel, addr = self.accept()
        print('received %s bytes from %s' % (channel, addr))


def main(argv):
    # initalise the multicast receiver 
 
    #server = MulticastServer()
    #addr = ('162.208.132.238', remote_port)
    MulticastServer()
    # hand over excicution flow to asyncore
    asyncore.loop()
   
if __name__ == '__main__':
    main(sys.argv[1:])

