#!/usr/bin/python
import argparse
import asyncore
import getopt
import json
import os
import pickle
import socket
import struct
import sys
import time
from calendar import TUESDAY
from re import I

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from common import (extract_rgbd_frames, extract_trianglemesh,
                    get_default_dataset, load_intrinsic, load_rgbd_file_names,
                    save_poses)
from config import ConfigParser
from icp_registration import registeration_ransac

#from ..real_sense.common import (extract_trianglemesh, load_intrinsic, save_poses)
#from ..real_sense.config import ConfigParser


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
mc_ip_address = '224.0.0.1'
port = 1024
local_port = 1025
chunk_size = 4096

#UDP client for each camera server 
class SlamClient(asyncore.dispatcher):

    def __init__(self, server, source, config):   
        asyncore.dispatcher.__init__(self, server)
        self.instrinsic = ''
        self.address = server.getsockname()[0]
        self.port = source[1]
        self.buffer = bytearray()
        self.windowName = self.port
        # open cv window which is unique to the port 
        self.remainingBytes = 0
        self.frame_id = 0
        self.pc_flag = 0
        self.pcd = o3d.geometry.PointCloud()
        self.config = config
        self.config.path_intrinsic = 'instrinsic.json'

        self.device = o3d.core.Device(self.config.device)
        self.T_frame_to_model = o3d.core.Tensor(np.identity(4))
        self.model = o3d.t.pipelines.slam.Model(self.config.voxel_size, 16,
                                            self.config.block_count, self.T_frame_to_model,
                                            self.device)

        cv2.namedWindow("window_RGB"+str(self.windowName))
        #cv2.namedWindow("window_Gray"+str(self.windowName))

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        #GLFW_KEY https://www.glfw.org/docs/latest/group__keys.html#ga4d7f0260c82e4ea3d6ebc7a21d6e3716
        self.geometry_added = False

        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
       
    def handle_read(self):
        if self.remainingBytes == 0:
            # get the expected frame siz

            #self.pc_flag = struct.unpack('i', self.recv(4))[0]

            self.timestamp = struct.unpack('<d', self.recv(8))
            self.frame_length = struct.unpack('<I', self.recv(4))[0]
            self.depth_length = struct.unpack('<I', self.recv(4))[0]
            if self.frame_id == 0:
                self.instrinsic_length = struct.unpack('<I', self.recv(4))[0]

                #print(self.instrinsic_length)
                self.instrinsic = self.recv(self.instrinsic_length).decode("utf-8")

                with open(self.config.path_intrinsic, "w") as outfile:
                    json_obj = json.dump(
                        json.loads(self.instrinsic),
                        outfile,
                        indent = 4)
                # get the timestamp of the current frame   
                # 
            self.remainingBytes = self.frame_length + self.depth_length
        
        # request the frame data until the frame is completely in buffer
        data = self.recv(self.remainingBytes)
        self.buffer += data
        self.remainingBytes -= len(data)
        # once the frame is fully recived, process/display it
        if len(self.buffer) == self.frame_length + self.depth_length:
            self.handle_frame()

    def handle_frame(self):
        # convert the frame from string to numerical data
        #print(self.buffer)
        imdata = pickle.loads(self.buffer[:self.frame_length])
        graydata = pickle.loads(self.buffer[self.frame_length:])

        if self.pc_flag == 1:
            cv2.destroyAllWindows()
            self.pcd.points = o3d.utility.Vector3dVector(imdata[:, 0:3])
            self.pcd.colors = o3d.utility.Vector3dVector(imdata[:, 3:6])
            if self.frame_id == 0:
                self.vis.add_geometry(self.pcd)

            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            
        else:
            bigDepth = cv2.resize(imdata, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST) 
            cv2.putText(bigDepth, str(self.timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (65536), 2, cv2.LINE_AA)
            print(imdata.shape)


            #start slam on every fram
            color_ref = o3d.t.geometry.Image(imdata.astype(np.uint8))
            depth_ref = o3d.t.geometry.Image(graydata.astype(np.uint16))
            depth = depth_ref.to(self.device)
            color = color_ref.to(self.device) 

            color_img = o3d.geometry.Image(imdata.astype(np.uint8))
            depth_img = o3d.geometry.Image(graydata.astype(np.uint16))

            start = time.time()
            
            if self.frame_id == 0:
                self.intrinsic = load_intrinsic(self.config)
                self.poses = []

                self.input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                                        self.intrinsic, self.device)
                self.raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                        depth_ref.columns, self.intrinsic,
                                                        self.device)
                self.raycast_frame.set_data_from_image('depth', depth)
                self.raycast_frame.set_data_from_image('color', color)     
 
            self.input_frame.set_data_from_image('depth', depth)
            self.input_frame.set_data_from_image('color', color)  

            try:
                if self.frame_id > 0:
                    
                    result = self.model.track_frame_to_model(self.input_frame, self.raycast_frame,
                                                        self.config.depth_scale,
                                                        self.config.depth_max,
                                                        self.config.odometry_distance_thr)
                    self.T_frame_to_model = self.T_frame_to_model @ result.transformation

                self.poses.append(self.T_frame_to_model.cpu().numpy())
                self.model.update_frame_pose(self.frame_id, self.T_frame_to_model)
                self.model.integrate(self.input_frame, self.config.depth_scale, self.config.depth_max,
                                self.config.trunc_voxel_multiplier)
                self.model.synthesize_model_frame(self.raycast_frame, self.config.depth_scale,
                                            self.config.depth_min, self.config.depth_max,
                                            self.config.trunc_voxel_multiplier, False)
                stop = time.time()
                print('{:04d} slam takes {:.4}s'.format(self.frame_id, stop - start))    


                pc = self.model.extract_pointcloud().transform(self.flip_transform)
                pcd = pc.to_legacy()


                """ if pcd.has_points():
                    self.vis.add_geometry(pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer() """

          
            except:
                print("Unexpected error: ", sys.exc_info()[1])
                
            #clipping_distance = self.config.clipping_distance_in_meters / self.config.depth_scale
            #intrinsic = o3d.io.read_pinhole_camera_intrinsic(self.config.path_intrinsic)

            
        
            cv2.imshow("window_RGB"+str(self.windowName), imdata)

            key = cv2.waitKey(1)
            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                #process model
                if os.path.exists(self.config.path_npz):
                    os.remove(self.config.path_npz)

                pc = self.model.extract_pointcloud().transform(self.flip_transform)
                print('Saving to {} and {}...'.format(self.config.path_npz, self.config.path_ply) )
                self.model.voxel_grid.save(self.config.path_npz)
                o3d.io.write_point_cloud(self.config.path_ply, pc.to_legacy())
                save_poses('output.log', self.poses)
                print('Saving finished')
                o3d.visualization.draw(pc.to_legacy())
                self.vis.destroy_window()
                
                exit()
            


        self.buffer = bytearray()
        self.frame_id += 1

    def readable(self):
        return True

    
class EtherSenseClient(asyncore.dispatcher):
    def __init__(self, config):
        asyncore.dispatcher.__init__(self)
        #self.server_address = ('', port)
        self.client_address = ('', local_port)
        # create a socket for TCP connection between the client and server
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)       
        self.bind(self.client_address) 	
        self.listen(10)
        self.config = config

    def writable(self): 
        return False # don't want write notifies

    def readable(self):
        return True
        
    def handle_connect(self):
        print("connection recvied")

    def handle_accept(self):
        pair = self.accept()
        #print(self.recv(10))
        if pair is not None:
            sock, addr = pair
            print ('Incoming connection from %s' % repr(addr))
            # when a connection is attempted, delegate image receival to the SlamClient 
            handler = SlamClient(sock, addr, self.config)

def multi_cast_message(ip_address, port, message, config):
    # send the multicast message
    multicast_group = (ip_address, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_addr = ('', local_port)
    sock.bind(client_addr)
    try:
        # Send data to the multicast group
        #print('sending "%s"' % message + str(multicast_group))
        sent = sock.sendto(message.encode(), multicast_group)
   
        # defer waiting for a response using Asyncore
        client = EtherSenseClient(config)
        asyncore.loop()

        # Look for responses from all recipients
        
    except socket.timeout:
        print('timed out, no more responses')
    finally:
        print(sys.stderr, 'closing socket')
        sock.close()

if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--path_npz',
            help='path to the npz file that stores voxel block grid.',
            default='output.npz')
    parser.add('--path_ply',
            help='path to the npz file that stores voxel block grid.',
            default='output.ply')    
    config = parser.get_config()
    #print(config)


    multi_cast_message(mc_ip_address, port, 'EtherSensePing', config)

    """ client = EtherSenseClient(config)
    asyncore.loop() """
