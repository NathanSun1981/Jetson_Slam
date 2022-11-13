"""ICP (Iterative Closest Point) registration algorithm"""
import copy
import os

import numpy as np
import open3d as o3d

TEMP_PATH = os.getcwd() + '/temp/'

class registeration_ransac:

    def visualize(pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.light_on = False
        opt.point_size = 2
        vis.run()
        vis.destroy_window()

    def preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                max_nn=30))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                                max_nn=100))
        return (pcd_down, pcd_fpfh)

    def execute_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        """ print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold) """

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                mutual_filter = False,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.
                TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    4000000, 0.999))
        return result


    def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, initial_guess):
        distance_threshold = voxel_size * 0.4
        """ print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold) """
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result



    def mapping(pcd_list, voxel_size):

        for frame_idx in range(len(pcd_list) - 1):
            # read data
            #print('Downsampling inputs')
            if frame_idx == 0:
                print("dst", pcd_list[frame_idx])
                map_pcd = o3d.io.read_point_cloud(pcd_list[frame_idx])
                
                map_down, map_fpfh = registeration_ransac.preprocess_point_cloud(map_pcd, voxel_size)
            print("src", pcd_list[frame_idx + 1])
            src_pcd = o3d.io.read_point_cloud(pcd_list[frame_idx + 1])
            src_down, src_fpfh = registeration_ransac.preprocess_point_cloud(src_pcd, voxel_size)

            #print('Running RANSAC')
            # global ICP 
            result_ransac = registeration_ransac.execute_global_registration(src_down, map_down,
                                                        src_fpfh, map_fpfh,
                                                        voxel_size)
            #print(result_ransac)
            # update map
            map_pcd += src_pcd.transform(result_ransac.transformation)
            #map_down, map_fpfh = preprocess_point_cloud(map_pcd, voxel_size)

        return map_pcd


    def pcd_registeration_ransac(path, voxel_size):      
        #voxel_size = 0.3  # means 5cm for the dataset
        data_list = os.listdir(path)
        print(data_list)
        data_list.sort(key=lambda x:int(x.split('.')[0]))
        path = TEMP_PATH

        print(data_list)

        pcds = [o3d.io.read_point_cloud(TEMP_PATH + path) for path in data_list]
        pcd_list = [TEMP_PATH + path for path in data_list]
        print(pcd_list) 
        map_pcd = registeration_ransac.mapping(pcd_list=pcd_list, voxel_size=voxel_size)


        #registeration_ransac.visualize(map_pcd)
        o3d.visualization.draw_geometries([map_pcd])

        o3d.io.write_point_cloud(path + "mapping.pcd", map_pcd)
