import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# DATA_DIR = "/disk1/panzhiyu/sync_cam4_dense/"
BONE_LINK = np.array([[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]])

def innerp_mat(a, b=None):
    if b is None: b = a
    assert a.shape == b.shape
    return np.sum(a * b, axis=-1)
    
def calculate_labels(points, joints):
    # Calculate the distance between each point to each bone
    # and assign the label to the point with the minimum distance
    N, M = points.shape[0], len(BONE_LINK)
    distpb = np.zeros((N, M))
    a, b = joints[BONE_LINK[:, 0]], joints[BONE_LINK[:, 1]]
    a = np.repeat(a[None, :], N, axis=0)
    b = np.repeat(b[None, :], N, axis=0)
    points = np.repeat(points[:, None, :], M, axis=1)
    ap, bp, ab = points - a, points - b, b - a
    t = innerp_mat(points - a, b - a) / innerp_mat(b - a, b - a)
    idx1 = np.where(t < 0)
    distpb[idx1] = innerp_mat(ap)[idx1]
    idx2 = np.where(t > 1)
    distpb[idx2] = innerp_mat(bp)[idx2]
    idx3 = np.where((t>=0) & (t<=1))
    distpb[idx3] = innerp_mat(ap - t[:, :, None] * ab)[idx3]
    labels = np.argmin(distpb, axis=1)
    return labels

def calculate_flow(pc1, ind1, mesh2):
    flow12 = mesh2[ind1] - pc1
    return flow12
    

class SyncDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_len = []
        for i in range(10):
            self.data_len.append(self._get_data_length(i))

    def load_mesh_points(self, ped_id, frame_id):
        filepath = os.path.join(self.data_dir, "ped_mesh", str(ped_id), f"{frame_id}.txt")
        mesh = np.loadtxt(filepath)
        mesh = mesh[:, [0, 2, 1]]
        return mesh

    def load_joints(self, ped_id, frame_id):
        filepath = os.path.join(self.data_dir, "joints", str(ped_id), f"joints_{frame_id}.txt")
        joints = np.loadtxt(filepath)
        joints = joints[:, [0, 2, 1]]
        return joints

    def load_points(self, ped_id, frame_id):
        filepath = os.path.join(self.data_dir, "points_ped", str(ped_id), f"{frame_id:05d}.ply")
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        return points
    
    def load_pcd(self, ped_id, frame_id):
        filepath = os.path.join(self.data_dir, "points_ped", str(ped_id), f"{frame_id:05d}.ply")
        return o3d.io.read_point_cloud(filepath)

    def _get_data_length(self, ped_id):
        filepath = os.path.join(self.data_dir, "points_ped", str(ped_id))
        return len(os.listdir(filepath))

    def get_data_length(self, ped_id):
        return self.data_len[ped_id]

