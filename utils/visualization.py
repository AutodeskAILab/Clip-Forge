import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys 
sys.path.append('../')
import trimesh
import torch
import seaborn as sns


def save_mesh(voxels, out_file=None, threshold=0.5, smooth=True, box_size=1.1, lamb=0.05):
    from utils import libmcubes
    n_x, n_y, n_z = voxels.shape
    box_size = box_size
    
    threshold = np.log(threshold) - np.log(1. - threshold)
    # Make sure that mesh is watertight
   
    #voxels = np.pad(voxels, 1, 'constant', constant_values=-1e6)
        
    
    vertices, triangles = libmcubes.marching_cubes(
        voxels, threshold)
    vertices -= 0.5
    # Undo padding
    #vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)
        
    
    normals = None
    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles,
                            vertex_normals=normals,
                            process=False)
    #print(mesh)
    if smooth == True:
        try:
            mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=lamb)
        except:
            print("Error Smoothing mesh")
        
    if out_file is not None:
        mesh.export(out_file)
    else:
        return mesh


                
def save_point_cloud(voxels, out_file):
    
    result = np.nonzero(voxels)
    pc = list(zip(result[0], result[1],  result[2]))

    file1 = open(out_file,"w")
    file1.write('ply\n')
    file1.write('format ascii 1.0\n')
    file1.write('element vertex '+ str(len(pc)) + ' \n')
    file1.write('property float x\n')
    file1.write('property float y\n')
    file1.write('property float z\n')
    file1.write('end_header\n')
    for index in range(len(pc)):
        file1.write(str(pc[index][1])+' '+str(pc[index][2])+' '+str(pc[index][0])+'\n')
    file1.close()
    

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def visualize_voxels(voxels, out_file=None, show=False, transpose=True):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if transpose == True:
        voxels = voxels.transpose(2, 0, 1)
    #else:
        #voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    
    ax.view_init(elev=30, azim=45)
    
    if out_file is not None:
        plt.axis('off')
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)
    
def sketch_point_cloud(points, save_loc=None, lmit=0.4):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    #plt.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-1 * lmit, lmit])
    ax.set_ylim([-1 * lmit, lmit])
    ax.set_zlim([-1 * lmit, lmit])
    ax.view_init(30, 0)
    if save_loc != None:
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        plt.savefig(save_loc)
    
    plt.show()
    
def visualize_pointcloud(points, save_loc=None, show=False):
    r''' Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
  
 
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if save_loc is not None:
        plt.axis('off')
        plt.savefig(save_loc)
    else:
        plt.show()
    plt.close(fig)
    
    
def plot_real_pred(real_points, pred_points, num_plots, lmit=0.6, save_loc=None):
    fig = plt.figure(figsize=(40,20))
    for i in range(num_plots):
        plt_num = str(i+1) + "21"
        #ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection='3d')
        real_point = real_points[i]
        ax.scatter(real_point[:,2], real_point[:,0], real_point[:,1])
        #ax.view_init(-30, 45)
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        ax.grid(False)
        
        plt_num = str(i+1) + "22"
        #ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection='3d')
        pred_point = pred_points[i]
        ax.scatter( pred_point[:,2], pred_point[:,0], pred_point[:,1])
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        #ax.view_init(-30, 45)
        ax.grid(False)
    #plt.axis('off')
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    
    plt.show()  
    
def multiple_plot_voxel(batch_data_points, save_loc=None, transpose=True):
        
    fig = plt.figure(figsize=(40,10))

    for i in range(len(batch_data_points)):
        plt_num = "1" + str(len(batch_data_points)) +str(i+1)
        ax = fig.add_subplot(plt_num, projection=Axes3D.name)
        data_points = batch_data_points[i]
        #print(data_points.shape)
        if transpose == True:
            data_points = data_points.transpose(2, 0, 1)
        #else:
            #data_points = data_points.transpose(1, 0, 2)
        ax.voxels(data_points, edgecolor='k')
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        if transpose == False:
            ax.view_init(elev=-30, azim=45)
        else:
            ax.view_init(elev=30, azim=45)
    
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()
    
def multiple_plot(batch_data_points, lmit=0.6, save_loc=None):
    
    my_colors = {0:'orange',1:'red',2:'green',3:'blue',4:'grey',5:'gold',6:'violet',7:'pink',8:'navy',9:'black'}

    fig = plt.figure(figsize=(40,10))

    for i in range(len(batch_data_points)):
        plt_num = "1" + str(len(batch_data_points)) +str(i+1)
        #print(plt_num)
        #ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection='3d')
        data_points = batch_data_points[i]
        
        for i, _ in enumerate(data_points):
            ax.scatter(data_points[i,2], data_points[i,0], data_points[i,1],  color = 'black')
            
            ax.set_xlim([-1 * lmit, lmit])
            ax.set_ylim([-1 * lmit, lmit])
            ax.set_zlim([-1 * lmit, lmit])
            #ax.view_init(-30, 45)
            ax.grid(False)

            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        #ax.view_init(-30, 45)
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()    
    