import os
import os.path as osp
import logging
import argparse
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from utils import helper
from utils import visualization

from networks import autoencoder

from dataset import shapenet_dataset
###################################### Experiment Utils########################################################


def experiment_name(args):

    from datetime import datetime

    tokens = ["Autoencoder", args.dataset_name, args.input_type, args.output_type, args.emb_dims, args.last_feature_transform]
           
    if args.categories != None:
        for i in args.categories:
            tokens.append(i)
        
    if args.num_sdf_points != 5000:
        tokens.append(args.num_sdf_points)
              
    tokens.append(args.seed)
    return "_".join(map(str, tokens))

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        
def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

###################################### Experiment Utils########################################################

############################################# data loader #################################################

def get_dataloader(args, split="train"):
    
    if args.dataset_name == "Shapenet":
        pointcloud_field = shapenet_dataset.PointCloudField("pointcloud.npz")
        points_field = shapenet_dataset.PointsField("points.npz",unpackbits=True)
        voxel_fields = shapenet_dataset.VoxelsField("model.binvox")

        fields = {}

        fields['pointcloud'] = pointcloud_field
        fields['points'] = points_field
        fields['voxels'] = voxel_fields

        if split == "train":
            dataset = shapenet_dataset.Shapes3dDataset(args.dataset_path, fields, split=split,
                     categories=args.categories, no_except=True, transform=None, num_points=args.num_points,           num_sdf_points=args.num_sdf_points, sampling_type=args.sampling_type)

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
            total_shapes = len(dataset)
        else:
            dataset = shapenet_dataset.Shapes3dDataset(args.dataset_path, fields, split=split,
                     categories=args.categories, no_except=True, transform=None, num_points=args.num_points, num_sdf_points=args.test_num_sdf_points,  sampling_type=args.sampling_type)
            dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
            total_shapes = len(dataset)
        return dataloader, total_shapes 
  
    
    else:
        raise ValueError("Dataset name is not defined {}".format(args.dataset_name))

############################################# data loader #################################################


############################## visualization #################################################

def visualization_model(model, args, test_dataloader, name_info):
    model.eval()
    test_loader = iter(test_dataloader)
    data = next(test_loader)
    
    
    if args.input_type == "Voxel":
        data_input = data['voxels'].type(torch.FloatTensor).to(args.device)
    elif args.input_type == "Pointcloud":
        data_input = data['pc_org'].type(torch.FloatTensor).to(args.device).transpose(-1, 1)

    if args.output_type == "Implicit":
        voxel_32 = data['voxels'].type(torch.FloatTensor).to(args.device)
        voxel_size = 32
        shape = (voxel_size, voxel_size, voxel_size)
        p = 1.1 * visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(args.test_batch_size, *p.size())
    elif args.output_type == "Pointcloud":
        query_points = None
        gt = data['pc_org'].type(torch.FloatTensor).to(args.device) 
                
   
    with torch.no_grad():
        pred, decoder_embs = model(data_input, query_points)
        
        if name_info is not None:
            save_loc = args.vis_dir + "/" + str(name_info) + "_"  
        else:
            save_loc = args.vis_dir + "/"
        
        if args.output_type == "Implicit":
            voxels_out = (pred[0].view(voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            real = voxel_32[0].detach().cpu().numpy()
            visualization.multiple_plot_voxel([real, voxels_out], save_loc=save_loc + "real_pred.png")
            #visualization.save_mesh(voxels_out, out_file=save_loc + "pred.obj")
        elif args.output_type == "Pointcloud":
            visualization.plot_real_pred(gt.detach().cpu().numpy(), pred.detach().cpu().numpy(), 1, save_loc=save_loc + "real_pred.png") 
        
############################## visualization #################################################
       
############################## validation #################################################

def val_one_epoch_iou(model, args, test_dataloader, epoch):
    model.eval()
    loss_reconstruction = []
    points_voxels = visualization.make_3d_grid(
        (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3).type(torch.FloatTensor).to(args.device)
    query_points = points_voxels.expand(args.test_batch_size, *points_voxels.size())
    
    with torch.no_grad():
        for data in test_dataloader:
            
            data_input = data['voxels'].type(torch.FloatTensor).to(args.device)
            
            voxels_occ_np =  (data['voxels'] >= 0.5).cpu().numpy()    
            
            if args.test_batch_size != data_input.size(0):
                query_points = points_voxels.expand( data_input.size(0), *points_voxels.size())
            
            pred, _ = model(data_input, query_points)
            
            occ_hat_np = (pred >= args.threshold).cpu().numpy()
                
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            loss_reconstruction.append(iou_voxels.item())
        
    loss_reconstruction = np.asarray(loss_reconstruction)
    loss_reconstruction = np.mean(loss_reconstruction)
    logging.info("[Val]  Epoch {} IOU Loss: {}".format(epoch, loss_reconstruction))
    return loss_reconstruction  

def val_one_epoch(model, args, test_dataloader, epoch):
    model.eval()
    loss_reconstruction = []

    with torch.no_grad():
        for data in test_dataloader:
            
            if args.input_type == "Voxel":
                data_input = data['voxels'].type(torch.FloatTensor).to(args.device)
            elif args.input_type == "Pointcloud":
                data_input = data['pc_org'].type(torch.FloatTensor).to(args.device).transpose(-1, 1)

            if args.output_type == "Implicit":
                query_points, occ = data['points'], data['points.occ']
                query_points = query_points.type(torch.FloatTensor).to(args.device)
                occ = occ.type(torch.FloatTensor).to(args.device)
                gt = occ 
            elif args.output_type == "Pointcloud":
                query_points = None
                gt = data['pc_org'].type(torch.FloatTensor).to(args.device) 
            
            pred, _ = model(data_input, query_points)
            loss_reconstuct = model.reconstruction_loss(pred, gt)

            loss_reconstruction.append(loss_reconstuct.item())
        
    loss_reconstruction = np.asarray(loss_reconstruction)
    loss_reconstruction = np.mean(loss_reconstruction)
    logging.info("[Val]  Epoch {} Loss: {}".format(epoch, loss_reconstruction))
    return loss_reconstruction  

############################## validation #################################################

############################## training #################################################

def train_one_epoch(model, args, train_dataloader, optimizer, scheduler, loss_meter, epoch):
    model.train()
    loss_reconstruction = []    
    iteration = 0
    for data in train_dataloader:
        iteration = iteration + 1
        optimizer.zero_grad()

        data_index =  data['idx'].to(args.device)
        
        if args.input_type == "Voxel":
            data_input = data['voxels'].type(torch.FloatTensor).to(args.device)
        elif args.input_type == "Pointcloud":
            data_input = data['pc_org'].type(torch.FloatTensor).to(args.device).transpose(-1, 1)
           
        if args.output_type == "Implicit":
            query_points, occ = data['points'], data['points.occ']
            query_points = query_points.type(torch.FloatTensor).to(args.device)
            occ = occ.type(torch.FloatTensor).to(args.device)
            gt = occ 
        elif args.output_type == "Pointcloud":
            query_points = None
            gt = data['pc_org'].type(torch.FloatTensor).to(args.device) 
        
        pred, shape_embs = model(data_input, query_points)

        loss_reconstuct = model.reconstruction_loss(pred, gt)
                   
        loss = loss_reconstuct 
        loss.backward()
        optimizer.step()
        loss_meter.update(loss, data_input.size(0))
        
        loss_reconstruction.append(loss_reconstuct.item())
           
        if iteration % args.print_every == 0:
            avg_reconstruction_loss = np.mean(np.asarray(loss_reconstruction))
          
            logging.info("[Train]  Epoch {}, Iteration {} loss: {}, recon loss: {}".format(epoch, iteration, loss_meter.avg, avg_reconstruction_loss))

############################## training #################################################   

def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    
    ### Sub Network details
    parser.add_argument("--input_type", type=str, default='Voxel', help='What is the input representation')
    parser.add_argument("--output_type", type=str, default='Implicit', help='What is the output representation')
    parser.add_argument("--encoder_type", type=str, default='Voxel_Encoder_BN', help='what is the encoder')
    parser.add_argument("--decoder_type", type=str, default='Occ_Simple_Decoder', help='what is the decoder')
    parser.add_argument('--emb_dims', type=int, default=128, help='Dimension of embedding')
    parser.add_argument('--last_feature_transform', type=str, default="add_noise", help='add_noise or none')
    parser.add_argument('--reconstruct_loss_type', type=str, default="sum", help='bce or sum (mse) or mean (mse)')
    parser.add_argument('--pc_dims', type=int, default=1024, help='Dimension of embedding')
                        
    ### Dataset details
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--dataset_name', type=str, default="Shapenet", help='Dataset path')
    parser.add_argument("--num_points", type=int, default=2025, help='Number of points')
    parser.add_argument("--num_sdf_points", type=int, default=5000, help='Number of points')
    parser.add_argument("--test_num_sdf_points", type=int, default=30000, help='Number of points')
    parser.add_argument('--categories',   nargs='+', default=None, metavar='N')
    parser.add_argument("--num_workers", type=int, default=4, help='Number of workers')                     
    
    ### training details
    parser.add_argument('--train_mode', type=str, default="train", help='train or test')
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    parser.add_argument('--epochs', type=int, default=300, help="Total epochs")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to load")
    parser.add_argument('--use_timestamp',  action='store_true', help='Whether to use timestamp in dump files')
    parser.add_argument('--num_iterations', type=int, default=300000, help='How long the training shoulf go on')    
    parser.add_argument('--gpu', nargs='+' , default="0", help='GPU list')
    parser.add_argument('--optimizer', type=str, choices=('SGD', 'Adam'), default='Adam')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=32, help='Dimension of embedding')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Dimension of embedding')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for voxel stuff')
    parser.add_argument('--sampling_type', type=str, default=None, help='what sampling type: None--> Uniform')
    
    ### Logging details 
    parser.add_argument('--print_every', type=int, default=50, help='Printing the loss every')
    parser.add_argument('--save_every', type=int, default=50, help='Saving the model every')
    parser.add_argument('--validation_every', type=int, default=5000, help='validation set every')
    parser.add_argument('--visualization_every', type=int, default=10, help='visualization of the results every')
    parser.add_argument("--log-level", type=str, choices=('info', 'warn', 'error'), default='info')
    parser.add_argument('--experiment_type', type=str, default="max", help='experiment type')
    parser.add_argument('--experiment_every', type=int, default=5, help='experiment every ')
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser


def main():
    args = parsing()
    exp_name = experiment_name(args)
    
    manualSeed = args.seed
    helper.set_seed(manualSeed)

    # Create directories for checkpoints and logging
    args.experiment_dir = osp.join('exps', exp_name)
    args.checkpoint_dir = osp.join('exps', exp_name, 'checkpoints')
    args.vis_dir =  osp.join('exps', exp_name, 'vis_dir') + "/"
    args.generate_dir =  osp.join('exps', exp_name, 'generate_dir') + "/"                          
    
    
    if args.train_mode != "test":
        log_filename = osp.join('exps', exp_name, 'log.txt')
        helper.create_dir(args.experiment_dir)
        helper.create_dir(args.checkpoint_dir)
        helper.create_dir(args.vis_dir)
        helper.create_dir(args.generate_dir)
        helper.setup_logging(log_filename, args.log_level, 'w')
    else:
        test_log_filename = osp.join('exps', exp_name, 'test_log.txt')
        helper.setup_logging(test_log_filename, args.log_level, 'w')
        args.examplar_generate_dir =  osp.join('exps', exp_name, 'exam_generate_dir') + "/"                          
        helper.create_dir(args.examplar_generate_dir)
        args.vis_gen_dir =  osp.join('exps', exp_name, 'vis_gen_dir') + "/"                          
        helper.create_dir(args.vis_gen_dir)
        
        
        
    logging.info("Experiment name: {}".format(exp_name))
    logging.info("{}".format(args))

    device, gpu_array = helper.get_device(args)
    args.device = device 
    
    logging.info("#############################")
    train_dataloader, total_shapes  = get_dataloader(args, split="train")
    args.total_shapes = total_shapes
    logging.info("Train Dataset size: {}".format(total_shapes))
    test_dataloader, total_shapes_test  = get_dataloader(args, split="val")
    logging.info("Test Dataset size: {}".format(total_shapes_test))
    logging.info("#############################")
    
    #####
    net = autoencoder.get_model(args).to(args.device)
    print(net)
    logging.info("#############################")
    
    if args.train_mode == "test":
        print("Test mode ")
        print("Loading model....", args.checkpoint)
        checkpoint = torch.load(args.checkpoint_dir +"/"+ args.checkpoint +".pt")
        net.load_state_dict(checkpoint['model'])

        full_test_dataloader, total_shapes_test  = get_dataloader(args, split="test")
        logging.info("Test Dataset size: {}".format(total_shapes_test))
        
        if args.output_type == "Implicit":
            test_iou = val_one_epoch_iou(net, args, full_test_dataloader, 0)
            logging.info("Test iou {}".format(test_iou))
        elif  args.output_type == "Pointcloud":
            test_val = val_one_epoch(net, args, full_test_dataloader, 0)
            logging.info("Test val loss {}".format(test_val))
        
    else:
        optimizer = helper.get_optimizer_model(args.optimizer, net, lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.num_iterations, 0.000001)

        start_epoch = 0 
        if args.checkpoint != None:
            print("Loading model....", args.checkpoint)
            checkpoint = torch.load(args.checkpoint_dir +"/"+ args.checkpoint +".pt")
            net.load_state_dict(checkpoint['model'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['current_epoch']
            

        best_loss = 100000    
        best_iou = 0
        best_32_loss = 100000

        for epoch in range(start_epoch, args.epochs):
            loss_meter = helper.AverageMeter()
            logging.info("#############################")
            #val_iou = val_one_epoch_iou(net, args, test_dataloader, epoch)
            train_one_epoch(net, args, train_dataloader, optimizer, scheduler, loss_meter, epoch)
    
            if (epoch + 1) % 5 == True:
                visualization_model(net, args, test_dataloader, epoch)
            
            if args.output_type == "Implicit":
                val_iou = val_one_epoch_iou(net, args, test_dataloader, epoch)
                if  best_iou < val_iou:
                    best_iou = val_iou
                    filename = '{}.pt'.format("best_iou")
                    logging.info("Saving Model........{}".format(filename))
                    helper.save_checkpoint(osp.join(args.checkpoint_dir, filename), net, args, optimizer, scheduler, epoch)
            elif args.output_type == "Pointcloud":
                val_loss = val_one_epoch(net, args, test_dataloader, epoch)
                if  best_loss > val_loss:
                    best_loss = val_loss
                    filename = '{}.pt'.format("best")
                    logging.info("Saving Model........{}".format(filename))
                    helper.save_checkpoint(osp.join(args.checkpoint_dir, filename), net, args, optimizer, scheduler, epoch)

            filename = '{}.pt'.format("last")
            logging.info("Saving Model........{}".format(filename))
            helper.save_checkpoint(osp.join(args.checkpoint_dir, filename), net, args, optimizer, scheduler, epoch)


if __name__ == "__main__":
    main()   
    