import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

from .network_utils import ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d, CBatchNorm1d_legacy, ResnetBlockConv1d

EPS = 1e-6

####################################################################################################
####################################################################################################

class VoxelEncoderBN(nn.Module):

    def __init__(self, dim=3, c_dim=128, last_feature_transform=None):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)
        
        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(256)
        
        self.last_feature_transform = last_feature_transform
 
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(self.conv0_bn(net)))
        net = self.conv_1(self.actvn(self.conv1_bn(net)))
        net = self.conv_2(self.actvn(self.conv2_bn(net)))
        net = self.conv_3(self.actvn(self.conv3_bn(net)))
        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        x = self.fc(self.actvn(hidden))
        
        if self.last_feature_transform == "add_noise" and self.training is True:
            x = x + 0.1*torch.randn(*x.size()).to(x.device)  

        return x
        
class Occ_Simple_Decoder(nn.Module):
    def __init__(self,  z_dim=128, point_dim=3, 
                 hidden_size=128, leaky=False, last_sig=False):
        super().__init__()
        self.z_dim = z_dim

        # Submodules
        self.fc_p = nn.Linear(point_dim, hidden_size)

        self.fc_z = nn.Linear(z_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            
        self.last_sig = last_sig
        
    
    def forward(self, p, z):

        batch_size, T, D = p.size()
        
        net = self.fc_p(p)

        net_z = self.fc_z(z).unsqueeze(1)
        net = net + net_z

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        
        if self.last_sig == True:
            out = torch.sigmoid(out)
            
        return out           
####################################################################################################
####################################################################################################

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        
        ### Local Model Hyperparameters
        self.args = args
        self.encoder_type = args.encoder_type 
        self.decoder_type = args.decoder_type
        self.emb_dims = args.emb_dims
        
        ### Sub-Network def 
        self.encoder = VoxelEncoderBN(dim=3, c_dim=args.emb_dims, last_feature_transform="add_noise")
        self.decoder = Occ_Simple_Decoder(z_dim=args.emb_dims)
        
      
                
    def decoding(self, shape_embedding, points=None): 
        return self.decoder(points, shape_embedding)
            

    def reconstruction_loss(self, pred, real_occ):
        loss = torch.mean((pred.squeeze(-1) - real_occ)**2)             
        return loss 
    

    def forward(self, data_input, query_points=None):
        shape_embs = self.encoder(data_input)  
        pred = self.decoding(shape_embs, points=query_points)
        return pred, shape_embs