import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm


class VoxelEncoderBN(nn.Module):
    def __init__(self, dim=3, c_dim=128):
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
        return x
    


class classifier_32(nn.Module):
    def __init__(self, encoder_type, num_classes, dropout=0.5):
        super(classifier_32, self).__init__()

        self.encoder_head = VoxelEncoderBN(c_dim=512)
                    
        self.projection = nn.Sequential(
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(512, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(512, num_classes)
                          )
            
        
    def forward(self, x):
        z = self.encoder_head(x)
        x = self.projection(z)
        return x, z
    
def get_activations(datapoints, model, args):
    model.eval()
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(datapoints).squeeze())
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    all_activation = []
    all_labels = []
    softmax = nn.Softmax(dim=-1).to(args.device)
    
    with torch.no_grad():
        for data in tqdm(loader):
            try:
                data_mod = data[0].type(torch.FloatTensor).to(args.device)  
                out, embeddings = model(data_mod)
                pred_label = softmax(out)
                _, pred_label = torch.max(pred_label, dim=-1)
            except:
                print("Some Error happened")
                print(data[0])
                raise "err"
                continue 
            all_activation.append(embeddings.detach().cpu().numpy())
            all_labels.append(pred_label.detach().cpu().numpy())
    
    all_activation = np.concatenate(all_activation)
    all_labels = np.concatenate(all_labels)
    return all_activation, all_labels  

