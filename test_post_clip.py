import os
import os.path as osp
import logging

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data import Dataset, DataLoader


from utils import helper
from utils import visualization
from utils import experimenter

from train_autoencoder import experiment_name, parsing
from train_post_clip import get_dataloader, experiment_name2, get_condition_embeddings, get_local_parser, get_clip_model
from dataset import shapenet_dataset
from networks import autoencoder, latent_flows

import clip

###################################### Text Queries ###############################################

id_to_label = {'02691156': 0, '02828884': 1, '02933112': 2, '02958343': 3, '03001627': 4, '03211117': 5, '03636649': 6, '03691459': 7, '04090263': 8, '04256520': 9, '04379243': 10, '04401088': 11, '04530566': 12}

label_to_category = {0: 'airplane', 1:'bench', 2:'cabinet', 3:'car',  4:'chair', 5:'monitor', 6:'lamp',  7:'loudspeaker',  8:'gun',  9:'sofa',  10:'table',  11:'phone',  12:'boat'}


id_to_sub_category = {

"02691156": ["airplane", "jet", "fighter plane", "biplane", "seaplane", "space shuttle", "supersonic plane", "rocket plane", "delta wing", "swept wing plane" , "straight wing plane", "propeller plane"],     

"02828884": ["bench", "pew", "flat bench", "settle", "back bench", "laboratory bench", "storage bench"],    

"02933112": ["cabinet", "garage cabinet", "desk cabinet"]  ,
    
"02958343": ["car", "bus", "shuttle-bus", "pickup car", "truck", "suv", "sports car", "limo", "jeep", "van", "gas guzzler", "race car", "monster truck", "armored", "atv", "microbus", "muscle car", "retro car", "wagon car", "hatchback", "sedan", "ambulance", "roadster car", "beach wagon"],

"03001627": ["chair", "arm chair",  "bowl chair", "rocking chair", "egg chair", "swivel chair", "bar stool", "ladder back chair", "throne", "office chair", "wheelchair", "stool", "barber chair", "folding chair", "lounge chair", "vertical back chair", "recliner", "wing chair", "sling"],  

"03211117": ["monitor", "crt monitor"],

"03636649": ["lamp",  "street lamp", "fluorescent lamp", "gas lamp", "bulb"],
    
"03691459": ["loudspeaker", "subwoofer speaker"],
    
"04090263": ["gun", "machine gun", "sniper rifle", "pistol", "shotgun"],    
 
"04256520": ["sofa", "double couch",  "love seat", "chesterfield", "convertiable sofa", "L shaped sofa", "settee sofa", "daybed", "sofa bed", "ottoman"],

"04379243": ["table", "dressing table", "desk", "refactory table",  "counter", "operating table", "stand", "billiard table", "pool table", "ping-pong table", "console table"],

 "04401088": ["phone", "desk phone", "flip-phone"],
   
"04530566": ["boat", "war ship", "sail boat", "speedboat", "cabin cruiser",  "yacht"],
    
}

id_to_shape_attribute = {

"02691156": ["triangular"],     

"02828884": ["square", "round", "circular", "rectangular", "thick", "thin"],    

"02933112": ["cuboid", "round", "rectangular", "thick", "thin"]  ,
    
"02958343": ["square", "round", "rectangular", "thick", "thin"],

"03001627": ["square", "round", "rectangular", "thick", "thin"],  

"03211117": ["square", "round", "rectangular", "thick", "thin"],

"03636649": ["square", "round", "rectangular", "cuboid", "circular", "thick", "thin"],
    
"03691459": ["square", "round", "rectangular", "circular", "thick", "thin"],
    
"04090263": ["thick", "thin"],    
 
"04256520": ["square", "round", "rectangular", "thick", "thin"],

"04379243": ["square", "round", "circular", "rectangular", "thick", "thin"],

"04401088": ["square", "rectangular", "thick", "thin"],
   
"04530566": ["square", "round", "rectangular", "thick", "thin"],
    
}

id_to_other_stuff = {

"02691156": ["boeing", "airbus", "f-16", "plane", "aeroplane", "aircraft", "commerical plane"],     

"02828884": ["park bench"],    

"02933112": ["dresser", "cupboard", "container", "case", "locker", "cupboard", "closet", "sideboard"]  ,
    
"02958343": ["auto", "automobile", "motor car"],

"03001627": ["seat", "cathedra"],  

"03211117": ["TV", "digital display", "flat panel display", "screen", "television", "telly", "video"],

"03636649": ["lantern", "table lamp", "torch"],
    
"03691459": ["speaker", "speaker unit", "tannoy"],
    
"04090263": ["ak-47", "uzi", "M1 Garand", "M-16","firearm", "shooter", "weapon"],    
 
"04256520": ["couch", "lounge", "divan", "futon"],

"04379243": ["altar table", "worktop", "workbench"],

"04401088": ["telephone",  "telephone set", "cellular telephone", "cellular phone", "cellphone", "cell", "mobile phone", "iphone"],
   
"04530566": ["rowing boat", "watercraft", "ship", "canal boat", "ferry", "steamboat", "barge"],
    
}

def generate_all_queries_2(prefix="a"):
    
    all_queries = []
    all_labels = []
    
    for category_id in id_to_sub_category:
        sub_category_queries = id_to_sub_category[category_id]
        main_category = sub_category_queries[0]
        
        new_prefix = prefix
        for shape_attributes_query in id_to_shape_attribute[category_id]:
            if prefix == "a" and shape_attributes_query[0] in ["a", "e", "i", "o", "u"]:
                new_prefix = "an"
            elif prefix == "a":
                new_prefix = "a"
            
            query = new_prefix + " " + shape_attributes_query + " " + main_category 
            all_queries.append(query)
            all_labels.append(id_to_label[category_id])
            
        for sub_category_query in sub_category_queries:
            if prefix == "a" and sub_category_query[0] in ["a", "e", "i", "o", "u"]:
                new_prefix = "an"
            elif prefix == "a":
                new_prefix = "a"
                
            query = new_prefix + " " + sub_category_query
            all_queries.append(query)
            all_labels.append(id_to_label[category_id])
            
        for other_query in id_to_other_stuff[category_id]:  
            if prefix == "a" and other_query[0] in ["a", "e", "i", "o", "u"]:
                new_prefix = "an"
            elif prefix == "a":
                new_prefix = "a"
                
            query = new_prefix + " " + other_query
            all_queries.append(query)
            all_labels.append(id_to_label[category_id])
            
                   
    return all_queries, all_labels

###################################### Text Queries ###############################################


def generate_voxel_32(net, latent_flow_model, clip_model, args, num_figs_per_query=5, prefix="a"):
    net.eval()
    latent_flow_model.eval()
    clip_model.eval()

    count = 1
    num_figs = num_figs_per_query
    with torch.no_grad():
        voxel_size = 32
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
        
        generated_voxel_array = []
        total_labels_array = []
        
        total_text_query, query_labels = generate_all_queries_2(prefix=prefix)
        print(total_text_query)
        
        count = 0 
        for  text_in in tqdm(total_text_query):
            ##########
            text = clip.tokenize([text_in]).to(args.device)
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ###########
            label = query_labels[count]
            
            mean_shape = torch.zeros(1, args.emb_dims).to(args.device) 
            noise = torch.Tensor(num_figs-1, args.emb_dims).normal_().to(args.device) 
            noise = torch.clip(noise, min=-1, max=1)
            noise = torch.cat([mean_shape, noise], dim=0).float()
            decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1).float())

            out = net.decoding(decoder_embs, query_points)
            voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            #print(voxels_out.shape)
            generated_voxel_array.append(voxels_out)
            total_labels_array.append(label)
            count = count + 1
            
        generated_voxel_array = np.concatenate(generated_voxel_array)
        total_labels_array = total_labels_array
        
        return generated_voxel_array, total_labels_array



def get_true_voxels(test_dataloader, args):
    voxel_array = []
    for data in tqdm(test_dataloader):
        data_input = data['voxels'].type(torch.FloatTensor).detach().cpu().numpy()
        voxel_array.append(data_input)
        #break
    voxel_array = np.concatenate(voxel_array)    
    return voxel_array

def voxel_save(voxels, text_name, out_file=None, transpose=True, show=False):

    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    #fig = plt.figure()
    fig = plt.figure(figsize=(40,20))
    
    ax = fig.add_subplot(111, projection=Axes3D.name)
    if transpose == True:
        voxels = voxels.transpose(2, 0, 1)
    #else:
        #voxels = voxels.transpose(2, 0, 1)
    

    ax.voxels(voxels, edgecolor='k', facecolors='coral', linewidth=0.5)
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    # Hide grid lines
    plt.grid(False)
    plt.axis('off')
    
    if text_name != None:
        plt.title(text_name, {'fontsize':30}, y=0.15)
    #plt.text(15, -0.01, "Correlation Graph between Citation & Favorite Count")

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.axis('off')
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def save_voxel_images(net, latent_flow_model, clip_model, args, total_text_query, save_path, resolution=64, num_figs_per_query=5):
    net.eval()
    latent_flow_model.eval()
    clip_model.eval()
    count = 1
    num_figs = num_figs_per_query
    with torch.no_grad():
        voxel_size = resolution
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
                
        for text_in in tqdm(total_text_query):
            ##########
            text = clip.tokenize([text_in]).to(args.device)
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ###########
            torch.manual_seed(5)
            mean_shape = torch.zeros(1, args.emb_dims).to(args.device) 
            noise = torch.Tensor(num_figs-1, args.emb_dims).normal_().to(args.device) 
            noise = torch.clip(noise, min=-1, max=1)
            noise = torch.cat([mean_shape, noise], dim=0)
            decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))

            out = net.decoding(decoder_embs, query_points)
            voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            
            voxel_num = 0
            for voxel_in in voxels_out:
                out_file = os.path.join(save_path, text_in + "_" + str(voxel_num) + ".png")
                voxel_save(voxel_in, None, out_file=out_file)
                voxel_num = voxel_num + 1
            


##################################### Main and Parser stuff #################################################3

def get_local_parser_test(mode="args"):
    parser = get_local_parser(mode="parser")
    parser.add_argument("--experiment_mode",  type=str, default='save_voxel_on_query', metavar='N', help='experiment type')
    parser.add_argument("--classifier_checkpoint",  type=str, default="./exps/classifier/checkpoints/best.pt", metavar='N', help='what is the classifier checkpoint for FID, Acc and Stuff')
    parser.add_argument("--checkpoint_nf",  type=str, default="best", metavar='N', help='what is the checkpoint for nf')
    parser.add_argument("--prefix",  type=str, default="a", metavar='N', help='add or remove')
    parser.add_argument("--post_dataset",  type=str, default=None, help='if want to use diff dataset during post')
    parser.add_argument("--checkpoint_dir_base",  type=str, default=None, help='Checkpoint directory for autoencoder')
    parser.add_argument("--output_dir",  type=str, default="./exps/output_dir", help='output dir')
    parser.add_argument("--checkpoint_dir_prior",  type=str, default=None, help='Checkpoint for prior')

    args = parser.parse_args()
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser
    
def main():
    args = get_local_parser_test() 
    
    ### Directories for generating stuff and logs    cls_cal_category
    test_log_filename = osp.join(args.output_dir, 'test_log.txt')
    helper.create_dir(args.output_dir)
    helper.setup_logging(test_log_filename, args.log_level, 'w')
    args.query_generate_dir =  osp.join(args.output_dir, 'query_generate_dir') + "/"                          
    helper.create_dir(args.query_generate_dir)
    args.vis_gen_dir =  osp.join(args.output_dir, 'vis_gen_dir') + "/"                          
    helper.create_dir(args.vis_gen_dir)

    manualSeed = args.seed_nf
    helper.set_seed(manualSeed)

    ### Dataloader stuff 
    if args.experiment_mode not in ["save_voxel_on_query", "cls_cal_single", "cls_cal_category"]:
        logging.info("#############################")
        train_dataloader, total_shapes  = get_dataloader(args, split="train")
        args.total_shapes = total_shapes
        logging.info("Train Dataset size: {}".format(total_shapes))
        val_dataloader, total_shapes_val  = get_dataloader(args, split="val")
        logging.info("Val Dataset size: {}".format(total_shapes_val))
        test_dataloader, total_shapes_test, test_dataset  = get_dataloader(args, split="test", dataset_flag=True)
        logging.info("Test Dataset size: {}".format(total_shapes_test))
        logging.info("#############################")

    device, gpu_array = helper.get_device(args)
    args.device = device     

    ### Network stuff 
    logging.info("#############################")

    net = autoencoder.get_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    logging.info("Loaded the autoencoder: {}".format(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt"))
    
    args, clip_model = get_clip_model(args)

    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    
    checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
    logging.info("Loaded the nf model: {}".format(checkpoint_nf_path))
    
    checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
    latent_flow_network.load_state_dict(checkpoint['model'])
    latent_flow_network.eval()


    logging.info("#############################")

    logging.info("Conducting the experiment {}".format(args.experiment_mode))

    if args.experiment_mode == "fid_cal":
        torch.multiprocessing.set_sharing_strategy('file_system')
        generated_voxels, _ = generate_voxel_32(net, latent_flow_network, clip_model, args, num_figs_per_query=1)
        true_voxels  = get_true_voxels(test_dataloader, args)
        logging.info("Size of  generated {} and true voxel is {}".format(generated_voxels.shape, true_voxels.shape))

        import classifier 
        from fid_cal import  calculate_activation_statistics, calculate_frechet_distance

        cls = classifier.classifier_32("Voxel_Encoder_BN", 13).to(args.device)
        cls_checkpoint = torch.load(args.classifier_checkpoint, map_location=args.device)
        cls.load_state_dict(cls_checkpoint['model'])
        activations1, _ = classifier.get_activations(true_voxels, cls, args)
        activations2, _ = classifier.get_activations(generated_voxels, cls, args)

        logging.info("Size of activatation for true {} and generated voxel is {}".format(activations1.shape, activations2.shape))

        mu1, sigma1 = calculate_activation_statistics(activations1)
        mu2, sigma2 = calculate_activation_statistics(activations2)
        fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
        logging.info("FID score is: {}".format(fid_score))

    elif args.experiment_mode == "cls_cal_single":
        torch.multiprocessing.set_sharing_strategy('file_system')
        generated_voxels, query_labels = generate_voxel_32(net, latent_flow_network, clip_model, args, num_figs_per_query=1)
        logging.info("Size of  generated voxel is {} and label length {}".format(generated_voxels.shape, len(query_labels)))
        
        import classifier 
        from fid_cal import  calculate_activation_statistics, calculate_frechet_distance
        from sklearn.metrics import accuracy_score
        
        cls = classifier.classifier_32("Voxel_Encoder_BN", 13).to(args.device)
        cls_checkpoint = torch.load(args.classifier_checkpoint, map_location=args.device)
        cls.load_state_dict(cls_checkpoint['model'])

        activations, pred_labels = classifier.get_activations(generated_voxels, cls, args)
        logging.info("Size of  activations  is {} and pred labels is {}".format(activations.shape, pred_labels.shape))

        acc = 100*accuracy_score(query_labels, pred_labels)
        logging.info("Cls score is: {}".format(acc))    
    elif args.experiment_mode == "cls_cal_category":
        torch.multiprocessing.set_sharing_strategy('file_system')
        generated_voxels, query_labels = generate_voxel_32(net, latent_flow_network, clip_model, args, num_figs_per_query=1)
     
        logging.info("Size of  generated voxel is {} and label length {}".format(generated_voxels.shape, len(query_labels)))
        
        import classifier 
        from fid_cal import  calculate_activation_statistics, calculate_frechet_distance
        from sklearn.metrics import accuracy_score
        
        cls = classifier.classifier_32("Voxel_Encoder_BN", 13).to(args.device)
        cls_checkpoint = torch.load(args.classifier_checkpoint, map_location=args.device)
        cls.load_state_dict(cls_checkpoint['model'])

        activations, pred_labels = classifier.get_activations(generated_voxels, cls, args)
        logging.info("Size of  activations  is {} and pred labels is {}".format(activations.shape, pred_labels.shape))
        
        conf_matrix = confusion_matrix(query_labels, pred_labels)
        count  = 0
        for i in conf_matrix:
            category_name =  label_to_category[count]
            total_labels = query_labels.count(count)
            acc = (conf_matrix[count,count]/ total_labels) *100
            logging.info("Cls score for class {}, total labels {} is: {}".format(category_name, total_labels, acc))
            count = count + 1
        acc = 100*accuracy_score(query_labels, pred_labels)
        logging.info("Cls score is: {}".format(acc))   
    elif args.experiment_mode == "save_voxel_on_query":
        save_path = args.vis_gen_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        torch.multiprocessing.set_sharing_strategy('file_system')
        if args.text_query is None:
            logging.info("Please add text query using text_query args argument")
        else: 
            save_voxel_images(net, latent_flow_network, clip_model, args, args.text_query, save_path, resolution=64, num_figs_per_query=1)
        
if __name__ == "__main__":
    main()  