

## CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation (CVPR 2022)  

![CLIP](/images/main.png)

Generating shapes using natural language can enable new ways of imagining and creating the things around us. While significant recent progress has been made in text-to-image generation, text-to-shape generation remains a challenging problem due to the unavailability of paired text and shape data at a large scale. We present a simple yet effective method for zero-shot text-to-shape generation that circumvents such data scarcity. Our proposed method, named CLIP-Forge, is based on a two-stage training process, which only depends on an unlabelled shape dataset and a pre-trained image-text network such as CLIP. Our method has the benefits of avoiding expensive inference time optimization, as well as the ability to generate multiple shapes for a given text. We not only demonstrate promising zero-shot generalization of the CLIP-Forge model qualitatively and quantitatively, but also provide extensive comparative evaluations to better understand its behavior.

Paper Link: [[Paper]](https://arxiv.org/pdf/2110.02624.pdf)

If you find our code or paper useful, you can cite at:

    @article{sanghi2021clip,
      title={Clip-forge: Towards zero-shot text-to-shape generation},
      author={Sanghi, Aditya and Chu, Hang and Lambourne, Joseph G and Wang, Ye and Cheng, Chin-Yi and Fumero, Marco},
      journal={arXiv preprint arXiv:2110.02624},
      year={2021}
    }

## Installation

First create an anaconda environment called `clip_forge` using
```
conda env create -f environment.yaml
conda activate clip_forge
```

Then, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision. Please change the CUDA version based on your requirements. 

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install git+https://github.com/openai/CLIP.git
pip install sklearn
```

Choose a folder to download the data, classifier and model: 
```
wget https://clip-forge-pretrained.s3.us-west-2.amazonaws.com/exps.zip
unzip exps.zip
```

## Training

For training, first you need to setup the dataset. We use the data prepared from occupancy networks (https://github.com/autonomousvision/occupancy_networks).
```
## Stage 1
python train_autoencoder.py --dataset_path /path/to/dataset/
 
## Stage 2
python train_post_clip.py  --dataset_path /path/to/dataset/ --checkpoint best_iou  --num_views 1 --text_query "a chair" "a limo" "a jet plane"
```

For Pointcloud code, please use the following code: 

```
## Stage 1
python train_autoencoder.py --dataset_path /path/to/dataset/ --input_type Pointcloud --output_type Pointcloud 
 
## Stage 2
python train_post_clip.py  --dataset_path /path/to/dataset/ --input_type Pointcloud --output_type Pointcloud  --checkpoint best  --num_views 1 --text_query "a chair" "a limo" "a jet plane"
```


## Inference

To generate shape renderings based on text query:
```
 python test_post_clip.py --checkpoint_dir_base "./exps/models/autoencoder" --checkpoint best_iou --checkpoint_nf best --experiment_mode save_voxel_on_query --checkpoint_dir_prior "./exps/models/prior" --text_query "a truck" "a round chair" "a limo" --threshold 0.1 --output_dir "./exps/hello_world"
```

The image rendering of the shapes will be present in output_dir. 

To calculate Accuracy, please make sure you have the classifier model. 
```
python test_post_clip.py --checkpoint_dir_base "./exps/models/autoencoder/" --checkpoint best_iou --checkpoint_nf best --experiment_mode cls_cal_category --checkpoint_dir_prior "./exps/models/prior/" --threshold 0.05 --classifier_checkpoint "./exps/classifier/checkpoints/best.pt"
```
To calculate FID, please make sure you have the classifier model and data loaded.
```
python test_post_clip.py --checkpoint_dir_base "./exps/models/autoencoder/" --checkpoint best_iou --checkpoint_nf best --experiment_mode fid_cal --dataset_path /path/to/dataset/ --checkpoint_dir_prior "./exps/models/prior/" --threshold 0.05 --classifier_checkpoint "./exps/classifier/checkpoints/best.pt"
```

## Inference Tips 

To get the optimal results use different threshold values as controlled by the argument `threshold` as shown in Figure 10 in the paper. We also recommend using world synonyms and text augmentation for best results. As the network is trained on Shapenet, we would recommend limiting the queries across the 13 categories present in ShapeNet. Note, we believe this method scales with data, but unfortunately public 3D data is limited. 



## Releasing Soon 

- [ ] Pointcloud code --> semi done (need to test code)
- [ ] Pretrained models for pointcloud experiments 



## Other interesting ideas 

- ClipMatrix (https://arxiv.org/pdf/2109.12922.pdf)
- Text2Mesh (https://threedle.github.io/text2mesh/)
- DreamFields (https://arxiv.org/pdf/2112.01455.pdf)
- https://arxiv.org/pdf/2203.13333.pdf







