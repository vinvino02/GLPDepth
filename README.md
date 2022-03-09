# Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth [[Paper]](https://arxiv.org/abs/2201.07436)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-path-networks-for-monocular/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=global-local-path-networks-for-monocular)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-path-networks-for-monocular/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=global-local-path-networks-for-monocular)

### Downloads
- [[Downloads]](https://drive.google.com/drive/folders/17yYbLZS2uQ6UVn5ET9RhVL0y_X3Ipl5_?usp=sharing) Trained ckpt files for NYU Depth V2 and KITTI
- [[Downloads]](https://drive.google.com/drive/folders/1LGNSKSaXguLTuCJ3Ay_UsYC188JNCK-j?usp=sharing) Predicted depth maps png files for NYU Depth V2 and KITTI Eigen split test set 

### Google Colab

<p>
<a href="https://colab.research.google.com/drive/1v6fzr4XusKdXAaeGZ1gKe1kh9Ce_WQhl?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>
Thanks for the great Colab demo from NielsRogge

### Requirements
Tested on 
```
python==3.7.7
torch==1.6.0
h5py==3.6.0
scipy==1.7.3
opencv-python==4.5.5
mmcv==1.4.3
timm=0.5.4
albumentations=1.1.0
tensorboardX==2.4.1
gdown==4.2.1
```
You can install above package with 
```
$ pip install -r requirements.txt
```

Or you can pull docker image with
```
$ docker pull doyeon0113/glpdepth
```
### Inference and Evaluate

#### Dataset
###### NYU Depth V2

```
$ cd ./datasets
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
$ python ../code/utils/extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```
###### KITTI
Download annotated depth maps data set (14GB) from [[link]](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) into ./datasets/kitti/data_depth_annotated
```
$ cd ./datasets/kitti/data_depth_annotated/
$ unzip data_depth_annotated.zip
```

With above two instrtuctions, you can perform eval_with_pngs.py/test.py for NYU Depth V2 and eval_with_pngs for KITTI.

To fully perform experiments, please follow [[BTS]](https://github.com/cleinc/bts/tree/master/pytorch) repository to obtain full dataset for NYU Depth V2 and KITTI datasets.

Your dataset directory should be
```
root
- nyu_depth_v2
  - bathroom_0001
  - bathroom_0002
  - ...
  - official_splits
- kitti
  - data_depth_annotated
  - raw_data
  - val_selection_cropped
```


#### Evaluation

- Evaluate with png images

  for NYU Depth V2
  ```
  $ python ./code/eval_with_pngs.py --dataset nyudepthv2 --pred_path ./best_nyu_preds/ --gt_path ./datasets/nyu_depth_v2/ --max_depth_eval 10.0 
  ```
  for KITTI
  ```
  $ python ./code/eval_with_pngs.py --dataset kitti --split eigen_benchmark --pred_path ./best_kitti_preds/ --gt_path ./datasets/kitti/ --max_depth_eval 80.0 --garg_crop
  ```
- Evaluate with model (NYU Depth V2)
  
  Result images will be saved in ./args.result_dir/args.exp_name (default: ./results/test)
   - To evaluate only
     ```
     $ python ./code/test.py --dataset nyudepthv2 --data_path ./datasets/ --ckpt_dir <path_for_ckpt> --do_evaluate  --max_depth 10.0 --max_depth_eval 10.0
     ```
   - To save pngs for eval_with_pngs
      ```
     $ python ./code/test.py --dataset nyudepthv2 --data_path ./datasets/ --ckpt_dir <path_for_ckpt> --save_eval_pngs  --max_depth 10.0 --max_depth_eval 10.0
     ```         
    
   - To save visualized depth maps
     ```
     $ python ./code/test.py --dataset nyudepthv2 --data_path ./datasets/ --ckpt_dir <path_for_ckpt> --save_visualize  --max_depth 10.0 --max_depth_eval 10.0
     ```
    
    In case of kitti, modify arguments to `--dataset kitti --max_depth 80.0 --max_depth_eval 80.0` and add `--kitti_crop [garg_crop or eigen_crop]`

#### Inference

- Inference with image directory
  ```
  $ python ./code/test.py --dataset imagepath --data_path <dir_to_imgs> --save_visualize
  ```
  
### Train

for NYU Depth V2
  ```
  $ python ./code/train.py --dataset nyudepthv2 --data_path ./datasets/ --max_depth 10.0 --max_depth_eval 10.0  
  ```
  for KITTI
  ```
  $ python ./code/train.py --dataset kitti --data_path ./datasets/ --max_depth 80.0 --max_depth_eval 80.0  --garg_crop
  ```

### To-Do
- [x] Add inference 
- [x] Add training codes
- [x] Add dockerHub link
- [x] Add colab


### License
For non-commercial purpose only (research, evaluation etc). We will update the license soon.

### Citation

```
@article{kim2022global,
  title={Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth},
  author={Kim, Doyeon and Ga, Woonghyun and Ahn, Pyungwhan and Joo, Donggyu and Chun, Sehwan and Kim, Junmo},
  journal={arXiv preprint arXiv:2201.07436},
  year={2022}
}
```

### References

[1] From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation. [[code]](https://github.com/cleinc/bts)

[2] SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. [[code]](https://github.com/NVlabs/SegFormer)
