# Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth

### Downloads

- [[Downloads]](https://drive.google.com/drive/folders/1UXTM7pH2lCogJx7kndtvhB1LywHQPisi) Predicted depth maps png files for NYU Depth V2 and KITTI Eigen split test set 

### Inference and Evaluate

#### Dataset
###### NYU Depth V2

```
$ cd ./datasets
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
$ python ../code/utils/extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

#### Evaluation

- evaluate with png images
for NYU Depth V2
```
python ./code/eval_with_pngs.py --dataset nyudepthv2 --pred_path ./best_nyu_preds/ --gt_path ./datasets/nyu_depth_v2/ --max_depth_eval 10.0 
```

<!---
- evaluate with pre-trained model
```
python ./code/inference.py --do_evaluate --dataset nyudepthv2
```
-->

### To-Do
- [ ] Add training codes
- [ ] Add Colab

### References

[1] From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation. [[code]](https://github.com/cleinc/bts)

[2] SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. [[code]](https://github.com/NVlabs/SegFormer)
