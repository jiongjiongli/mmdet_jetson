# 调优目标检测模型并在NVIDIA Jetson 平台部署教程



# 1 介绍

Github 项目链接



# 2 模型选择

```
cd ~
git clone https://github.com/jiongjiongli/mmdet_jetson.git
cd ~/mmdet_jetson


git clone --branch v3.0.0rc5 https://github.com/open-mmlab/mmdetection.git


module avail

module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.4.0-gcc-4.8.5-3cj
module load gcc/9.4.0-gcc-4.8.5

python -m venv  ~/mmdet_jetson/py38
source ~/mmdet_jetson/py38/bin/activate

cd ~/mmdet_jetson
pip install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmengine==0.7.1
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

git clone --branch v2.0.0rc4 https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .

pip install mmdet==3.0.0rc5
# cd ~/mmdet_jetson/mmdetection
# pip install -e .

pip install shapely
pip install numpy==1.23.1

mkdir ~/mmdet_jetson/model
wget -P ~/mmdet_jetson/model/ https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

ln -s ~/mmdet_jetson/model ~/mmdet_jetson/mmdetection/model
```



# 3 数据集和数据处理

数据集



```
mkdir ~/mmdet_jetson/data
wget -P ~/mmdet_jetson/data/ https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip
cd ~/mmdet_jetson/data
unzip balloon_dataset.zip


cd ~/mmdet_jetson
python prepare_data.py

ln -s ~/mmdet_jetson/data ~/mmdet_jetson/mmdetection/data
```

预处理

# 4 模型调优

```
cp ~/mmdet_jetson/rtmdet_tiny_1xb12-40e_balloon.py ~/mmdet_jetson/mmdetection/configs/rtmdet/

cd ~/mmdet_jetson/mmdetection
python tools/test.py configs/rtmdet/rtmdet_tiny_1xb12-40e_balloon.py model/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

cd ~/mmdet_jetson/mmdetection
python tools/train.py configs/rtmdet/rtmdet_tiny_1xb12-40e_balloon.py


cd ~/mmdet_jetson/mmdetection
python tools/test.py configs/rtmdet/rtmdet_tiny_1xb12-40e_balloon.py work_dirs/rtmdet_tiny_1xb12-40e_balloon/epoch_40.pth

```





# 5 模型部署



```
[input]
min_shape = [1, 3, 640, 640]
opt_shape = [1, 3, 640, 640]
max_shape = [1, 3, 640, 640]
```

`mmdeploy/configs/mmdet/detection/detection_tensorrt_static-300x300.py`



| Key             |                             |                            |       |           | Value              |
| --------------- | --------------------------- | -------------------------- | ----- | --------- | ------------------ |
| backend_config  | type                        |                            |       |           | tensorrt           |
|                 | common_config               | fp16_mode                  |       |           | False              |
|                 |                             | max_workspace_size         |       |           | 1 << 30            |
|                 | model_inputs                | input_shapes               | input | min_shape | [1, 3, 300, 300]   |
|                 |                             |                            |       | opt_shape | [1, 3, 300, 300]   |
|                 |                             |                            |       | max_shape | [1, 3, 300, 300]   |
| onnx_config     | type                        |                            |       |           | onnx               |
|                 | export_params               |                            |       |           | True               |
|                 | keep_initializers_as_inputs |                            |       |           | False              |
|                 | opset_version               |                            |       |           | 11                 |
|                 | save_file                   |                            |       |           | end2end.onnx       |
|                 | input_names                 |                            |       |           | ['input']          |
|                 | output_names                |                            |       |           | ['dets', 'labels'] |
|                 | input_shape                 |                            |       |           | (300, 300)         |
|                 | optimize                    |                            |       |           | True               |
| codebase_config | type                        |                            |       |           | mmdet              |
|                 | task                        |                            |       |           | ObjectDetection    |
|                 | model_type                  |                            |       |           | end2end            |
|                 | post_processing             | score_threshold            |       |           | 0.05               |
|                 |                             | confidence_threshold       |       |           | 0.005              |
|                 |                             | iou_threshold              |       |           | 0.5                |
|                 |                             | max_output_boxes_per_class |       |           |                    |
|                 |                             | pre_top_k                  |       |           | 5000               |
|                 |                             | keep_top_k                 |       |           | 100                |
|                 |                             | background_label_id        |       |           | -1                 |





# 6 mmdeploy源码分析



```
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
```



```
mmdeploy/tools/deploy.py
```



```
def main():

	deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
	
	# ir: intermediate representation, onnx or torchscript
	# ir_config or onnx_config
	ir_config = get_ir_config(deploy_cfg)
	
	# ir_type='onnx'
	# torch2ir('onnx')=torch2onnx
	# -> torch.onnx.export
    torch2ir(ir_type)(
        args.img,
        args.work_dir,
        ir_save_file,        # end2end.onnx
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=args.device)

	# onnx file path generated by method torch2ir
	backend_files = ir_files
    # backend=Backend.TENSORRT
    backend = get_backend(deploy_cfg)
    
    PIPELINE_MANAGER.set_log_level(log_level, [to_backend])
    if backend == Backend.TENSORRT:
        PIPELINE_MANAGER.enable_multiprocess(True, [to_backend])
    backend_files = to_backend(
        backend,
        ir_files,
        work_dir=args.work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=args.device,
        uri=args.uri)
        
        
    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
              args.device),
        kwargs=extra,
        ret_value=ret_value)

    # get pytorch model inference result, try visualize if possible
    create_process(
        'visualize pytorch model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, [checkpoint_path],
              args.test_img, args.device),
        kwargs=dict(
            backend=Backend.PYTORCH,
            output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
            show_result=args.show),
        ret_value=ret_value)
```





# 7 总结和评估





