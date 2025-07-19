# Scalpel: Automotive Deep Learning Framework Testing via Assembling Model Components

This is the implementation repository of our *ICSE'26* paper: **Scalpel: Automotive Deep Learning Framework Testing via
Assembling Model Components**.



## 1. Description

Deep learning (DL) plays a key role in autonomous driving systems. DL models support perception modules, equipped with tasks
such as object detection and sensor fusion. These DL models enable vehicles to process multi-sensor inputs to understand complex surroundings. Deploying DL models in autonomous driving systems faces stringent challenges, including real-time processing, limited computational resources, and strict power constraints. To address these challenges, automotive DL frameworks (e.g., Paddle-Inference) have emerged to optimize inference efficiency. However, these frameworks encounter unique quality issues due to their more complex deployment environments, such as crashes stemming from limited scheduled memory and incorrect memory allocation. Unfortunately, existing DL framework testing methods fail to detect these quality issues due to the failure in deploying generated test input models, as these models lack three essential capabilities: (1) multi-input/output tensor processing, (2) multi-modal data processing, and (3) multi-level data feature extraction. These capabilities
necessitate specialized model components, which existing testing methods neglect during model generation. To bridge this gap, we propose Scalpel, an automotive DL frameworks testing method that generates test input models at the model component level. Scalpel generates models by assembling model components (heads, necks, backbones) to support capabilities required by autonomous driving systems. Specifically, Scalpel maintains and updates a repository of model components, generating test inputs by selecting, mutating, and assembling them. Successfully generated models are added back to enrich the repository. Newly generated models are then deployed within the autonomous driving system to test automotive DL frameworks via differential testing. The experimental results demonstrate that Scalpel outperforms existing methods in both effectiveness and efficiency. In Apollo, Scalpel detects 16 crashes and 21 NaN & inconsistency bugs. All detected bugs have been reported to open-source communities, with 10 crashes confirmed. Scalpel achieves 27.44× and 8.5× improvements in model generation efficiency and bug detection efficiency. Additionally, Scalpel detects nine crashes and 16 NaN & inconsistency bugs in Autoware, which shows its excellent generalization.

You can access this repository using the following command:

```shell
git clone https://github.com/DLScalpel/Scalpel.git
```



## 2. API version

We deploy our method in the most widely used open-source autonomous driving system, Apollo and Autoware, and test their native automotive DL framework, PaddleInference and TensorRT. The adopted API versions are as follows.

| PaddlePaddle | PaddleInference | CUDA | CUDNN | NVIDIA-driver | Autoware | Apollo | TensorRT |
| :----------: | :-------------: | :--: | :---: | :-----------: | :------: | :----: | :------: |
|    2.6.2     |      2.6.2      | 12.4 | 9.6.0 |  535.216.01   | universe |  9.0   | 8.5.3.1  |



## 3. Environment

**Step 0:** Clone the source code of Apollo. Run the following command. Move it under the folder ***Scalpel***.

```sh
git clone git@github.com:ApolloAuto/apollo.git
cd apollo
git checkout master
```

Clone the source code of Autoware. Run the following command. Move it under the folder ***Scalpel***.

```
git clone https://github.com/autowarefoundation/autoware_universe.git
cd autoware_universe
git checkout main
```

**Make sure you have configured CUDA, CUDNN, and NVIDIA-driver properly**

**Step 1:** Modify the VERSION_X86_64 image version in docker/scripts/dev_start.sh as follows.

```sh
VERSION_X86_64="dev-x86_64-18.04-20231128_2222"
```

**Step 2:** Set up the container. Run the following command.

```sh
bash docker/scripts/dev_start.sh
```

**Step 3:** Enter the container.

```sh
bash docker/scripts/dev_into.sh
```

**Step 4:** Change the content in third_party/centerpoint_infer_op/workspace.bzl as follows.

```
"""Loads the paddlelite library"""
﻿
# Sanitize a dependency so that it works correctly from code that includes
# Apollo as a submodule.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
﻿
def clean_dep(dep):
    return str(Label(dep))
﻿
def repo():
    http_archive(
        name = "centerpoint_infer_op-x86_64",
        sha256 = "038470fc2e741ebc43aefe365fc23400bc162c1b4cbb74d8c8019f84f2498190",
        strip_prefix = "centerpoint_infer_op",
        urls = ["https://apollo-pkg-beta.bj.bcebos.com/archive/centerpoint_infer_op_cu118.tar.gz"],
    )
﻿
    http_archive(
        name = "centerpoint_infer_op-aarch64",
        sha256 = "e7c933db4237399980c5217fa6a81dff622b00e3a23f0a1deb859743f7977fc1",
        strip_prefix = "centerpoint_infer_op",
        urls = ["https://apollo-pkg-beta.bj.bcebos.com/archive/centerpoint_infer_op-linux-aarch64-1.0.0.tar.gz"],
    )
﻿
```

**Step 5:** Change the content in third_party/paddleinference/workspace.bzl as follows.

```
"""Loads the paddlelite library"""
﻿
# Sanitize a dependency so that it works correctly from code that includes
# Apollo as a submodule.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
﻿
def clean_dep(dep):
    return str(Label(dep))
﻿
def repo():
    http_archive(
        name = "paddleinference-x86_64",
        sha256 = "7498df1f9bbaf5580c289a67920eea1a975311764c4b12a62c93b33a081e7520",
        strip_prefix = "paddleinference",
        urls = ["https://apollo-pkg-beta.cdn.bcebos.com/archive/paddleinference-cu118-x86.tar.gz"],
    )
﻿
    http_archive(
        name = "paddleinference-aarch64",
        sha256 = "048d1d7799ffdd7bd8876e33bc68f28c3af911ff923c10b362340bd83ded04b3",
        strip_prefix = "paddleinference",
        urls = ["https://apollo-pkg-beta.bj.bcebos.com/archive/paddleinference-linux-aarch64-1.0.0.tar.gz"],
    )
﻿
```

**Step 6:** Set up the PaddlePaddle. Run the following command. 

```
pip install paddlepaddle-gpu==2.6.2
```

**Step 7:** Download the KITTI dataset in [The KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/user_submit.php). Among them, the left color images of object data set (12GB) is adopted for Single Camera Detection and Multiple Camera Detection, and the Velodyne point clouds (29GB) is adopted for LiDAR Detection. In addition, the camera calibration matrices of object data set (16MB), and the training labels of object data set (5MB) are also necessary for data preprocess.

**Step 8:** Download the dataset split file list using the following command:

```
wget https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz
```

**Step 9:** Organize the extracted data for image meta data according to the directory structure below.

```
$ tree kitti_dataset_root
kitti_dataset_root
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   └── val.txt
└── training
    ├── calib
    ├── image_2
    └── label_2
    └── velodyne
```

**(If you want to adopt Nuscenes dataset, please download it and organize it in the same way.)**

**Step 10:** Set paths in ***Scalpel/Datastruct/globalConfig.py*** to your path, including ***modeltype_and_configpath***, ***exported_model_path***, and ***exported_model_weight_path***.

**Step 11:** Set dataset paths in ***Scalpel/configs*** to the path of your dataset!

**Step 12:** Run Scalpel using the following command:

```python
python main.py
```

All data reported in all RQs are produced by step 12. Before executing, the [global configuration file](https://github.com/DLScalpel/Scalpel/blob/main/Scalpel/DataStruct/globalConfig.py) should be properly set. Specifically, the parameters "modeltype_and_configpath" and "this_modeltype" determine the deployment scenario of the model.

For "modeltype_and_configpath", it has three available values: 1) "MonoDetection_SingleImage" corresponds to the scenario "Camera-only detection". 2) "MultiViewDetection_Image_3DCoord" corresponds to the scenario "Camera-LiDAR detection". 3) 'LiDARDetection' corresponds to the scenario "LiDAR-only" detection.

For "this_modeltype", it has three available values: 1) "MonoDetection_SingleImage" corresponds to the scenario "Camera-only detection". 2) "MultiViewDetection_Image_3DCoord" corresponds to the scenario "Camera-LiDAR detection". 3) "LiDARDetection" corresponds to the scenario "LiDAR-only" detection. After loading the appropriate dataset and setting the correct parameters, executing Step 12 yields the experimental results presented in each RQ.

During execution, guarantee the Apollo container is on!!!

## 4. File structure

This project contains five folders. The **LEMON-master** folder is the downloaded open source code for LEMON. The **Muffin-main** folder is the downloaded open source code for Muffin. The **Gandalf-main** folder is the downloaded open source code for Gandalf. The **Scalpel** folder is the source code for our method. The **result** folder is the experimental result data. To know the execution methods of our baselines, please refer to the corresponding research papers. In this document, we will introduce how to run the source code for **Scalpel**.

In the source code for **Scalpel**, the program entry of the method is **main.py**. Run **main.py** to run DLMOSA after installing the experimental environment. The automatic crash log analysis method has been added in the folder **tools**!!!

If you do not want to reproduce the experiment, experimental results are available in the folder **result**. There are two folders in the folder **result**: 1) Folder **crash_logs** for the logs of all detected crashes. 2) Folder **NaN&inconsistency** for the logs of all detected NaN & Inconsistency bugs. **Experimental results for Autoware have been added here!!!**

## 5. What is passed into Scalpel for sketch generation?

Sketch template and datasets are passed into Scalpel for sketch generation. Among them, sketch template is the source code. Conceptually speaking, these source code defines the inputs, outputs, and required model structures for models in each deployment scenario. In implementation, these source code is composed of base classes. Each base class corresponds to a deployment scenario for models in autonomous driving systems. For example, the base class "base_mono_detection" corresponds to the scenario "Camera-only detection". The base class "base_LiDAR_detection" corresponds to the scenario "LiDAR-only detection". The base class "base_multiview_detection" corresponds to the scenario "Camera-LiDAR detection". All models in autonomous driving systems depend on their respective base classes. Datasets include KITTI and Nuscence. Scalpel judges the data modalities in datasets and determines the model deployment scenario based on the data modalities. According to the model deployment scenario, Scalpel modifies the sketch template to generate model sketches.
