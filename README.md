# PoseAssess 项目说明

## 项目概述

基于机器学习的姿态评估系统，使用XGBoost算法对视频中的人体姿态进行分析和评估，能够区分和识别起点姿势与终点姿势。

## 目录结构

```
PoseAssess/
├── all_frames/                    # 视频帧提取目录
│   └── [视频对应的所有帧图像文件]
│
├── annotations/                   # 标注数据目录
│   └── [训练所需的标注信息文件]
│
├── logs/                         # 系统日志目录
│   └── [训练和推理过程的日志文件]
│
├── models/                       # 模型存储目录
│   └── [预训练模型文件]
│
├── performers/                   # 性能评估目录
│   └── [预训练模型性能表现报告]
│
├── train/                        # 训练数据目录
│   └── [训练所需的图像信息]
│
├── video/                        # 原始视频目录
│   └── [原始视频流文件]
│
├── api.py                        # HTTP服务接口
├── extract_frames.py             # 视频抽帧脚本
├── preprocess_data.py            # 数据预处理和标注脚本
├── train.py                      # 模型训练脚本
├── requirements.txt              # Python依赖包列表
└── start.bat                     # 一键启动服务批处理脚本
```

## 核心功能模块

### 数据处理模块

* **extract\_frames.py**: 从原始视频中提取关键帧，支持批量处理和自定义采样率
* **preprocess\_data.py**: 数据预处理管道，包括图像标准化、特征提取和标注数据生成
* **annotations/**: 存储人工标注的姿态关键点信息，用于监督学习

### 模型训练模块

* **train.py**: 基于XGBoost的模型训练脚本，支持超参数调优和交叉验证
* **models/**: 存储训练完成的XGBoost模型文件和检查点
* **performers/**: 模型性能评估报告，包括准确率、召回率、F1分数等指标

### 服务接口模块

* **api.py**: Flask/FastAPI HTTP服务接口，提供姿态评估REST API
* **start.bat**: Windows批处理脚本，一键启动服务和相关依赖

## 技术架构

### 机器学习框架

* **XGBoost**: 核心训练算法，用于姿态分类和评估

  * 支持起点姿势识别
  * 支持终点姿势识别
  * 多类别分类能力
  * 高精度和快速推理

### 姿态识别流程

1. **视频输入**: 接收原始视频文件
2. **帧提取**: 智能采样提取关键帧
3. **姿态检测**: 提取人体关键点特征
4. **特征工程**: 构建姿态特征向量
5. **模型推理**: XGBoost模型分类预测
6. **结果输出**: 返回姿态评估结果

## 安装和使用

### 环境要求

```bash
# 安装依赖包
pip install -r requirements.txt
```

### 快速启动

```bash
# Windows环境一键启动
start.bat

# 或手动启动API服务
python api.py
```

### 数据准备

```bash
# 1. 视频抽帧
python extract_frames.py --input video/ --output all_frames/

# 2. 数据预处理和标注
python preprocess_data.py --frames all_frames/ --annotations annotations/

# 3. 模型训练
python train.py --data train/ --model models/ --log logs/
```

## 项目特点

* **智能姿态分析**: 基于XGBoost的高精度姿态分类算法
* **起终点识别**: 专门优化的起点姿势和终点姿势区分能力
* **批量处理**: 支持大规模视频数据的批量分析
* **性能监控**: 完整的模型性能评估和日志记录系统
* **服务化部署**: HTTP API接口支持，便于集成到其他系统
* **一键启动**: 简化的部署和启动流程

```
