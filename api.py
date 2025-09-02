import os
import cv2
import json
import mediapipe as mp
import numpy as np
import joblib
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import tempfile
from datetime import datetime

# ----------------- 日志配置 -----------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)  # 确保 logs 文件夹存在
log_file = LOG_DIR / f"pose_assess_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

app = Flask(__name__)

# MediaPipe 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 关键点映射
KEYPOINTS = {
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value
}

# ----------------- 加载模型 -----------------
MODEL_DIR = Path('models')
models = {}

# 只加载训练过的模型（不包括total_score）
individual_targets = ['trunk_score', 'hip_score', 'knee_score']

for target in individual_targets:
    model_path = MODEL_DIR / f'{target}_model.pkl'
    if model_path.exists():
        models[target] = joblib.load(model_path)
        logging.info(f"加载模型成功: {model_path}")
    else:
        logging.error(f"模型文件缺失: {model_path}")
        raise FileNotFoundError(f"模型文件缺失: {model_path}")

# 加载预处理工具
try:
    main_scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    label_encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    feature_columns = joblib.load(MODEL_DIR / 'feature_columns.pkl')
    logging.info("预处理工具加载完成（scaler, label_encoder, feature_columns）")
except FileNotFoundError as e:
    logging.error(f"预处理工具文件缺失: {e}")
    raise


# ----------------- 辅助函数 -----------------
def calculate_angle(point1, point2, point3):
    """计算三点之间的角度"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_trunk_angle(shoulder, hip):
    """计算躯干角度"""
    trunk_vector = [hip[0] - shoulder[0], hip[1] - shoulder[1]]
    vertical_vector = [0, 1]
    dot_product = trunk_vector[0] * vertical_vector[0] + trunk_vector[1] * vertical_vector[1]
    magnitude_trunk = np.sqrt(trunk_vector[0]**2 + trunk_vector[1]**2)
    if magnitude_trunk == 0:
        return 0
    cos_angle = dot_product / magnitude_trunk
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle) * 180 / np.pi


def calculate_start_score(landmarks, w, h):
    """计算起点姿势分数（规则-based）"""
    coords = {name: [landmarks[idx].x * w, landmarks[idx].y * h] for name, idx in KEYPOINTS.items()}
    
    # 躯干角度计算
    left_trunk_angle = calculate_trunk_angle(coords['left_shoulder'], coords['left_hip'])
    right_trunk_angle = calculate_trunk_angle(coords['right_shoulder'], coords['right_hip'])
    trunk_angle = min(left_trunk_angle, right_trunk_angle)
    trunk_score = max(1, 30 - int(trunk_angle)) if trunk_angle <= 29 else 1

    # 髋关节角度计算
    left_hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
    right_hip_angle = calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])
    hip_angle = max(left_hip_angle, right_hip_angle)
    if 159 <= hip_angle <= 169:
        hip_score = 40
    elif hip_angle > 169:
        hip_score = max(1, 209 - int(hip_angle))
    else:
        hip_score = max(1, int(hip_angle) - 119) if hip_angle >= 120 else 1
        hip_score = min(hip_score, 40)

    # 膝关节角度计算
    left_knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
    right_knee_angle = calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
    knee_angle = max(left_knee_angle, right_knee_angle)
    knee_score = max(1, int(knee_angle) - 129) if knee_angle >= 130 else 1
    knee_score = min(knee_score, 30)

    # 总分数 = 各部分分数之和
    total_score = trunk_score + hip_score + knee_score
    
    return {
        'trunk_angle': round(trunk_angle, 2),
        'trunk_score': trunk_score,
        'hip_angle': round(hip_angle, 2),
        'hip_score': hip_score,
        'knee_angle': round(knee_angle, 2),
        'knee_score': knee_score,
        'total_score': total_score
    }


def calculate_end_score(landmarks, w, h):
    """计算终点姿势分数（规则-based）"""
    coords = {name: [landmarks[idx].x * w, landmarks[idx].y * h] for name, idx in KEYPOINTS.items()}
    
    # 躯干角度计算
    left_trunk_angle = calculate_trunk_angle(coords['left_shoulder'], coords['left_hip'])
    right_trunk_angle = calculate_trunk_angle(coords['right_shoulder'], coords['right_hip'])
    trunk_angle = min(left_trunk_angle, right_trunk_angle)
    trunk_score = max(1, 30 - int(trunk_angle)) if trunk_angle <= 29 else 1

    # 髋关节角度计算
    left_hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
    right_hip_angle = calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])
    hip_angle = min(left_hip_angle, right_hip_angle)
    hip_score = max(1, 80 - int(hip_angle)) if hip_angle <= 80 else 1
    hip_score = min(hip_score, 40)

    # 膝关节角度计算
    left_knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
    right_knee_angle = calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
    knee_angle = max(left_knee_angle, right_knee_angle)
    knee_score = max(1, int(knee_angle) - 129) if knee_angle >= 130 else 1
    knee_score = min(knee_score, 30)

    # 总分数 = 各部分分数之和
    total_score = trunk_score + hip_score + knee_score
    
    return {
        'trunk_angle': round(trunk_angle, 2),
        'trunk_score': trunk_score,
        'hip_angle': round(hip_angle, 2),
        'hip_score': hip_score,
        'knee_angle': round(knee_angle, 2),
        'knee_score': knee_score,
        'total_score': total_score
    }


def predict_with_model(pose_type, video_id, trunk_angle, hip_angle, knee_angle, image_width, image_height):
    """使用 XGBoost 模型预测分数（确保总分=各部分分数之和）"""
    try:
        # 编码姿势类型
        pose_type_encoded = 0 if pose_type == 'begin' else 1
        
        # 构造特征
        features = {
            'pose_type_encoded': pose_type_encoded,
            'video_id': video_id,
            'trunk_angle': trunk_angle,
            'hip_angle': hip_angle,
            'knee_angle': knee_angle,
            'image_width': image_width,
            'image_height': image_height,
            'angle_ratio_hip_knee': hip_angle / (knee_angle + 1e-6),
            'angle_ratio_trunk_hip': trunk_angle / (hip_angle + 1e-6),
            'total_angle': trunk_angle + hip_angle + knee_angle,
            'aspect_ratio': image_width / image_height
        }
        
        # 转换为数组并标准化
        X = np.array([[features[col] for col in feature_columns]])
        X_scaled = main_scaler.transform(X)
        
        # 预测各部分分数
        predictions = {}
        
        # 预测躯干分数
        trunk_pred = models['trunk_score'].predict(X_scaled)[0]
        predictions['trunk_score'] = max(1, min(30, round(trunk_pred)))
        
        # 预测髋关节分数
        hip_pred = models['hip_score'].predict(X_scaled)[0]
        predictions['hip_score'] = max(1, min(40, round(hip_pred)))
        
        # 预测膝关节分数
        knee_pred = models['knee_score'].predict(X_scaled)[0]
        predictions['knee_score'] = max(1, min(30, round(knee_pred)))
        
        # 计算总分（确保等于各部分之和）
        predictions['total_score'] = (predictions['trunk_score'] + 
                                     predictions['hip_score'] + 
                                     predictions['knee_score'])
        
        return predictions
        
    except Exception as e:
        logging.error(f"模型预测出错: {e}")
        # 返回默认值
        return {
            'trunk_score': 15,
            'hip_score': 20,
            'knee_score': 15,
            'total_score': 50
        }


def process_video(video_path):
    """处理视频，提取起点和终点姿势，并计算分数"""
    logging.info(f"开始处理视频: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("无法打开视频文件")
        return None, "无法打开视频文件"

    max_hip_angle = -1
    min_hip_angle = float('inf')
    start_landmarks = None
    end_landmarks = None
    start_frame_size = None
    end_frame_size = None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            coords = {name: [landmarks[idx].x * w, landmarks[idx].y * h] for name, idx in KEYPOINTS.items()}
            
            # 计算髋关节角度
            left_hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
            right_hip_angle = calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])
            current_hip_angle = max(left_hip_angle, right_hip_angle)

            # 寻找起点姿势（髋关节角度最大）
            if current_hip_angle > max_hip_angle:
                max_hip_angle = current_hip_angle
                start_landmarks = landmarks
                start_frame_size = (w, h)

            # 寻找终点姿势（髋关节角度最小）
            current_hip_min = min(left_hip_angle, right_hip_angle)
            if current_hip_min < min_hip_angle:
                min_hip_angle = current_hip_min
                end_landmarks = landmarks
                end_frame_size = (w, h)

    cap.release()
    logging.info(f"处理了 {frame_count} 帧")

    if not start_landmarks or not end_landmarks:
        logging.warning("视频中未检测到有效姿势")
        return None, "视频中未检测到有效姿势"

    # 计算起点分数（规则 + 模型的平均值）
    rule_start = calculate_start_score(start_landmarks, *start_frame_size)
    model_start = predict_with_model('begin', 1,
                                     rule_start['trunk_angle'], rule_start['hip_angle'], rule_start['knee_angle'],
                                     start_frame_size[0], start_frame_size[1])
    
    start_scores = {
        'trunk_score': round((rule_start['trunk_score'] + model_start['trunk_score']) / 2),
        'hip_score': round((rule_start['hip_score'] + model_start['hip_score']) / 2),
        'knee_score': round((rule_start['knee_score'] + model_start['knee_score']) / 2),
    }
    # 确保总分等于各部分分数之和
    start_scores['total_score'] = start_scores['trunk_score'] + start_scores['hip_score'] + start_scores['knee_score']

    # 计算终点分数（规则 + 模型的平均值）
    rule_end = calculate_end_score(end_landmarks, *end_frame_size)
    model_end = predict_with_model('end', 1,
                                   rule_end['trunk_angle'], rule_end['hip_angle'], rule_end['knee_angle'],
                                   end_frame_size[0], end_frame_size[1])
    
    if rule_end['trunk_score'] + rule_end['hip_score'] + rule_end['knee_score'] < 60 and rule_end['trunk_score'] + rule_end['hip_score'] + rule_end['knee_score'] < 60 :
        end_scores = {
            'trunk_score': min((rule_end['trunk_score'], model_end['trunk_score'])),
            'hip_score': min((rule_end['hip_score'], model_end['hip_score'])),
            'knee_score': min((rule_end['knee_score'], model_end['knee_score'])),
        }
    elif rule_end['trunk_score'] + rule_end['hip_score'] + rule_end['knee_score'] >= 60 and rule_end['trunk_score'] + rule_end['hip_score'] + rule_end['knee_score'] >= 60 :
        end_scores = {
            'trunk_score': max((rule_end['trunk_score'], model_end['trunk_score'])),
            'hip_score': max((rule_end['hip_score'], model_end['hip_score'])),
            'knee_score': max((rule_end['knee_score'], model_end['knee_score'])),
        }
    else :
        end_scores = {
            'trunk_score': round((rule_end['trunk_score'] + model_end['trunk_score']) / 2),
            'hip_score': round((rule_end['hip_score'] + model_end['hip_score']) / 2),
            'knee_score': round((rule_end['knee_score'] + model_end['knee_score']) / 2),
        }
    # 确保总分等于各部分分数之和
    end_scores['total_score'] = end_scores['trunk_score'] + end_scores['hip_score'] + end_scores['knee_score']

    logging.info(f"起点分数: {start_scores}")
    logging.info(f"终点分数: {end_scores}")
    logging.info(f"起点角度 - 躯干: {rule_start['trunk_angle']}, 髋: {rule_start['hip_angle']}, 膝: {rule_start['knee_angle']}")
    logging.info(f"终点角度 - 躯干: {rule_end['trunk_angle']}, 髋: {rule_end['hip_angle']}, 膝: {rule_end['knee_angle']}")

    # 返回标准格式，只包含分数信息
    return {
        'start': {
            'trunk_score': start_scores['trunk_score'],
            'hip_score': start_scores['hip_score'], 
            'knee_score': start_scores['knee_score'],
            'total_score': start_scores['total_score']
        }, 
        'end': {
            'trunk_score': end_scores['trunk_score'],
            'hip_score': end_scores['hip_score'],
            'knee_score': end_scores['knee_score'], 
            'total_score': end_scores['total_score']
        }
    }, None


# ----------------- Flask API -----------------
@app.route('/analyze_pose', methods=['POST'])
def score_video():
    """分析视频中的姿势并返回评分"""
    if 'video' not in request.files:
        logging.warning("请求未包含视频文件")
        return jsonify({'error': '未上传视频文件'}), 400

    file = request.files['video']
    if file.filename == '':
        logging.warning("上传了空文件名")
        return jsonify({'error': '文件名为空'}), 400

    # 保存临时文件
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            file.save(temp_file.name)
            video_path = temp_file.name

        logging.info(f"收到视频文件: {file.filename}, 保存为临时文件: {video_path}")

        # 处理视频
        scores, error = process_video(video_path)
        
        # 清理临时文件
        os.unlink(video_path)

        if error:
            logging.error(f"视频处理出错: {error}")
            return jsonify({'error': error}), 500

        # 验证总分一致性
        start_total_check = scores['start']['trunk_score'] + scores['start']['hip_score'] + scores['start']['knee_score']
        end_total_check = scores['end']['trunk_score'] + scores['end']['hip_score'] + scores['end']['knee_score']
        
        if start_total_check != scores['start']['total_score']:
            logging.warning(f"起点总分不一致: 计算={start_total_check}, 返回={scores['start']['total_score']}")
            scores['start']['total_score'] = start_total_check
            
        if end_total_check != scores['end']['total_score']:
            logging.warning(f"终点总分不一致: 计算={end_total_check}, 返回={scores['end']['total_score']}")
            scores['end']['total_score'] = end_total_check

        # 确保返回格式完全符合要求
        result = {
            "start": {
                "trunk_score": scores['start']['trunk_score'],
                "hip_score": scores['start']['hip_score'],
                "knee_score": scores['start']['knee_score'],
                "total_score": scores['start']['total_score']
            },
            "end": {
                "trunk_score": scores['end']['trunk_score'],
                "hip_score": scores['end']['hip_score'],
                "knee_score": scores['end']['knee_score'], 
                "total_score": scores['end']['total_score']
            }
        }

        logging.info(f"最终返回结果: {result}")
        return jsonify(result), 200

    except Exception as e:
        logging.error(f"处理视频时发生异常: {e}")
        # 清理临时文件
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'available_models': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'loaded_models': list(models.keys()),
        'feature_columns': feature_columns,
        'model_dir': str(MODEL_DIR),
        'total_score_method': 'calculated_sum'  # 说明总分是通过计算得出的
    }), 200


if __name__ == '__main__':
    logging.info("动作打分服务已启动")
    logging.info(f"加载的模型: {list(models.keys())}")
    logging.info("总分数将通过各部分分数求和计算")    
    app.run(host='0.0.0.0', port=5000, debug=False)  # 生产环境建议关闭debug
