import os
import cv2
import json
import mediapipe as mp
import numpy as np
from pathlib import Path

# MediaPipe 初始化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 初始化姿势检测模型
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 关键点映射（MediaPipe标准索引）
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

def calculate_angle(point1, point2, point3):
    """计算三点之间的角度"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_trunk_angle(shoulder, hip):
    """计算躯干与垂直线的角度"""
    trunk_vector = [hip[0] - shoulder[0], hip[1] - shoulder[1]]
    vertical_vector = [0, 1]  # 垂直向下的向量
    
    dot_product = trunk_vector[0] * vertical_vector[0] + trunk_vector[1] * vertical_vector[1]
    magnitude_trunk = np.sqrt(trunk_vector[0]**2 + trunk_vector[1]**2)
    
    if magnitude_trunk == 0:
        return 0
    
    cos_angle = dot_product / magnitude_trunk
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def calculate_start_score(landmarks, w, h):
    """计算起点姿势分数"""
    # 提取关键点坐标
    coords = {}
    for name, idx in KEYPOINTS.items():
        landmark = landmarks[idx]
        coords[name] = [landmark.x * w, landmark.y * h]
    
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
        hip_score = max(1, 209 - int(hip_angle))  # 超过169°后，每增加1°扣1分
    else:
        hip_score = max(1, int(hip_angle) - 119) if hip_angle >= 120 else 1
        hip_score = min(hip_score, 40)

    # 膝关节角度计算
    left_knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
    right_knee_angle = calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
    knee_angle = max(left_knee_angle, right_knee_angle)
    knee_score = max(1, int(knee_angle) - 129) if knee_angle >= 130 else 1
    knee_score = min(knee_score, 30)

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
    """计算终点姿势分数"""
    # 提取关键点坐标
    coords = {}
    for name, idx in KEYPOINTS.items():
        landmark = landmarks[idx]
        coords[name] = [landmark.x * w, landmark.y * h]
    
    # 躯干角度计算
    left_trunk_angle = calculate_trunk_angle(coords['left_shoulder'], coords['left_hip'])
    right_trunk_angle = calculate_trunk_angle(coords['right_shoulder'], coords['right_hip'])
    trunk_angle = min(left_trunk_angle, right_trunk_angle)
    trunk_score = max(1, 30 - int(trunk_angle)) if trunk_angle <= 29 else 1

    # 髋关节角度计算（终点规则不同）
    left_hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
    right_hip_angle = calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])
    hip_angle = min(left_hip_angle, right_hip_angle)  # 终点取较小值
    hip_score = max(1, 80 - int(hip_angle)) if hip_angle <= 80 else 1
    hip_score = min(hip_score, 40)

    # 膝关节角度计算
    left_knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
    right_knee_angle = calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
    knee_angle = max(left_knee_angle, right_knee_angle)
    knee_score = max(1, int(knee_angle) - 129) if knee_angle >= 130 else 1
    knee_score = min(knee_score, 30)

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

def visualize_pose(image, landmarks, score_data, pose_type):
    """在图片上可视化姿势关键点和分数（可选功能）"""
    # 绘制姿势关键点
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, 
        landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    
    # 添加分数信息
    h, w, _ = image.shape
    cv2.putText(annotated_image, f"{pose_type.upper()} - Total Score: {score_data['total_score']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_image, f"Trunk: {score_data['trunk_score']}, Hip: {score_data['hip_score']}, Knee: {score_data['knee_score']}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_image

def process_image(image_path, pose_type, save_visualization=False):
    """处理单张图片并计算分数"""
    try:
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
            
        h, w, _ = image.shape
        
        # 转换为RGB格式（MediaPipe需要RGB格式）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用MediaPipe进行姿势检测
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"未检测到姿势关键点: {image_path}")
            return None
            
        landmarks = results.pose_landmarks.landmark
        
        # 检查所需关键点的可见性
        required_points = list(KEYPOINTS.values())
        for point_idx in required_points:
            if point_idx >= len(landmarks):
                print(f"关键点索引超出范围: {image_path}")
                return None
            if landmarks[point_idx].visibility < 0.5:
                print(f"关键点 {point_idx} 可见性低: {image_path}")
                # 注意：这里可以选择继续处理还是跳过
                # return None
        
        # 根据姿势类型计算分数
        if pose_type == 'begin':
            score_data = calculate_start_score(landmarks, w, h)
        elif pose_type == 'end':
            score_data = calculate_end_score(landmarks, w, h)
        else:
            print(f"未知的姿势类型: {pose_type}")
            return None
        
        # 可选：保存可视化结果
        if save_visualization:
            vis_image = visualize_pose(image, results.pose_landmarks, score_data, pose_type)
            vis_path = image_path.parent / f"vis_{image_path.name}"
            cv2.imwrite(str(vis_path), vis_image)
            
        return score_data
        
    except Exception as e:
        print(f"处理图片时出错 {image_path}: {str(e)}")
        return None

def process_pose_frames(base_path, save_visualizations=False):
    """处理姿势帧图片"""
    base_path = Path(base_path)
    results = []
    
    # 处理起点和终点文件夹
    for pose_type in ['begin', 'end']:
        pose_folder = base_path / pose_type
        
        if not pose_folder.exists():
            print(f"文件夹不存在: {pose_folder}")
            continue
            
        print(f"正在处理 {pose_type} 姿势帧...")
        
        # 处理每个视频文件夹（1-5）
        for video_num in range(1, 6):
            video_folder = pose_folder / str(video_num)
            
            if not video_folder.exists():
                print(f"  视频文件夹不存在: {video_folder}")
                continue
                
            print(f"  处理视频 {video_num} 的帧...")
            
            # 查找所有图片文件
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(video_folder.glob(ext))
            
            if not image_files:
                print(f"    未找到图片文件")
                continue
            
            # 按文件名排序
            image_files = sorted(image_files)
            
            # 存储当前视频的处理结果
            video_results = []
            
            for image_file in image_files:
                print(f"    处理帧: {image_file.name}")
                
                score_data = process_image(image_file, pose_type, save_visualizations)
                
                if score_data:
                    # 构建相对路径
                    relative_path = image_file.relative_to(base_path)
                    
                    # 获取图片尺寸
                    temp_image = cv2.imread(str(image_file))
                    img_height, img_width = temp_image.shape[:2]
                    
                    result_entry = {
                        'image_path': str(relative_path).replace('\\', '/'),
                        'pose_type': pose_type,
                        'video_id': video_num,
                        'frame_name': image_file.name,
                        'scores': score_data,
                        'image_dimensions': {
                            'width': img_width,
                            'height': img_height
                        }
                    }
                    
                    video_results.append(result_entry)
                    print(f"      ✓ 总分: {score_data['total_score']} "
                          f"(躯干:{score_data['trunk_score']} "
                          f"髋关节:{score_data['hip_score']} "
                          f"膝关节:{score_data['knee_score']})")
                else:
                    print(f"      ✗ 处理失败")
            
            # 对当前视频的结果进行填充到30条
            if video_results:
                padded_results = pad_video_data(video_results, pose_type, video_num)
                results.extend(padded_results)
                print(f"    视频{video_num}原始数据: {len(video_results)}条, 填充后: {len(padded_results)}条")
            else:
                print(f"    视频{video_num}没有有效数据")
    
    return results

def pad_video_data(video_results, pose_type, video_num, target_count=30):
    """对单个视频的数据进行填充到指定数量"""
    if len(video_results) == 0:
        return []
    
    # 如果数据已经达到或超过目标数量，直接返回
    if len(video_results) >= target_count:
        return video_results[:target_count]  # 如果超过30条，只取前30条
    
    # 使用最后一条数据进行填充
    last_entry = video_results[-1].copy()  # 深拷贝最后一条数据
    padded_results = video_results.copy()
    
    # 填充数据到目标数量
    fill_count = target_count - len(video_results)
    print(f"    需要填充 {fill_count} 条数据")
    
    for i in range(fill_count):
        # 创建填充数据项
        padded_entry = last_entry.copy()
        padded_entry['scores'] = last_entry['scores'].copy()
        padded_entry['image_dimensions'] = last_entry['image_dimensions'].copy()
        
        # 生成新的帧名称（基于最后一个帧的序号递增）
        original_frame_name = last_entry['frame_name']
        
        # 提取原始帧号（假设格式为frame_XXXXXX.jpg）
        if 'frame_' in original_frame_name:
            try:
                frame_base = original_frame_name.split('frame_')[1].split('.')[0]
                frame_number = int(frame_base) + i + 1
                new_frame_name = f"frame_{frame_number:06d}.jpg"
                padded_entry['frame_name'] = new_frame_name
                
                # 更新图片路径
                path_parts = padded_entry['image_path'].split('/')
                path_parts[-1] = new_frame_name
                padded_entry['image_path'] = '/'.join(path_parts)
                
            except (ValueError, IndexError):
                # 如果帧名格式不符合预期，使用序号命名
                new_frame_name = f"padded_frame_{i+1:03d}.jpg"
                padded_entry['frame_name'] = new_frame_name
                
                path_parts = padded_entry['image_path'].split('/')
                path_parts[-1] = new_frame_name
                padded_entry['image_path'] = '/'.join(path_parts)
        else:
            # 如果不是标准格式，使用序号命名
            new_frame_name = f"padded_frame_{i+1:03d}.jpg"
            padded_entry['frame_name'] = new_frame_name
            
            path_parts = padded_entry['image_path'].split('/')
            path_parts[-1] = new_frame_name
            padded_entry['image_path'] = '/'.join(path_parts)
        
        padded_results.append(padded_entry)
    
    return padded_results

def save_training_data(results, output_path):
    """保存训练数据"""
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    # 保存完整的标注数据（JSON格式）
    annotations_path = output_path / "pose_annotations.json"
    with open(annotations_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存简化的训练数据（CSV格式）
    training_data = []
    for result in results:
        training_entry = {
            'image_path': result['image_path'],
            'pose_type': result['pose_type'],
            'video_id': result['video_id'],
            'frame_name': result['frame_name'],
            'trunk_angle': result['scores']['trunk_angle'],
            'trunk_score': result['scores']['trunk_score'],
            'hip_angle': result['scores']['hip_angle'],
            'hip_score': result['scores']['hip_score'],
            'knee_angle': result['scores']['knee_angle'],
            'knee_score': result['scores']['knee_score'],
            'total_score': result['scores']['total_score'],
            'image_width': result['image_dimensions']['width'],
            'image_height': result['image_dimensions']['height']
        }
        training_data.append(training_entry)
    
    # 保存为CSV
    try:
        import pandas as pd
        df = pd.DataFrame(training_data)
        csv_path = output_path / "training_data.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV数据已保存: {csv_path}")
    except ImportError:
        print("未安装pandas，使用手动方式保存CSV")
        # 手动保存CSV
        csv_path = output_path / "training_data.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 写入表头
            headers = list(training_data[0].keys()) if training_data else []
            f.write(','.join(headers) + '\n')
            # 写入数据
            for entry in training_data:
                values = [str(entry[h]) for h in headers]
                f.write(','.join(values) + '\n')
        print(f"CSV数据已保存（手动写入）: {csv_path}")
    
    print(f"完整标注数据已保存: {annotations_path}")
    
    # 打印统计信息
    print_statistics(results)
    
    return annotations_path

def print_statistics(results):
    """打印数据统计信息"""
    print(f"\n=== 数据处理统计 ===")
    print(f"总处理图片数: {len(results)}")
    
    begin_results = [r for r in results if r['pose_type'] == 'begin']
    end_results = [r for r in results if r['pose_type'] == 'end']
    
    print(f"起点姿势帧: {len(begin_results)}")
    print(f"终点姿势帧: {len(end_results)}")
    
    # 按视频统计
    print(f"\n按视频统计（填充后）:")
    for video_id in range(1, 6):
        begin_count = len([r for r in begin_results if r['video_id'] == video_id])
        end_count = len([r for r in end_results if r['video_id'] == video_id])
        print(f"视频{video_id}: 起点 {begin_count} 帧, 终点 {end_count} 帧")
    
    # 检查填充情况
    print(f"\n填充状态检查:")
    for video_id in range(1, 6):
        for pose_type in ['begin', 'end']:
            video_data = [r for r in results if r['video_id'] == video_id and r['pose_type'] == pose_type]
            original_count = len([r for r in video_data if not r['frame_name'].startswith('padded_frame_')])
            padded_count = len([r for r in video_data if r['frame_name'].startswith('padded_frame_')])
            print(f"  视频{video_id}_{pose_type}: 原始{original_count}条, 填充{padded_count}条, 总计{len(video_data)}条")
    
    # 分数统计
    if begin_results:
        begin_scores = [r['scores']['total_score'] for r in begin_results]
        print(f"\n起点姿势分数统计:")
        print(f"  平均分: {np.mean(begin_scores):.2f}")
        print(f"  最高分: {np.max(begin_scores)}")
        print(f"  最低分: {np.min(begin_scores)}")
        print(f"  标准差: {np.std(begin_scores):.2f}")
    
    if end_results:
        end_scores = [r['scores']['total_score'] for r in end_results]
        print(f"\n终点姿势分数统计:")
        print(f"  平均分: {np.mean(end_scores):.2f}")
        print(f"  最高分: {np.max(end_scores)}")
        print(f"  最低分: {np.min(end_scores)}")
        print(f"  标准差: {np.std(end_scores):.2f}")

def main():
    """主函数"""
    # 设置路径
    train_path = "train"  # 您的train文件夹路径
    output_path = "annotations"  # 输出标注文件的路径
    
    # 是否保存可视化结果（用于调试和查看）
    save_visualizations = False  # 设置为True可以保存带关键点的可视化图片
    
    print("开始处理姿势评分数据（包含数据填充）...")
    print(f"输入路径: {train_path}")
    print(f"输出路径: {output_path}")
    print(f"保存可视化: {save_visualizations}")
    print(f"每个视频目标数据量: 30条")
    print("-" * 50)
    
    # 检查输入路径
    if not Path(train_path).exists():
        print(f"错误: 输入路径 '{train_path}' 不存在!")
        print("请确保文件夹结构如下:")
        print("train/")
        print("├── begin/")
        print("│   ├── 1/ (包含图片)")
        print("│   ├── 2/ (包含图片)")
        print("│   └── ...")
        print("└── end/")
        print("    ├── 1/ (包含图片)")
        print("    └── ...")
        return
    
    try:
        # 处理姿势帧数据
        results = process_pose_frames(train_path, save_visualizations)
        
        if results:
            # 保存训练数据
            save_training_data(results, output_path)
            print("\n✅ 处理完成! 数据已准备好用于训练。")
            print(f"输出文件:")
            print(f"  - pose_annotations.json (完整标注数据)")
            print(f"  - training_data.csv (训练数据)")
            if save_visualizations:
                print(f"  - vis_*.jpg (可视化图片)")
            
            print(f"\n📊 数据填充完成:")
            print(f"  - 每个视频的起点和终点数据都已填充到30条")
            print(f"  - 填充数据使用最后一条真实数据的分数")
            print(f"  - 填充的帧名称以'padded_frame_'开头便于识别")
            
        else:
            print("❌ 没有成功处理任何图片，请检查:")
            print("1. 路径是否正确")
            print("2. 图片格式是否支持 (jpg, jpeg, png, bmp)")
            print("3. 图片中是否包含清晰的人体姿势")
            print("4. MediaPipe是否正确安装")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 释放MediaPipe资源
        pose.close()

if __name__ == "__main__":
    main()
