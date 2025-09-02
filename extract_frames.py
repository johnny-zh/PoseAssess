import cv2
import os
import glob

def extract_all_frames(video_path, output_dir):
    """
    从视频文件中提取所有帧
    
    参数:
    video_path: 视频文件路径
    output_dir: 输出目录
    """
    # 获取视频文件名（不包含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 为当前视频创建输出文件夹
    video_output_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"正在处理视频: {video_name}")
    print(f"总帧数: {total_frames}, 帧率: {fps:.2f}, 视频时长: {duration:.2f}秒")
    print(f"预计将提取 {total_frames} 帧图像")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 构造输出文件名（使用6位数字补零）
        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)
        
        # 保存帧图像
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
        # 每处理100帧显示一次进度
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"进度: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    print(f"视频 {video_name} 所有帧提取完成！")
    print(f"共提取 {frame_count} 帧，保存到: {video_output_dir}")

def main():
    # 设置路径
    data_folder = "data"  # 视频文件所在文件夹
    output_folder = "all_frames"  # 输出文件夹名称改为all_frames更直观
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 查找data文件夹中的所有mp4文件
    mp4_files = glob.glob(os.path.join(data_folder, "*.mp4"))
    
    if len(mp4_files) == 0:
        print("在data文件夹中没有找到MP4文件")
        return
    
    print(f"找到 {len(mp4_files)} 个MP4文件:")
    for video_file in mp4_files:
        print(f"- {os.path.basename(video_file)}")
    
    print("\n开始提取所有帧...")
    
    # 处理每个视频文件
    total_extracted = 0
    for i, video_file in enumerate(mp4_files, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(mp4_files)}] 正在处理: {os.path.basename(video_file)}")
        print(f"{'='*50}")
        
        # 获取处理前的帧数，用于统计
        cap_temp = cv2.VideoCapture(video_file)
        frames_in_video = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        
        extract_all_frames(video_file, output_folder)
        total_extracted += frames_in_video
    
    print(f"\n{'='*50}")
    print("所有视频处理完成！")
    print(f"总共提取了约 {total_extracted} 帧图像")
    print(f"输出目录: {output_folder}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
