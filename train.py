import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import callback  # 新增导入，用于早停回调
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PoseScorePredictor:
    def __init__(self, data_path):
        """初始化姿势评分预测器"""
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def check_score_consistency(self):
        """检查分数一致性"""
        print("\n=== 检查分数一致性 ===")
        
        # 计算期望的总分
        self.data['expected_total'] = (self.data['trunk_score'] + 
                                      self.data['hip_score'] + 
                                      self.data['knee_score'])
        
        # 计算差异
        self.data['score_diff'] = self.data['total_score'] - self.data['expected_total']
        
        # 统计不一致的情况
        inconsistent_count = (self.data['score_diff'] != 0).sum()
        print(f"不一致的样本数量: {inconsistent_count}/{len(self.data)}")
        print(f"不一致比例: {inconsistent_count/len(self.data)*100:.2f}%")
        
        if inconsistent_count > 0:
            print(f"分数差异统计:")
            print(self.data['score_diff'].describe())
            
            # 显示一些不一致的样本
            print("\n不一致样本示例:")
            inconsistent_samples = self.data[self.data['score_diff'] != 0].head()
            print(inconsistent_samples[['trunk_score', 'hip_score', 'knee_score', 
                                       'total_score', 'expected_total', 'score_diff']])
        
        return inconsistent_count
    
    def fix_score_consistency(self):
        """修正分数一致性"""
        inconsistent_count = self.check_score_consistency()
        
        if inconsistent_count > 0:
            inconsistent_ratio = inconsistent_count / len(self.data)
            
            if inconsistent_ratio < 0.1:  # 不一致率小于10%
                print("不一致率较低，修正总分数为各部分分数之和")
                # 备份原始总分
                self.data['original_total_score'] = self.data['total_score'].copy()
                # 修正总分
                self.data['total_score'] = self.data['expected_total']
                print("总分数已修正完成")
            else:
                print(f"不一致率为{inconsistent_ratio*100:.2f}%，建议检查数据质量")
                print("将采用计算方式获得总分，不直接训练总分模型")
                return False
        else:
            print("所有样本的分数都保持一致性")
        
        return True
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        
        # 读取CSV文件
        df = pd.read_csv(self.data_path)
        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        
        # 检查数据基本信息
        print("\n数据基本信息:")
        print(df.info())
        print("\n缺失值统计:")
        print(df.isnull().sum())
        
        # 编码姿势类型
        df['pose_type_encoded'] = self.label_encoder.fit_transform(df['pose_type'])
        
        # 定义特征列（用于预测的输入特征）
        self.feature_columns = [
            'pose_type_encoded',  # 姿势类型
            'video_id',           # 视频ID
            'trunk_angle',        # 躯干角度
            'hip_angle',          # 髋关节角度  
            'knee_angle',         # 膝关节角度
            'image_width',        # 图像宽度
            'image_height'        # 图像高度
        ]
        
        # 检查特征列是否存在
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            print(f"警告：缺失特征列: {missing_features}")
        
        # 创建额外特征
        df['angle_ratio_hip_knee'] = df['hip_angle'] / (df['knee_angle'] + 1e-6)  # 髋膝角度比
        df['angle_ratio_trunk_hip'] = df['trunk_angle'] / (df['hip_angle'] + 1e-6)  # 躯干髋角度比
        df['total_angle'] = df['trunk_angle'] + df['hip_angle'] + df['knee_angle']  # 总角度
        df['aspect_ratio'] = df['image_width'] / df['image_height']  # 图像长宽比
        
        # 更新特征列
        self.feature_columns.extend([
            'angle_ratio_hip_knee',
            'angle_ratio_trunk_hip', 
            'total_angle',
            'aspect_ratio'
        ])
        
        self.data = df
        
        # 修正分数一致性问题
        self.consistent_scores = self.fix_score_consistency()
        
        # 定义目标列（根据一致性情况决定是否包含总分）
        if self.consistent_scores:
            self.target_columns = [
                'trunk_score',        # 躯干分数
                'hip_score',          # 髋关节分数
                'knee_score',         # 膝关节分数
                'total_score'         # 总分数
            ]
        else:
            # 如果不一致率过高，只训练各部分分数的模型
            self.target_columns = [
                'trunk_score',        # 躯干分数
                'hip_score',          # 髋关节分数
                'knee_score'          # 膝关节分数
            ]
        
        print(f"\n预处理后数据形状: {df.shape}")
        print(f"特征列数量: {len(self.feature_columns)}")
        print(f"目标列数量: {len(self.target_columns)}")
        
        return df
    
    def explore_data(self):
        """数据探索和可视化"""
        print("\n=== 数据探索 ===")
        
        # 姿势类型分布
        pose_counts = self.data['pose_type'].value_counts()
        print(f"\n姿势类型分布:")
        print(pose_counts)
        
        # 视频分布
        video_counts = self.data['video_id'].value_counts().sort_index()
        print(f"\n视频ID分布:")
        print(video_counts)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 姿势类型分布
        axes[0, 0].pie(pose_counts.values, labels=pose_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('姿势类型分布')
        
        # 分数分布对比
        for i, score_col in enumerate(['trunk_score', 'hip_score', 'knee_score']):
            row, col = divmod(i+1, 3)
            if col == 0:
                row -= 1
                col = 3
            
            sns.boxplot(data=self.data, x='pose_type', y=score_col, ax=axes[row, col-1])
            axes[row, col-1].set_title(f'{score_col}分布对比')
            axes[row, col-1].tick_params(axis='x', rotation=45)
        
        # 总分分布
        sns.histplot(data=self.data, x='total_score', hue='pose_type', ax=axes[1, 1])
        axes[1, 1].set_title('总分分布')
        
        # 角度相关性热力图
        angle_cols = ['trunk_angle', 'hip_angle', 'knee_angle']
        score_cols = ['trunk_score', 'hip_score', 'knee_score', 'total_score']
        corr_data = self.data[angle_cols + score_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=axes[1, 2])
        axes[1, 2].set_title('角度与分数相关性')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 统计信息
        print(f"\n各分数统计信息:")
        available_score_cols = [col for col in ['trunk_score', 'hip_score', 'knee_score', 'total_score'] 
                               if col in self.data.columns]
        print(self.data[available_score_cols].describe())
        
    def train_models(self, test_size=0.2, random_state=42):
        """训练XGBoost模型"""
        print("\n=== 开始训练模型 ===")
        
        # 准备特征数据
        X = self.data[self.feature_columns]
        
        # 检查是否有缺失值或无穷值
        print(f"特征矩阵形状: {X.shape}")
        print(f"特征列: {self.feature_columns}")
        
        if X.isnull().sum().sum() > 0:
            print("警告：存在缺失值，正在填充...")
            X = X.fillna(X.median())
            
        if np.isinf(X).sum().sum() > 0:
            print("警告：存在无穷值，正在处理...")
            X = X.replace([np.inf, -np.inf], X.median())
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.main_scaler = scaler
        
        # 只训练各部分分数的模型
        individual_targets = ['trunk_score', 'hip_score', 'knee_score']
        
        for target in individual_targets:
            print(f"\n训练 {target} 模型...")
            
            y = self.data[target]
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=self.data['pose_type']
            )
            # 打印训练集和测试集的基本信息
            print("=== 数据集划分结果 ===")
            print(f"训练集特征 X_train 形状: {X_train.shape}")
            print(f"测试集特征 X_test 形状: {X_test.shape}")
            print(f"训练集标签 y_train 形状: {y_train.shape}")
            print(f"测试集标签 y_test 形状: {y_test.shape}")

            print("\n=== 训练集标签分布 ===")
            print("y_train 值分布:")
            print(y_train.value_counts().sort_index())

            print("\n=== 测试集标签分布 ===")
            print("y_test 值分布:")
            print(y_test.value_counts().sort_index())

            # 如果需要查看具体的数值，可以打印前几行
            print("\n=== 训练集前5行样本 ===")
            print("X_train 前5行:")
            print(X_train[:5])
            print("\ny_train 前5个标签:")
            print(y_train[:5].values)

            print("\n=== 测试集前5行样本 ===")
            print("X_test 前5行:")
            print(X_test[:5])
            print("\ny_test 前5个标签:")
            print(y_test[:5].values)
            
            # XGBoost参数
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': random_state,
                'n_jobs': -1
            }
            
            # 训练模型（使用callbacks实现早停，兼容新版本XGBoost）
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
                callbacks=[callback.EarlyStopping(rounds=20)]
            )
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 评估
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            print(f"  训练集 - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
            print(f"  测试集 - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            
            # 保存模型
            self.models[target] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'feature_importance': dict(zip(self.feature_columns, model.feature_importances_))
            }
        
        # 验证总分的计算准确性
        self.validate_total_score_calculation()
        
        print("\n所有模型训练完成！")
    
    def validate_total_score_calculation(self):
        """验证总分计算的准确性"""
        print("\n=== 验证总分计算 ===")
        
        X = self.data[self.feature_columns]
        X_scaled = self.main_scaler.transform(X)
        
        # 预测各部分分数
        trunk_pred = self.models['trunk_score']['model'].predict(X_scaled)
        hip_pred = self.models['hip_score']['model'].predict(X_scaled)
        knee_pred = self.models['knee_score']['model'].predict(X_scaled)
        
        # 计算总分
        total_pred = trunk_pred + hip_pred + knee_pred
        
        # 与实际总分对比
        if 'total_score' in self.data.columns:
            total_true = self.data['total_score']
            r2 = r2_score(total_true, total_pred)
            rmse = np.sqrt(mean_squared_error(total_true, total_pred))
            mae = mean_absolute_error(total_true, total_pred)
            
            print(f"计算总分 vs 实际总分:")
            print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    def plot_feature_importance(self):
        """绘制特征重要性"""
        # 只绘制训练的模型的特征重要性
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (target, model_info) in enumerate(self.models.items()):
            # 获取特征重要性
            importance = model_info['feature_importance']
            features = list(importance.keys())
            values = list(importance.values())
            
            # 排序
            sorted_idx = np.argsort(values)[::-1]
            features = [features[j] for j in sorted_idx]
            values = [values[j] for j in sorted_idx]
            
            # 绘图
            axes[i].barh(features, values)
            axes[i].set_title(f'{target} - 特征重要性')
            axes[i].set_xlabel('重要性分数')
            
            # 显示数值
            for j, v in enumerate(values):
                axes[i].text(v, j, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_pose_scores(self, pose_type, video_id, trunk_angle, hip_angle, knee_angle, 
                          image_width=720, image_height=1280):
        """预测姿势分数，确保总分=各部分分数之和"""
        
        # 编码姿势类型
        if pose_type == 'begin':
            pose_type_encoded = 0
        elif pose_type == 'end':
            pose_type_encoded = 1
        else:
            pose_type_encoded = self.label_encoder.transform([pose_type])[0]
        
        # 构造特征向量
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
        X = np.array([[features[col] for col in self.feature_columns]])
        X_scaled = self.main_scaler.transform(X)
        
        # 预测各部分分数
        predictions = {}
        
        # 预测躯干分数
        trunk_pred = self.models['trunk_score']['model'].predict(X_scaled)[0]
        predictions['trunk_score'] = max(1, min(30, round(trunk_pred)))
        
        # 预测髋关节分数  
        hip_pred = self.models['hip_score']['model'].predict(X_scaled)[0]
        predictions['hip_score'] = max(1, min(40, round(hip_pred)))
        
        # 预测膝关节分数
        knee_pred = self.models['knee_score']['model'].predict(X_scaled)[0]
        predictions['knee_score'] = max(1, min(30, round(knee_pred)))
        
        # 计算总分（确保等于各部分之和）
        predictions['total_score'] = (predictions['trunk_score'] + 
                                     predictions['hip_score'] + 
                                     predictions['knee_score'])
        
        return predictions
    
    def evaluate_by_pose_type(self):
        """按姿势类型评估模型性能"""
        print("\n=== 按姿势类型评估模型 ===")
        
        X = self.data[self.feature_columns]
        X_scaled = self.main_scaler.transform(X)
        
        for pose_type in ['begin', 'end']:
            print(f"\n{pose_type.upper()} 姿势评估:")
            
            # 筛选对应姿势类型的数据
            mask = self.data['pose_type'] == pose_type
            X_pose = X_scaled[mask]
            
            # 评估各部分分数模型
            for target in self.models.keys():
                y_true = self.data[mask][target]
                y_pred = self.models[target]['model'].predict(X_pose)
                
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                
                print(f"  {target}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
            
            # 评估计算得到的总分
            if 'total_score' in self.data.columns:
                trunk_pred = self.models['trunk_score']['model'].predict(X_pose)
                hip_pred = self.models['hip_score']['model'].predict(X_pose)
                knee_pred = self.models['knee_score']['model'].predict(X_pose)
                total_pred = trunk_pred + hip_pred + knee_pred
                
                y_true_total = self.data[mask]['total_score']
                r2_total = r2_score(y_true_total, total_pred)
                rmse_total = np.sqrt(mean_squared_error(y_true_total, total_pred))
                mae_total = mean_absolute_error(y_true_total, total_pred)
                
                print(f"  计算总分: R² = {r2_total:.4f}, RMSE = {rmse_total:.4f}, MAE = {mae_total:.4f}")
    
    def save_models(self, save_dir='models'):
        """保存训练好的模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 保存模型
        for target, model_info in self.models.items():
            model_path = save_dir / f'{target}_model.pkl'
            joblib.dump(model_info['model'], model_path)
            print(f"模型已保存: {model_path}")
        
        # 保存标准化器
        scaler_path = save_dir / 'scaler.pkl'
        joblib.dump(self.main_scaler, scaler_path)
        
        # 保存标签编码器
        encoder_path = save_dir / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        
        # 保存特征列名
        feature_path = save_dir / 'feature_columns.pkl'
        joblib.dump(self.feature_columns, feature_path)
        
        # 保存目标列名（用于加载时知道训练了哪些模型）
        target_path = save_dir / 'target_columns.pkl'
        joblib.dump(list(self.models.keys()), target_path)
        
        print(f"所有模型组件已保存到: {save_dir}")
    
    def load_models(self, save_dir='models'):
        """加载训练好的模型"""
        save_dir = Path(save_dir)
        
        # 加载目标列名
        target_path = save_dir / 'target_columns.pkl'
        if target_path.exists():
            trained_targets = joblib.load(target_path)
        else:
            trained_targets = ['trunk_score', 'hip_score', 'knee_score']
        
        # 加载模型
        for target in trained_targets:
            model_path = save_dir / f'{target}_model.pkl'
            if model_path.exists():
                self.models[target] = {'model': joblib.load(model_path)}
                print(f"模型已加载: {model_path}")
        
        # 加载其他组件
        self.main_scaler = joblib.load(save_dir / 'scaler.pkl')
        self.label_encoder = joblib.load(save_dir / 'label_encoder.pkl')
        self.feature_columns = joblib.load(save_dir / 'feature_columns.pkl')
        
        print("所有模型组件加载完成")

def main():
    """主函数"""
    # 创建预测器
    predictor = PoseScorePredictor('annotations/training_data.csv')
    
    # 加载和预处理数据
    df = predictor.load_and_preprocess_data()
    
    # 数据探索
    predictor.explore_data()
    
    # 训练模型
    predictor.train_models()
    
    # 绘制特征重要性
    predictor.plot_feature_importance()
    
    # 按姿势类型评估
    predictor.evaluate_by_pose_type()
    
    # 保存模型
    predictor.save_models()
    
    # 测试预测功能
    print("\n=== 预测示例 ===")
    
    # 测试起点姿势预测
    begin_pred = predictor.predict_pose_scores(
        pose_type='begin',
        video_id=1, 
        trunk_angle=173.34,
        hip_angle=172.63, 
        knee_angle=178.15
    )
    print(f"起点姿势预测: {begin_pred}")
    print(f"验证总分: {begin_pred['trunk_score'] + begin_pred['hip_score'] + begin_pred['knee_score']} = {begin_pred['total_score']}")
    
    # 测试终点姿势预测
    end_pred = predictor.predict_pose_scores(
        pose_type='end',
        video_id=1,
        trunk_angle=165.0,
        hip_angle=75.0,
        knee_angle=170.0
    )
    print(f"终点姿势预测: {end_pred}")
    print(f"验证总分: {end_pred['trunk_score'] + end_pred['hip_score'] + end_pred['knee_score']} = {end_pred['total_score']}")
    
    print("\n训练完成！模型已保存，可用于新数据预测。")
    print("注意：总分数现在严格等于各部分分数之和！")

if __name__ == "__main__":
    main()
