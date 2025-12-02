"""
双色球LSTM预测模型（使用50期数据）
使用LSTM神经网络预测下一期双色球号码
训练时使用50期历史数据来预测下一期
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention, MultiHeadAttention, LayerNormalization, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler
from tensorflow.keras import backend as K
import pickle
import os
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler



def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss用于处理类别不平衡问题（适配sparse categorical）
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # 将sparse categorical转换为one-hot
        y_true_one_hot = K.one_hot(K.cast(y_true, 'int32'), K.int_shape(y_pred)[-1])
        y_true_one_hot = K.cast(y_true_one_hot, K.floatx())
        
        # 计算p_t（预测概率）
        p_t = K.sum(y_true_one_hot * y_pred, axis=-1)
        
        # 计算alpha_t
        alpha_t = y_true_one_hot * alpha + (1.0 - y_true_one_hot) * (1 - alpha)
        alpha_t = K.sum(alpha_t, axis=-1)
        
        # Focal Loss
        focal_loss = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t + epsilon)
        return K.mean(focal_loss)
    return focal_loss_fixed


def label_smoothing_crossentropy(smoothing=0.1):
    """
    标签平滑交叉熵损失
    有助于防止过拟合，提高泛化能力
    """
    def loss_fn(y_true, y_pred):
        num_classes = K.int_shape(y_pred)[-1]
        y_true_smooth = K.one_hot(K.cast(y_true, 'int32'), num_classes)
        y_true_smooth = y_true_smooth * (1.0 - smoothing) + smoothing / num_classes
        return K.categorical_crossentropy(y_true_smooth, y_pred)
    return loss_fn


def top_k_loss(k=5):
    """
    Top-K损失函数：只要预测在Top-K中就算对
    更适合彩票预测这种场景
    """
    def loss_fn(y_true, y_pred):
        # 获取Top-K预测
        top_k_values, top_k_indices = tf.nn.top_k(y_pred, k=k)
        # 检查真实标签是否在Top-K中
        y_true_expanded = K.expand_dims(K.cast(y_true, 'int32'), axis=-1)
        in_top_k = tf.reduce_any(tf.equal(top_k_indices, y_true_expanded), axis=-1)
        # 如果不在Top-K中，使用标准交叉熵；如果在，使用较小的损失
        standard_loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.where(in_top_k, standard_loss * 0.1, standard_loss)
    return loss_fn


def swish_activation(x):
    """
    Swish激活函数：x * sigmoid(x)
    在某些任务上比ReLU表现更好
    """
    return x * K.sigmoid(x)


def gelu_activation(x):
    """
    GELU激活函数：Gaussian Error Linear Unit
    在Transformer等模型中表现优异
    """
    return 0.5 * x * (1 + tf.math.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


class MeanTeacherCallback(Callback):
    """
    平均老师（EMA）权重更新回调
    """
    def __init__(self, teacher_model, alpha=0.99):
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = alpha

    def on_train_begin(self, logs=None):
        # 开始训练前，先对齐老师模型和学生模型的初始权重
        if self.teacher_model is not None:
            self.teacher_model.set_weights(self.model.get_weights())

    def on_train_batch_end(self, batch, logs=None):
        if self.teacher_model is None:
            return
        student_weights = self.model.get_weights()
        teacher_weights = self.teacher_model.get_weights()
        updated_weights = []
        for tw, sw in zip(teacher_weights, student_weights):
            updated_weights.append(self.alpha * tw + (1.0 - self.alpha) * sw)
        self.teacher_model.set_weights(updated_weights)



class SSQLSTMModel50:
    """
    使用50期数据训练的LSTM模型
    """
    def __init__(self, model_file="ssq_lstm_model_50.weights.h5", use_mean_teacher=True, teacher_alpha=0.99, use_classification=True):
        if not model_file.endswith(".weights.h5"):
            base, _ = os.path.splitext(model_file)
            model_file = f"{base}.weights.h5"
        self.model_file = model_file
        self.scaler_file = "scaler.pkl"
        self.processed_data_file = "processed_data_50.npz"  # 使用50期的数据文件
        self.data_file = "ssq_history.csv"
        self.model = None
        self.teacher_model = None
        self.use_mean_teacher = use_mean_teacher
        self.teacher_alpha = teacher_alpha
        self.use_classification = use_classification
        self.seq_length = 50  # 使用50期数据
        
    def load_data(self):
        """加载处理后的数据（50期）"""
        if not os.path.exists(self.processed_data_file):
            raise FileNotFoundError(f"处理后的数据文件不存在，请先运行 data_processor.py 处理数据（seq_length=50）")
        
        data = np.load(self.processed_data_file, allow_pickle=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        seq_length = int(data['seq_length'])
        
        if seq_length != 50:
            print(f"警告: 数据文件中的seq_length是{seq_length}，不是50。请重新处理数据。")
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 如果使用分类模式，需要转换标签格式
        if self.use_classification:
            # 先反标准化y_train和y_test
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            def inverse_transform_y(y_scaled):
                """批量反标准化y数据"""
                n_samples = len(y_scaled)
                dummy_features = np.zeros((n_samples, scaler.n_features_in_))
                dummy_features[:, :7] = y_scaled
                y_original = scaler.inverse_transform(dummy_features)[:, :7]
                return y_original
            
            y_train_original = inverse_transform_y(y_train)
            y_test_original = inverse_transform_y(y_test)
            
            # 将红球从连续值转换为类别索引（1-33 -> 0-32）
            y_train_classification = {}
            y_test_classification = {}
            
            for i in range(6):
                red_train = np.clip(np.round(y_train_original[:, i]), 1, 33).astype(int)
                red_test = np.clip(np.round(y_test_original[:, i]), 1, 33).astype(int)
                
                if np.any(red_train < 1) or np.any(red_train > 33):
                    red_train = np.clip(red_train, 1, 33)
                if np.any(red_test < 1) or np.any(red_test > 33):
                    red_test = np.clip(red_test, 1, 33)
                
                y_train_classification[f'red_ball_{i}'] = (red_train - 1).astype(int)
                y_test_classification[f'red_ball_{i}'] = (red_test - 1).astype(int)
            
            # 蓝球：将值转换为类别索引（1-16 -> 0-15）
            blue_train = np.clip(np.round(y_train_original[:, 6]), 1, 16).astype(int)
            blue_test = np.clip(np.round(y_test_original[:, 6]), 1, 16).astype(int)
            
            if np.any(blue_train < 1) or np.any(blue_train > 16):
                blue_train = np.clip(blue_train, 1, 16)
            if np.any(blue_test < 1) or np.any(blue_test > 16):
                blue_test = np.clip(blue_test, 1, 16)
            
            y_train_classification['blue_ball'] = (blue_train - 1).astype(int)
            y_test_classification['blue_ball'] = (blue_test - 1).astype(int)
            
            return X_train, X_test, y_train_classification, y_test_classification, seq_length
        else:
            return X_train, X_test, y_train, y_test, seq_length
    
    def build_model(self, input_shape, use_classification=True):
        """
        构建优化的LSTM模型（使用50期数据）
        优化点：
        1. 更深的网络结构（4层LSTM）
        2. 多层注意力机制
        3. 更多残差连接
        4. 改进的激活函数和正则化
        5. 更大的模型容量
        """
        if use_classification:
            # 优化版本：更深的网络和更好的架构
            inputs = Input(shape=input_shape)
            
            # ========== LSTM特征提取层 ==========
            # 第一层LSTM：提取基础特征（增加容量）
            lstm1 = LSTM(384, return_sequences=True, name='lstm1')(inputs)
            lstm1_norm = BatchNormalization(name='lstm1_norm')(lstm1)
            lstm1_drop = Dropout(0.25, name='lstm1_drop')(lstm1_norm)
            
            # 第二层LSTM：提取更深层特征
            lstm2 = LSTM(384, return_sequences=True, name='lstm2')(lstm1_drop)
            lstm2_norm = BatchNormalization(name='lstm2_norm')(lstm2)
            lstm2_drop = Dropout(0.25, name='lstm2_drop')(lstm2_norm)
            
            # 残差连接1：连接第一层和第二层
            if lstm1_drop.shape[-1] == lstm2_drop.shape[-1]:
                lstm2_drop = Add(name='lstm_residual1')([lstm1_drop, lstm2_drop])
            
            # 第三层LSTM：进一步提取特征
            lstm3 = LSTM(256, return_sequences=True, name='lstm3')(lstm2_drop)
            lstm3_norm = BatchNormalization(name='lstm3_norm')(lstm3)
            lstm3_drop = Dropout(0.25, name='lstm3_drop')(lstm3_norm)
            
            # 第四层LSTM：高级特征提取
            lstm4 = LSTM(256, return_sequences=True, name='lstm4')(lstm3_drop)
            lstm4_norm = BatchNormalization(name='lstm4_norm')(lstm4)
            lstm4_drop = Dropout(0.25, name='lstm4_drop')(lstm4_norm)
            
            # 残差连接2：连接第三层和第四层
            if lstm3_drop.shape[-1] == lstm4_drop.shape[-1]:
                lstm4_drop = Add(name='lstm_residual2')([lstm3_drop, lstm4_drop])
            
            # ========== 注意力机制层 ==========
            # 第一层注意力：关注短期模式
            attention1 = MultiHeadAttention(num_heads=8, key_dim=64, name='attention1')(lstm4_drop, lstm4_drop, lstm4_drop)
            attention1_norm = LayerNormalization(name='attention1_norm')(attention1)
            attention1_drop = Dropout(0.2, name='attention1_drop')(attention1_norm)
            
            # 残差连接：注意力层
            if lstm4_drop.shape[-1] == attention1_drop.shape[-1]:
                attention1_drop = Add(name='attention_residual1')([lstm4_drop, attention1_drop])
            
            # 第二层注意力：关注长期依赖
            attention2 = MultiHeadAttention(num_heads=8, key_dim=64, name='attention2')(attention1_drop, attention1_drop, attention1_drop)
            attention2_norm = LayerNormalization(name='attention2_norm')(attention2)
            attention2_drop = Dropout(0.2, name='attention2_drop')(attention2_norm)
            
            # 残差连接：第二层注意力
            if attention1_drop.shape[-1] == attention2_drop.shape[-1]:
                attention2_drop = Add(name='attention_residual2')([attention1_drop, attention2_drop])
            
            # ========== 最终LSTM层 ==========
            # 最后一层LSTM：汇总所有信息
            x = LSTM(256, return_sequences=False, name='lstm_final')(attention2_drop)
            x = BatchNormalization(name='final_norm')(x)
            x = Dropout(0.3, name='final_drop')(x)
            
            # ========== 共享特征层 ==========
            # 共享的全连接层（增加容量和深度，使用改进的激活函数）
            use_swish = True  # 使用Swish激活函数（可选：True/False，False则使用ReLU）
            
            if use_swish:
                shared = Dense(768, name='shared_dense1')(x)
                shared = Activation(swish_activation, name='shared_act1')(shared)
            else:
                shared = Dense(768, activation='relu', name='shared_dense1')(x)
            shared = BatchNormalization(name='shared_norm1')(shared)
            shared = Dropout(0.3, name='shared_drop1')(shared)
            
            if use_swish:
                shared = Dense(512, name='shared_dense2')(shared)
                shared = Activation(swish_activation, name='shared_act2')(shared)
            else:
                shared = Dense(512, activation='relu', name='shared_dense2')(shared)
            shared = BatchNormalization(name='shared_norm2')(shared)
            shared = Dropout(0.3, name='shared_drop2')(shared)
            
            if use_swish:
                shared = Dense(256, name='shared_dense3')(shared)
                shared = Activation(swish_activation, name='shared_act3')(shared)
            else:
                shared = Dense(256, activation='relu', name='shared_dense3')(shared)
            shared = BatchNormalization(name='shared_norm3')(shared)
            shared = Dropout(0.25, name='shared_drop3')(shared)
            
            if use_swish:
                shared = Dense(128, name='shared_dense4')(shared)
                shared = Activation(swish_activation, name='shared_act4')(shared)
            else:
                shared = Dense(128, activation='relu', name='shared_dense4')(shared)
            shared = BatchNormalization(name='shared_norm4')(shared)
            shared = Dropout(0.2, name='shared_drop4')(shared)
            
            # ========== 红球输出层 ==========
            # 红球输出：6个位置，每个位置33个类别（1-33）
            # 使用更深的网络和更好的正则化
            red_outputs = []
            for i in range(6):
                red_dense = Dense(384, activation='relu', name=f'red_dense1_{i}')(shared)
                red_dense = BatchNormalization(name=f'red_norm1_{i}')(red_dense)
                red_dense = Dropout(0.3, name=f'red_drop1_{i}')(red_dense)
                
                red_dense2 = Dense(256, activation='relu', name=f'red_dense2_{i}')(red_dense)
                red_dense2 = BatchNormalization(name=f'red_norm2_{i}')(red_dense2)
                red_dense2 = Dropout(0.25, name=f'red_drop2_{i}')(red_dense2)
                
                red_dense3 = Dense(128, activation='relu', name=f'red_dense3_{i}')(red_dense2)
                red_dense3 = BatchNormalization(name=f'red_norm3_{i}')(red_dense3)
                red_dense3 = Dropout(0.2, name=f'red_drop3_{i}')(red_dense3)
                
                red_dense4 = Dense(64, activation='relu', name=f'red_dense4_{i}')(red_dense3)
                red_dense4 = Dropout(0.15, name=f'red_drop4_{i}')(red_dense4)
                
                red_output = Dense(33, activation='softmax', name=f'red_ball_{i}')(red_dense4)
                red_outputs.append(red_output)
            
            # ========== 蓝球输出层 ==========
            # 蓝球输出：分类（16个类别，1-16）
            # 增强蓝球输出层：更深的网络和更大的容量，提高蓝球预测能力
            if use_swish:
                blue_dense = Dense(384, name='blue_dense1')(shared)
                blue_dense = Activation(swish_activation, name='blue_act1')(blue_dense)
            else:
                blue_dense = Dense(384, activation='relu', name='blue_dense1')(shared)
            blue_dense = BatchNormalization(name='blue_norm1')(blue_dense)
            blue_dense = Dropout(0.3, name='blue_drop1')(blue_dense)
            
            if use_swish:
                blue_dense2 = Dense(256, name='blue_dense2')(blue_dense)
                blue_dense2 = Activation(swish_activation, name='blue_act2')(blue_dense2)
            else:
                blue_dense2 = Dense(256, activation='relu', name='blue_dense2')(blue_dense)
            blue_dense2 = BatchNormalization(name='blue_norm2')(blue_dense2)
            blue_dense2 = Dropout(0.25, name='blue_drop2')(blue_dense2)
            
            if use_swish:
                blue_dense3 = Dense(128, name='blue_dense3')(blue_dense2)
                blue_dense3 = Activation(swish_activation, name='blue_act3')(blue_dense3)
            else:
                blue_dense3 = Dense(128, activation='relu', name='blue_dense3')(blue_dense2)
            blue_dense3 = BatchNormalization(name='blue_norm3')(blue_dense3)
            blue_dense3 = Dropout(0.2, name='blue_drop3')(blue_dense3)
            
            if use_swish:
                blue_dense4 = Dense(64, name='blue_dense4')(blue_dense3)
                blue_dense4 = Activation(swish_activation, name='blue_act4')(blue_dense4)
            else:
                blue_dense4 = Dense(64, activation='relu', name='blue_dense4')(blue_dense3)
            blue_dense4 = BatchNormalization(name='blue_norm4')(blue_dense4)
            blue_dense4 = Dropout(0.15, name='blue_drop4')(blue_dense4)
            
            blue_output = Dense(16, activation='softmax', name='blue_ball')(blue_dense4)
            
            # 组合所有输出
            outputs = red_outputs + [blue_output]
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # ========== 损失函数和优化器 ==========
            losses = {}
            loss_weights = {}
            
            # 使用改进的损失函数组合
            # 方案1：Focal Loss + 标签平滑（处理类别不平衡和过拟合）
            # 方案2：Top-K损失（更适合彩票预测场景）
            # 方案3：标准交叉熵（作为baseline）
            
            # 当前使用：Focal Loss（处理类别不平衡）+ 标签平滑（防止过拟合）
            use_focal_loss = True  # 是否使用Focal Loss
            use_label_smoothing = True  # 是否使用标签平滑
            smoothing_rate = 0.1  # 标签平滑率
            
            if use_focal_loss:
                # 使用Focal Loss处理类别不平衡
                # 蓝球使用更强的Focal Loss参数（更高的gamma），提高对难样本的关注
                focal_loss_fn_red = focal_loss(gamma=2.0, alpha=0.25)
                focal_loss_fn_blue = focal_loss(gamma=3.0, alpha=0.3)  # 蓝球使用更强的Focal Loss
                for i in range(6):
                    losses[f'red_ball_{i}'] = focal_loss_fn_red
                    loss_weights[f'red_ball_{i}'] = 1.0
                losses['blue_ball'] = focal_loss_fn_blue
                loss_weights['blue_ball'] = 2.0  # 进一步提高蓝球权重（从1.5提高到2.0）
            elif use_label_smoothing:
                # 使用标签平滑交叉熵
                label_smooth_fn = label_smoothing_crossentropy(smoothing=smoothing_rate)
                for i in range(6):
                    losses[f'red_ball_{i}'] = label_smooth_fn
                    loss_weights[f'red_ball_{i}'] = 1.0
                losses['blue_ball'] = label_smooth_fn
                loss_weights['blue_ball'] = 2.0  # 进一步提高蓝球权重
            else:
                # 标准交叉熵
                for i in range(6):
                    losses[f'red_ball_{i}'] = 'sparse_categorical_crossentropy'
                    loss_weights[f'red_ball_{i}'] = 1.0
                losses['blue_ball'] = 'sparse_categorical_crossentropy'
                loss_weights['blue_ball'] = 2.0  # 进一步提高蓝球权重
            
            # 编译模型（使用优化的Adam优化器 + 学习率调度）
            # 使用余弦退火学习率
            initial_lr = 0.001
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=initial_lr,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7,
                    amsgrad=True  # 使用AMSGrad变体，更稳定
                ),
                loss=losses,
                loss_weights=loss_weights,
                metrics={
                    **{f'red_ball_{i}': 'sparse_categorical_accuracy' for i in range(6)},
                    'blue_ball': 'sparse_categorical_accuracy'
                }
            )
        else:
            # 回归版本
            model = Sequential([
                LSTM(256, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(128, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(7, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def train(self, epochs=200, batch_size=24, validation_split=0.2):
        """
        训练优化的LSTM模型（使用50期数据）
        优化点：
        1. 增加训练轮数（200轮）
        2. 优化批次大小（24，更好的梯度估计）
        3. 改进回调函数策略
        """
        print("开始训练优化的LSTM模型（使用50期数据）...")
        print("=" * 60)
        print("优化配置:")
        print(f"  - 训练轮数: {epochs}")
        print(f"  - 批次大小: {batch_size}")
        print(f"  - 验证集比例: {validation_split}")
        print(f"  - 平均老师策略: {'启用' if self.use_mean_teacher else '未启用'}")
        print("=" * 60)
        
        # 加载数据
        X_train, X_test, y_train, y_test, seq_length = self.load_data()
        
        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, use_classification=self.use_classification)
        self.teacher_model = None
        
        print("\n模型结构:")
        self.model.summary()
        
        # 优化的回调函数
        # 1. 余弦退火学习率调度（更平滑的学习率衰减）
        def cosine_annealing_schedule(epoch, lr):
            """余弦退火学习率调度"""
            initial_lr = 0.001
            min_lr = 1e-7
            max_epochs = epochs
            if epoch < max_epochs * 0.1:  # 前10%保持初始学习率
                return initial_lr
            else:
                # 余弦退火
                progress = (epoch - max_epochs * 0.1) / (max_epochs * 0.9)
                return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=60,  # 进一步增加patience，给模型更多训练时间
                restore_best_weights=True,
                verbose=1,
                min_delta=0.00001,  # 更小的改进阈值
                mode='min'
            ),
            ModelCheckpoint(
                self.model_file,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode='min'
            ),
            # 余弦退火学习率调度
            LearningRateScheduler(
                cosine_annealing_schedule,
                verbose=1
            ),
            # 备用：基于验证损失的学习率衰减
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # 学习率衰减因子
                patience=25,  # 增加patience
                min_lr=1e-8,
                verbose=1,
                mode='min',
                cooldown=10,
                min_delta=0.0001
            )
        ]

        # 平均老师策略
        if self.use_mean_teacher:
            self.teacher_model = self.build_model(input_shape, use_classification=self.use_classification)
            teacher_callback = MeanTeacherCallback(self.teacher_model, alpha=self.teacher_alpha)
            callbacks.append(teacher_callback)
            print(f"已启用平均老师策略 (alpha={self.teacher_alpha})")

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        print("\n评估模型...")
        student_train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        student_test_loss = self.model.evaluate(X_test, y_test, verbose=0)

        if self.use_classification:
            print(f"学生模型训练集总损失: {student_train_loss[0]:.6f}")
            print(f"学生模型测试集总损失: {student_test_loss[0]:.6f}")
            # evaluate返回顺序：loss(0), red_ball_0_loss(1), red_ball_1_loss(2), ..., blue_ball_loss(7),
            #                   red_ball_0_acc(8), red_ball_1_acc(9), ..., blue_ball_acc(14)
            # 准确率的索引是：8, 9, 10, 11, 12, 13 (红球0-5), 14 (蓝球)
            
            # 计算Top-K准确率（更合理的评估指标）
            def calculate_topk_accuracy(y_true_dict, y_pred_list, k=5):
                """计算Top-K准确率"""
                topk_acc = {}
                for i in range(6):
                    true_labels = y_true_dict[f'red_ball_{i}']
                    pred_probs = y_pred_list[i]
                    topk_pred = np.argsort(pred_probs, axis=1)[:, -k:]
                    correct = np.sum([true_labels[j] in topk_pred[j] for j in range(len(true_labels))])
                    topk_acc[f'red_ball_{i}'] = correct / len(true_labels) * 100
                
                # 蓝球
                true_labels = y_true_dict['blue_ball']
                pred_probs = y_pred_list[6]
                topk_pred = np.argsort(pred_probs, axis=1)[:, -k:]
                correct = np.sum([true_labels[j] in topk_pred[j] for j in range(len(true_labels))])
                topk_acc['blue_ball'] = correct / len(true_labels) * 100
                return topk_acc
            
            # 计算Top-5准确率
            train_pred = self.model.predict(X_train, verbose=0)
            test_pred = self.model.predict(X_test, verbose=0)
            train_top5 = calculate_topk_accuracy(y_train, train_pred, k=5)
            test_top5 = calculate_topk_accuracy(y_test, test_pred, k=5)
            
            for i in range(6):
                acc_idx = 8 + i  # 红球i的准确率索引：8, 9, 10, 11, 12, 13
                if len(student_train_loss) > acc_idx:
                    train_acc = student_train_loss[acc_idx] * 100  # 转换为百分比
                    test_acc = student_test_loss[acc_idx] * 100
                    train_top5_acc = train_top5[f'red_ball_{i}']
                    test_top5_acc = test_top5[f'red_ball_{i}']
                    print(f"  红球{i+1} - Top-1准确率: 训练{train_acc:.2f}%, 测试{test_acc:.2f}% | Top-5准确率: 训练{train_top5_acc:.2f}%, 测试{test_top5_acc:.2f}% (随机猜测: 3.03%)")
            # 蓝球准确率索引：14
            blue_acc_idx = 14
            if len(student_train_loss) > blue_acc_idx:
                train_acc = student_train_loss[blue_acc_idx] * 100
                test_acc = student_test_loss[blue_acc_idx] * 100
                train_top5_acc = train_top5['blue_ball']
                test_top5_acc = test_top5['blue_ball']
                print(f"  蓝球 - Top-1准确率: 训练{train_acc:.2f}%, 测试{test_acc:.2f}% | Top-5准确率: 训练{train_top5_acc:.2f}%, 测试{test_top5_acc:.2f}% (随机猜测: 6.25%)")
        else:
            print(f"学生模型训练集损失: {student_train_loss[0]:.6f}, MAE: {student_train_loss[1]:.6f}")
            print(f"学生模型测试集损失: {student_test_loss[0]:.6f}, MAE: {student_test_loss[1]:.6f}")

        if self.teacher_model is not None:
            teacher_train_loss = self.teacher_model.evaluate(X_train, y_train, verbose=0)
            teacher_test_loss = self.teacher_model.evaluate(X_test, y_test, verbose=0)

            if self.use_classification:
                print(f"教师模型训练集总损失: {teacher_train_loss[0]:.6f}")
                print(f"教师模型测试集总损失: {teacher_test_loss[0]:.6f}")
            else:
                print(f"教师模型训练集损失: {teacher_train_loss[0]:.6f}, MAE: {teacher_train_loss[1]:.6f}")
                print(f"教师模型测试集损失: {teacher_test_loss[0]:.6f}, MAE: {teacher_test_loss[1]:.6f}")

            self.model = self.teacher_model

        # 保存训练历史
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("training_history_50.csv", index=False)
        print("训练历史已保存到 training_history_50.csv")

        # 保存最终模型权重
        self.model.save_weights(self.model_file)
        status = '启用' if self.teacher_model is not None else '未启用'
        print(f"最终模型权重已保存到 {self.model_file} (平均老师策略: {status})")

        return history
    
    def load_model(self):
        """加载已训练的模型"""
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"模型文件 {self.model_file} 不存在，请先训练模型")
        
        # 加载数据以获取input_shape
        X_train, _, _, _, _ = self.load_data()
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 构建模型结构
        self.model = self.build_model(input_shape, use_classification=self.use_classification)
        
        # 加载权重
        self.model.load_weights(self.model_file)
        print(f"模型已从 {self.model_file} 加载（使用50期数据）")
        
        return self.model
    
    def _prepare_features_from_csv(self, df):
        """
        从CSV数据准备增强的特征（与data_processor_50.py保持一致）
        包括：基础特征 + 频率特征 + 趋势特征 + 相关性特征
        """
        red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
        blue_col = '蓝球'
        
        red_balls = df[red_cols].astype(int).values
        blue_balls = df[blue_col].astype(int).values
        
        # 计算历史频率（用于频率特征）
        number_frequency = {}  # 记录每个号码的出现频率
        for num in range(1, 34):
            number_frequency[num] = []
        
        features = []
        for i in range(len(df)):
            red = red_balls[i]
            blue = blue_balls[i]
            
            red_sorted = sorted(red)
            
            # ========== 基础统计特征 ==========
            red_sum = sum(red_sorted)
            red_mean = np.mean(red_sorted)
            red_std = np.std(red_sorted)
            red_max = max(red_sorted)
            red_min = min(red_sorted)
            red_range = red_max - red_min
            red_median = np.median(red_sorted)
            red_odd_count = sum(1 for x in red_sorted if x % 2 == 1)
            red_even_count = 6 - red_odd_count
            red_small_count = sum(1 for x in red_sorted if x <= 17)
            red_large_count = 6 - red_small_count
            sum_zone = red_sum // 50
            
            # ========== 频率特征（最近N期的出现频率）==========
            # 计算每个号码在最近10期的出现频率
            freq_window = min(10, i)  # 使用最近10期，如果数据不足则用全部
            if freq_window > 0:
                recent_reds = red_balls[max(0, i-freq_window):i].flatten()
                number_counts = {}
                for num in range(1, 34):
                    number_counts[num] = np.sum(recent_reds == num)
                # 当前期号码的频率特征
                current_freq = [number_counts[num] for num in red_sorted]
                avg_freq = np.mean(current_freq)
                max_freq = max(current_freq) if current_freq else 0
                min_freq = min(current_freq) if current_freq else 0
            else:
                current_freq = [0] * 6
                avg_freq = 0
                max_freq = 0
                min_freq = 0
            
            # ========== 趋势特征 ==========
            # 计算号码的移动平均和趋势
            if i >= 3:
                # 3期移动平均
                recent_reds = red_balls[i-3:i]
                ma3_sum = np.mean([np.sum(r) for r in recent_reds])
                ma3_mean = np.mean([np.mean(r) for r in recent_reds])
                trend_sum = red_sum - ma3_sum  # 和值趋势
                trend_mean = red_mean - ma3_mean  # 均值趋势
            else:
                ma3_sum = red_sum
                ma3_mean = red_mean
                trend_sum = 0
                trend_mean = 0
            
            if i >= 5:
                # 5期移动平均
                recent_reds = red_balls[i-5:i]
                ma5_sum = np.mean([np.sum(r) for r in recent_reds])
                ma5_mean = np.mean([np.mean(r) for r in recent_reds])
            else:
                ma5_sum = red_sum
                ma5_mean = red_mean
            
            # ========== 相关性特征 ==========
            # 号码之间的间隔
            intervals = [red_sorted[j+1] - red_sorted[j] for j in range(5)]
            avg_interval = np.mean(intervals)
            max_interval = max(intervals)
            min_interval = min(intervals)
            interval_std = np.std(intervals)
            
            # 连号数量（相邻号码差为1）
            consecutive_count = sum(1 for interval in intervals if interval == 1)
            
            # 号码分布特征（三个区间：1-11, 12-22, 23-33）
            zone1_count = sum(1 for x in red_sorted if 1 <= x <= 11)
            zone2_count = sum(1 for x in red_sorted if 12 <= x <= 22)
            zone3_count = sum(1 for x in red_sorted if 23 <= x <= 33)
            
            # 号码跨度特征（最大最小值的差）
            span_ratio = red_range / 33.0  # 跨度比例
            
            # ========== 组合特征 ==========
            # 红球与蓝球的和
            total_sum = red_sum + blue
            
            # 红球与蓝球的差值
            red_blue_diff = abs(red_mean - blue)
            
            # 号码的方差
            red_variance = np.var(red_sorted)
            
            # ========== 构建特征向量 ==========
            feature_vector = [
                # 基础特征（7个号码）
                *red_sorted,
                blue,
                # 基础统计特征（12个）
                red_sum,
                red_mean,
                red_std,
                red_median,
                red_range,
                red_odd_count,
                red_even_count,
                red_small_count,
                red_large_count,
                sum_zone,
                red_variance,
                # 频率特征（4个）
                avg_freq,
                max_freq,
                min_freq,
                np.std(current_freq) if current_freq else 0,
                # 趋势特征（6个）
                ma3_sum,
                ma3_mean,
                ma5_sum,
                ma5_mean,
                trend_sum,
                trend_mean,
                # 相关性特征（8个）
                avg_interval,
                max_interval,
                min_interval,
                interval_std,
                consecutive_count,
                zone1_count,
                zone2_count,
                zone3_count,
                span_ratio,
                # 组合特征（3个）
                total_sum,
                red_blue_diff,
            ]
            
            features.append(feature_vector)
            
            # 更新频率统计
            for num in red_sorted:
                number_frequency[num].append(i)
        
        return np.array(features)
    
    def predict_next(self, use_probability_sampling=False, random_seed=None, use_latest_data=True):
        """
        预测下一期号码（使用从第1期到当前期的所有历史数据）
        
        改进的预测策略：
        1. 红球：使用加权平均概率 + 温度采样，避免出现等差数列或规律性号码
        2. 蓝球：动态调整温度采样，根据概率分布集中度自动选择最佳采样策略
        3. 增加随机性和多样性，避免总是预测相同或相似的号码
        """
        if self.model is None:
            self.load_model()
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 加载数据
        if use_latest_data and os.path.exists(self.data_file):
            print("从CSV文件读取最新数据...")
            df = pd.read_csv(self.data_file, encoding='utf-8-sig')
            df = df.sort_values('期号', ascending=True)
            
            print(f"数据总期数: {len(df)}")
            if len(df) > 0:
                print(f"最新期号: {df.iloc[-1]['期号']}")
                latest_red = [df.iloc[-1][f'红球{i+1}'] for i in range(6)]
                latest_blue = df.iloc[-1]['蓝球']
                print(f"最新开奖: 红球{latest_red}, 蓝球{latest_blue}")
            
            features = self._prepare_features_from_csv(df)
            
            if not os.path.exists(self.scaler_file):
                raise FileNotFoundError(f"Scaler文件 {self.scaler_file} 不存在")
            
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            features_scaled = scaler.transform(features)
            
            # 使用从第1期到当前期的所有数据
            actual_seq_length = len(features_scaled)
            
            # 从processed_data.npz获取模型训练时的最大序列长度
            if os.path.exists(self.processed_data_file):
                data = np.load(self.processed_data_file, allow_pickle=True)
                model_seq_length = int(data['seq_length'])
            else:
                model_seq_length = actual_seq_length
            
            # 如果实际数据长度小于模型需要的长度，用最早的数据填充
            if actual_seq_length < model_seq_length:
                padding_needed = model_seq_length - actual_seq_length
                padding = np.tile(features_scaled[0:1], (padding_needed, 1))
                features_scaled = np.vstack([padding, features_scaled])
                print(f"警告: 数据不足，使用 {actual_seq_length} 期数据 + {padding_needed} 期填充")
            elif actual_seq_length > model_seq_length:
                # 如果实际数据长度大于模型需要的长度，只使用最近的model_seq_length期
                features_scaled = features_scaled[-model_seq_length:]
                print(f"注意: 数据有 {actual_seq_length} 期，模型使用最近 {model_seq_length} 期")
            else:
                print(f"使用所有 {actual_seq_length} 期数据进行预测")
            
            # 显示用于预测的数据（显示最近20期，如果数据少于20期则显示全部）
            display_count = min(20, len(df))
            print(f"\n用于预测的数据（显示最近 {display_count} 期）:")
            for i in range(max(0, len(df) - display_count), len(df)):
                red = [df.iloc[i][f'红球{j+1}'] for j in range(6)]
                blue = df.iloc[i]['蓝球']
                period = df.iloc[i]['期号'] if '期号' in df.columns else f"第{i+1}期"
                print(f"  {period}: 红球{red}, 蓝球{blue}")
            
            sequence = features_scaled.reshape(1, len(features_scaled), -1)
        else:
            if not os.path.exists(self.processed_data_file):
                raise FileNotFoundError(f"处理后的数据文件不存在")
            data = np.load(self.processed_data_file, allow_pickle=True)
            features_scaled = data['features_scaled']
            seq_length = int(data['seq_length'])
            print(f"使用processed_data.npz中的旧数据，数据量: {len(features_scaled)}")
            sequence = features_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # 预测
        predictions = self.model.predict(sequence, verbose=0)
        
        # 处理预测结果
        if self.use_classification:
            print("\n" + "=" * 60)
            print("预测概率信息:")
            print("=" * 60)
            for i in range(6):
                prob_dist = predictions[i][0]
                top5_indices = np.argsort(prob_dist)[-5:][::-1]
                top5_probs = prob_dist[top5_indices]
                print(f"  红球位置{i+1} 前5候选: ", end="")
                for idx, prob in zip(top5_indices, top5_probs):
                    print(f"{idx+1}({prob:.3f}) ", end="")
                print()
            
            # ========== 改进的红球选择策略 ==========
            def improved_red_ball_selection(predictions, temperature=2.5, top_k=15, seed=None):
                """
                改进的红球选择策略：使用加权采样增加多样性
                1. 合并所有位置的预测概率（使用加权平均）
                2. 使用温度采样增加随机性
                3. 从Top-K候选中采样，确保结果多样化
                """
                prob_dists = [predictions[i][0].copy() for i in range(6)]
                
                # 使用加权平均：不同位置给予不同权重
                # 位置越靠前，权重越大（因为通常前面的位置更稳定）
                position_weights = np.array([0.18, 0.18, 0.16, 0.16, 0.16, 0.16])
                position_weights = position_weights / position_weights.sum()  # 归一化
                
                # 计算加权平均概率
                combined_probs = np.zeros(33)
                for i in range(6):
                    combined_probs += prob_dists[i] * position_weights[i]
                
                # 同时考虑最大概率（增加多样性）
                max_probs = np.max(prob_dists, axis=0)
                
                # 混合策略：70%加权平均 + 30%最大概率
                final_probs = 0.7 * combined_probs + 0.3 * max_probs
                
                # 应用温度缩放，增加随机性
                scaled_probs = np.power(final_probs + 1e-10, 1.0 / temperature)
                scaled_probs = scaled_probs / scaled_probs.sum()
                
                # 从Top-K候选中采样
                top_k_indices = np.argsort(scaled_probs)[-top_k:][::-1]
                top_k_probs = scaled_probs[top_k_indices]
                top_k_probs = top_k_probs / top_k_probs.sum()
                
                # 采样6个不重复的号码
                red_balls = []
                remaining_probs = top_k_probs.copy()
                remaining_indices = top_k_indices.copy()
                
                rng = np.random.RandomState(seed) if seed is not None else np.random
                
                for i in range(6):
                    if len(remaining_indices) == 0:
                        # 如果候选用完了，从全部33个号码中随机选择
                        available = [x for x in range(1, 34) if x not in red_balls]
                        if available:
                            red_balls.append(rng.choice(available))
                        break
                    
                    # 从剩余候选中采样
                    selected_idx = rng.choice(len(remaining_indices), p=remaining_probs)
                    selected_ball = remaining_indices[selected_idx] + 1
                    
                    red_balls.append(selected_ball)
                    
                    # 移除已选择的号码
                    mask = remaining_indices != remaining_indices[selected_idx]
                    remaining_indices = remaining_indices[mask]
                    remaining_probs = remaining_probs[mask]
                    if len(remaining_probs) > 0:
                        remaining_probs = remaining_probs / remaining_probs.sum()
                
                # 确保有6个不重复的号码
                red_balls = sorted(list(set(red_balls)))
                
                # 如果不足6个，从所有号码中补充
                if len(red_balls) < 6:
                    available = [x for x in range(1, 34) if x not in red_balls]
                    if len(available) >= (6 - len(red_balls)):
                        # 从可用号码中按概率补充
                        available_probs = scaled_probs[np.array(available) - 1]
                        available_probs = available_probs / available_probs.sum()
                        
                        needed = 6 - len(red_balls)
                        selected = rng.choice(
                            available, 
                            size=min(needed, len(available)), 
                            replace=False, 
                            p=available_probs
                        )
                        red_balls.extend(selected.tolist())
                    else:
                        # 如果还是不够，随机补充
                        remaining = [x for x in range(1, 34) if x not in red_balls]
                        red_balls.extend(remaining[:6 - len(red_balls)])
                
                return sorted(red_balls[:6])
            
            # 使用改进的选择策略
            print("\n" + "=" * 60)
            print("红球选择策略:")
            print("=" * 60)
            print("  方法: 加权平均概率 + 温度采样")
            print("  温度: 2.5 (增加随机性和多样性)")
            print("  候选池: Top-15 号码")
            print("  特点: 避免等差数列，增加预测多样性")
            
            # 计算合并概率用于展示
            prob_dists = [predictions[i][0].copy() for i in range(6)]
            position_weights = np.array([0.18, 0.18, 0.16, 0.16, 0.16, 0.16])
            position_weights = position_weights / position_weights.sum()
            combined_probs = np.zeros(33)
            for i in range(6):
                combined_probs += prob_dists[i] * position_weights[i]
            max_probs = np.max(prob_dists, axis=0)
            final_probs = 0.7 * combined_probs + 0.3 * max_probs
            scaled_probs = np.power(final_probs + 1e-10, 1.0 / 2.5)
            scaled_probs = scaled_probs / scaled_probs.sum()
            
            # 显示Top-15候选号码
            top15_indices = np.argsort(scaled_probs)[-15:][::-1]
            top15_probs = scaled_probs[top15_indices]
            print(f"\n  合并后Top-15候选号码:")
            for i, (idx, prob) in enumerate(zip(top15_indices, top15_probs)):
                print(f"    {idx+1:2d}: {prob:.4f} ({prob*100:.2f}%)", end="  ")
                if (i + 1) % 3 == 0:
                    print()
            if len(top15_indices) % 3 != 0:
                print()
            
            red_balls = improved_red_ball_selection(predictions, temperature=2.5, top_k=15, seed=random_seed)
            
            # 显示最终预测结果
            print(f"\n  最终预测的红球: {red_balls}")
            
            # 检查是否有规律性
            intervals = [red_balls[i+1] - red_balls[i] for i in range(5)]
            if len(set(intervals)) == 1:
                print(f"  警告: 检测到等差数列 (间隔={intervals[0]})")
            else:
                print(f"  号码间隔: {intervals} (无规律)")
            
            # 蓝球预测：使用改进的策略（避免总是预测同一个号码）
            blue_prob_dist = predictions[6][0]
            
            print(f"\n蓝球预测信息:")
            top5_indices = np.argsort(blue_prob_dist)[-5:][::-1]
            top5_probs = blue_prob_dist[top5_indices]
            print(f"  前5候选: ", end="")
            for idx, prob in zip(top5_indices, top5_probs):
                print(f"{idx+1}({prob:.3f}) ", end="")
            print()
            
            # 检查概率分布
            max_prob = np.max(blue_prob_dist)
            entropy = -np.sum(blue_prob_dist * np.log(blue_prob_dist + 1e-10))
            max_entropy = np.log(16)
            entropy_ratio = entropy / max_entropy
            
            print(f"  概率分布熵: {entropy:.4f} (比例: {entropy_ratio:.2%})")
            print(f"  最高概率: {max_prob:.4f} ({max_prob*100:.2f}%)")
            
            # 改进策略：使用更强的温度采样，增加多样性
            # 动态调整温度：如果概率分布过于集中，使用更高的温度
            if entropy_ratio < 0.4 or max_prob > 0.2:
                # 概率分布过于集中，使用更高的温度
                temperature = 3.5
                top_k = 8
                strategy_name = "高温度采样（概率分布过于集中）"
            elif entropy_ratio < 0.6 or max_prob > 0.15:
                # 中等集中度，使用中等温度
                temperature = 2.5
                top_k = 6
                strategy_name = "中等温度采样"
            else:
                # 概率分布较分散，使用较低温度但仍有随机性
                temperature = 2.0
                top_k = 5
                strategy_name = "温度采样（概率分布较分散）"
            
            # 温度采样：使用温度参数调整概率分布
            scaled_probs = np.power(blue_prob_dist + 1e-10, 1.0 / temperature)
            scaled_probs = scaled_probs / scaled_probs.sum()
            
            # 从Top-K中采样（避免选择概率太低的号码）
            top_k_indices_full = np.argsort(scaled_probs)[-top_k:][::-1]
            top_k_scaled_probs = scaled_probs[top_k_indices_full]
            top_k_scaled_probs = top_k_scaled_probs / top_k_scaled_probs.sum()
            
            if random_seed is not None:
                np.random.seed(random_seed + 200)
            blue_ball = np.random.choice(top_k_indices_full + 1, p=top_k_scaled_probs)
            print(f"  策略: Top-{top_k}温度采样 (温度={temperature}, {strategy_name})")
            
            print(f"  最终预测: {blue_ball} (原始概率: {blue_prob_dist[blue_ball-1]:.3f}, 缩放后概率: {top_k_scaled_probs[np.where(top_k_indices_full == blue_ball-1)[0][0]]:.3f})")
            
            # 返回单个结果
            return {
                '红球': red_balls,
                '蓝球': blue_ball
            }
        else:
            # 回归模式
            prediction_scaled = predictions[0] if isinstance(predictions, list) else predictions
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            dummy_features = np.zeros((1, scaler.n_features_in_))
            dummy_features[0, :7] = prediction_scaled[0]
            prediction_original = scaler.inverse_transform(dummy_features)[0]
            red_balls = np.clip(np.round(prediction_original[:6]), 1, 33).astype(int)
            red_balls = sorted(np.unique(red_balls))
            while len(red_balls) < 6:
                available = [x for x in range(1, 34) if x not in red_balls]
                if available:
                    red_balls.append(available[0])
                else:
                    break
            red_balls = sorted(red_balls[:6])
            blue_min = scaler.data_min_[6]
            blue_max = scaler.data_max_[6]
            blue_ball_scaled = prediction_scaled[0][6]
            blue_ball_original = blue_ball_scaled * (blue_max - blue_min) + blue_min
            blue_ball = int(np.round(np.clip(blue_ball_original, 1, 16)))
        
        return {
            '红球': red_balls,
            '蓝球': blue_ball
        }


if __name__ == "__main__":
    model = SSQLSTMModel50()
    
    # 训练模型
    print("开始训练（使用50期数据）...")
    model.train(epochs=200, batch_size=32)
    
    # 预测下一期
    print("\n预测下一期号码:")
    prediction = model.predict_next()
    print(f"红球: {prediction['红球']}")
    print(f"蓝球: {prediction['蓝球']}")

