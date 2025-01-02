import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# 生成随机数据集
def generate_random_dataset(num_users=100, num_items=200, density=0.05):
    num_interactions = int(num_users * num_items * density)
    user_ids = np.random.randint(0, num_users, size=num_interactions)
    item_ids = np.random.randint(0, num_items, size=num_interactions)
    
    # 确保 item_ids 不超出范围
    item_ids = item_ids % num_items  # 保证 item_ids 在 [0, num_items-1] 之间

    interactions = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'interaction': 1
    })

    interactions = interactions.groupby(['user_id', 'item_id'], as_index=False).sum()
    return interactions

# 生成数据集
random_dataset = generate_random_dataset()
num_users = random_dataset['user_id'].nunique()
num_items = random_dataset['item_id'].nunique()
print(f"Number of users: {num_users}, Number of items: {num_items}")
print(random_dataset.head())

# 将交互数据转换为用户-项目矩阵
user_item_matrix = random_dataset.pivot(index='user_id', columns='item_id', values='interaction').fillna(0)

# 矩阵分解类
class MatrixFactorization:
    def __init__(self, num_users, num_items, latent_dim):
        # 确保 user_embeddings 和 item_embeddings 的维度正确
        self.user_embeddings = np.random.rand(num_users, latent_dim)
        self.item_embeddings = np.random.rand(num_items, latent_dim)

    def predict(self, user_id, item_id):
        # 确保用户和项目ID在合理范围内
        if user_id >= len(self.user_embeddings) or item_id >= len(self.item_embeddings):
            raise IndexError(f"user_id {user_id} or item_id {item_id} out of bounds.")
        return np.dot(self.user_embeddings[user_id], self.item_embeddings[item_id])

    def train(self, interactions, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            for (user_id, item_id) in interactions:
                if user_id >= len(self.user_embeddings) or item_id >= len(self.item_embeddings):
                    continue  # 跳过超出范围的ID
                prediction = self.predict(user_id, item_id)
                error = 1 - prediction  # 所有交互目标为1
                self.user_embeddings[user_id] += learning_rate * error * self.item_embeddings[item_id]
                self.item_embeddings[item_id] += learning_rate * error * self.user_embeddings[user_id]

# 定义负采样类
class AugmentedNegativeSampling:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def generate_negative_samples(self, positive_samples):
        negative_samples = []
        for pos in positive_samples:
            new_negative_sample = pos + np.random.normal(0, 0.1)  # 简单示例
            negative_samples.append(int(new_negative_sample % len(self.item_embeddings)))  # 确保索引有效
        return negative_samples

class HardNegativeSampling:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def generate_negative_samples(self, positive_samples):
        return [max(positive_samples) % len(self.item_embeddings)]  # 示例逻辑，确保索引有效

class RandomNegativeSampling:
    def __init__(self, num_items):
        self.num_items = num_items

    def generate_negative_samples(self):
        # 返回负样本的列表
        return np.random.randint(0, self.num_items, size=10)  # 假设生成10个负样本

# 实验设置
latent_dim = 64
mf_model_ans = MatrixFactorization(num_users=num_users, num_items=num_items, latent_dim=latent_dim)
mf_model_hns = MatrixFactorization(num_users=num_users, num_items=num_items, latent_dim=latent_dim)
mf_model_rns = MatrixFactorization(num_users=num_users, num_items=num_items, latent_dim=latent_dim)

interactions_list = list(zip(random_dataset['user_id'], random_dataset['item_id']))

# 模型训练
mf_model_ans.train(interactions_list)
mf_model_hns.train(interactions_list)
mf_model_rns.train(interactions_list)

# 模拟预测并计算评估指标
positive_samples = list(range(num_items))  # 假设所有项目都是正样本
ans_sampler = AugmentedNegativeSampling(mf_model_ans.item_embeddings)
hns_sampler = HardNegativeSampling(mf_model_hns.item_embeddings)
rns_sampler = RandomNegativeSampling(num_items)

# 生成负样本
ans_neg_samples = ans_sampler.generate_negative_samples(positive_samples)
hns_neg_samples = hns_sampler.generate_negative_samples(positive_samples)
rns_neg_samples = rns_sampler.generate_negative_samples()

# 定义评估函数
def evaluate_model(true_samples, negative_samples):
    # 假设所有正样本都为1
    y_true = np.ones(len(true_samples))
    y_pred = np.zeros(len(true_samples))  # 初始预测值为负样本
    
    # 假设负样本的预测为0
    for i in range(len(true_samples)):
        if true_samples[i] in negative_samples:
            y_pred[i] = 0  # 负样本
        else:
            y_pred[i] = 1  # 正样本
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# 评估各个负采样方法
ans_precision, ans_recall, ans_f1 = evaluate_model(positive_samples, ans_neg_samples)
hns_precision, hns_recall, hns_f1 = evaluate_model(positive_samples, hns_neg_samples)
rns_precision, rns_recall, rns_f1 = evaluate_model(positive_samples, rns_neg_samples)

# 输出对比结果
print(f"ANS Precision: {ans_precision:.4f}, Recall: {ans_recall:.4f}, F1-score: {ans_f1:.4f}")
print(f"HNS Precision: {hns_precision:.4f}, Recall: {hns_recall:.4f}, F1-score: {hns_f1:.4f}")
print(f"RNS Precision: {rns_precision:.4f}, Recall: {rns_recall:.4f}, F1-score: {rns_f1:.4f}")
