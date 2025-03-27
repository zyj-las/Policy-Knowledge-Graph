import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau # 增加学习率调度器
import nlpaug.augmenter.word as naw



# 读取Excel数据
train_data = pd.read_excel('D:\\File_zyj\\2.论文文件\\4.小论文_政策知识图谱\\数据分析文件0929\\政策目标-训练.xlsx')

train_texts = train_data['政策段落'].tolist()
train_labels = train_data['政策目标'].apply(lambda x: x.split(',')).tolist()

# 标签编码
mlb = MultiLabelBinarizer(classes=['g1', 'g2', 'g3', 'g4', 'g5'])
train_labels_encoded = mlb.fit_transform(train_labels)
label_classes = mlb.classes_
# 创建增强器
syn_aug = naw.SynonymAug(aug_src='wordnet')  # 使用WordNet同义词替换
# 增加训练数据
augmented_texts = [syn_aug.augment(text) for text in train_texts]
train_texts.extend(augmented_texts)
train_labels_encoded = np.concatenate([train_labels_encoded, train_labels_encoded]) # type: ignore
# 标签解释
label_descriptions = [
    "制度规则建设是实现科学规范、激励有效、惩处有力的科研诚信制度规则",
    "工作机制建设是实现职责清晰、协调有序、监管到位的科研诚信工作机制",
    "信息系统建设是实现覆盖全面、共享联动、动态管理的科研诚信信息系统",
    "意识精神建设是实现科研人员诚信意识显著增强与科学精神得到大力弘扬,弘扬科学精神、恪守诚信规范成为科技界共同理念和自觉行动",
    "社会生态建设是全社会的诚信基础和创新生态持续巩固发展"
]

class PolicyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        ) # type: ignore

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# 创建数据集
# 加载 tinybert 的 tokenizer 和 模型
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model_name = 'huawei-noah/TinyBERT_General_4L_312D'  # 使用tinybert

# 划分训练集和验证集，80% 作为训练集，20% 作为验证集
train_texts, val_texts, train_labels_encoded, val_labels_encoded = train_test_split(
    train_texts, train_labels_encoded, test_size=0.2, random_state=42
)

train_dataset = PolicyDataset(train_texts, train_labels_encoded, tokenizer)
val_dataset = PolicyDataset(val_texts, val_labels_encoded, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义模型 (替换为 tinybert)
class BertForMultiLabelClassification(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)  # 使用tinybert
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits
    
# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForMultiLabelClassification(model_name, num_labels=len(label_classes))
model = model.to(device)
# 训练参数
num_epochs = 30  # 增加训练轮次
# 在验证过程中跟踪最佳 Val F1 Macro
best_val_f1_macro = 0
patience = 3  # 如果 F1 Macro 没有提升的容忍度
patience_counter = 0
# 优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)  # 降低学习率
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)  # 学习率调度器
# 根据标签频率计算权重
label_weights = torch.tensor([10, 15, 1, 8, 7]).to(device)  # 这里的权重根据实际情况设置
loss_fn = torch.nn.BCEWithLogitsLoss()  # 可以根据类别权重调整此损失函数


# 记录文件路径
log_file_path = 'D:\\Experiment_zyj\\Python_exp\\policy_map\\goal_tinybert_迭代过程.txt'
log_file = open(log_file_path, 'w')

# 保存结果并绘制曲线的变量
train_losses = []
val_losses = []
val_accuracies = []
val_accuracies_labelAve=[]

# 模型训练和评估函数
def train_model(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # 记录预测值和真实标签
            all_preds.append(outputs.sigmoid().cpu().numpy())  # 使用sigmoid输出预测概率
            all_labels.append(labels.cpu().numpy())
    
    # 将所有批次的预测结果和标签连接起来
    avg_loss = total_loss / len(data_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 将预测值转换为0或1的二进制格式
    preds_binary = (all_preds > 0.5).astype(int)
    
    # 计算样本级别的整体准确率和微平均F1分数
    val_acc = accuracy_score(all_labels, preds_binary)
    val_f1_micro = f1_score(all_labels, preds_binary, average='micro')
    
    # 计算标签级别的准确率（逐个标签和总标签的准确率）和宏平均F1分数
    val_acc_label = (preds_binary == all_labels).mean(axis=0)  # 逐个标签的平均准确率
    val_acc_labelAve = val_acc_label.mean()  # 所有标签准确率的平均值
    val_f1_macro = f1_score(all_labels, preds_binary, average='macro')

    return avg_loss, val_acc, val_f1_micro, val_acc_label, val_acc_labelAve, val_f1_macro

# 训练和验证循环

for epoch in range(num_epochs):
    '''
    # 在第5个epoch时，解冻BERT的最后两层以进行微调
    if epoch == 5:
        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        print(f"Epoch {epoch + 1}: Unfreezing the last 2 layers of BERT for fine-tuning.")
    '''
    # 训练阶段
    train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
     # 验证阶段
    val_loss, val_acc, val_f1_micro, val_acc_label, val_acc_labelAve, val_f1_macro = eval_model(model, val_loader, loss_fn, device)
    '''
    # 检查 Val F1 Macro 是否提升
    if val_f1_macro > best_val_f1_macro:
        best_val_f1_macro = val_f1_macro
        patience_counter = 0  # 重置计数器
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
    '''
    # 打印当前学习率
    current_lr = scheduler.get_last_lr()
    lr_msg = f"Current learning rate: {current_lr}\n"
    print(lr_msg)
    log_file.write(lr_msg)  # 保存当前学习率到日志文件

    # 记录损失和准确率
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_accuracies_labelAve.append(val_acc_labelAve)
    
    # 打印并保存结果到文件
    log_msg = (f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
               f"Val Acc: {val_acc:.4f}, Val F1 Micro: {val_f1_micro:.4f}, Val F1 Macro: {val_f1_macro:.4f}\n"
               f"Label-wise Accuracies: {val_acc_label}\n"
               f"Average Label Accuracy: {val_acc_labelAve:.4f}\n")
    
    print(log_msg)
    log_file.write(log_msg)

    # 调整学习率
    scheduler.step(val_loss)

# 关闭文件
log_file.close()

# 绘制Loss曲线和Acc曲线
plt.figure(figsize=(12, 5))

# 绘制Loss曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制Acc曲线
plt.subplot(1, 2, 2)
plt.plot(val_accuracies_labelAve, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 保存图像
plt.savefig('D:\\Experiment_zyj\\Python_exp\\policy_map\\goal_tinybert_曲线图.png')
#plt.show()
print(f"曲线图已保存")

# 保存模型权重
model_save_path = 'D:\\Experiment_zyj\\Python_exp\\policy_map\\goal_tinybert_model.pth'

# 保存模型、优化器、调度器和标签处理器
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'mlb': mlb  # 保存MultiLabelBinarizer对象
}, model_save_path)

print(f"政策目标模型已保存到 {model_save_path}")