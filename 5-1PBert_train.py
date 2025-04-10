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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nlpaug.augmenter.word as naw

# 读取数据
train_data = pd.read_excel('D:\\政策要素_train.xlsx')
train_texts = train_data['政策段落'].tolist()
train_labels = train_data['政策要素'].apply(lambda x: x.split(',')).tolist()#可替换为相应的要素

# 标签编码
mlb = MultiLabelBinarizer(classes=['e1', 'e2', 'e3', 'e4', 'e5'])
train_labels_encoded = mlb.fit_transform(train_labels)
label_classes = mlb.classes_
# 创建增强器
syn_aug = naw.SynonymAug(aug_src='wordnet')  # 使用WordNet同义词替换
# 增加训练数据
augmented_texts = [syn_aug.augment(text) for text in train_texts]
train_texts.extend(augmented_texts)
train_labels_encoded = np.concatenate([train_labels_encoded, train_labels_encoded]) # type: ignore
# 标签解释示例
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
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 划分训练集和验证集，80% 作为训练集，20% 作为验证集
train_texts, val_texts, train_labels_encoded, val_labels_encoded = train_test_split(
    train_texts, train_labels_encoded, test_size=0.2, random_state=42
)

train_dataset = PolicyDataset(train_texts, train_labels_encoded, tokenizer)
val_dataset = PolicyDataset(val_texts, val_labels_encoded, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义模型，增加dropout层，增加正则化
class BertForMultiLabelClassification(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)  # 添加dropout
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)  # 应用dropout
        logits = self.classifier(pooled_output)
        return logits
    
# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForMultiLabelClassification('bert-base-chinese', len(label_classes))
model = model.to(device)

# 优化器和损失函数
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2) 
scheduler = ReduceLROnPlateau(optimizer,mode='max', patience=2, factor=0.2, threshold=0.01)
#label_weights = torch.tensor([10, 15, 1, 8, 7]).to(device) 
#loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights)
loss_fn = torch.nn.BCEWithLogitsLoss() 

# 记录训练过程
log_file_path = 'D:\\PBert_element.txt'
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
    
    avg_loss = total_loss / len(data_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
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
num_epochs = 15
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc, val_f1_micro, val_acc_label, val_acc_labelAve, val_f1_macro = eval_model(model, val_loader, loss_fn, device)
    
    # 打印当前学习率
    current_lr = scheduler.get_last_lr()
    lr_msg = f"Current learning rate: {current_lr}\n"
    print(lr_msg)
    log_file.write(lr_msg) 

    # 记录损失和准确率
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_accuracies_labelAve.append(val_acc_labelAve)
    log_msg = (f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
               f"Val Acc: {val_acc:.4f}, Val F1 Micro: {val_f1_micro:.4f}, Val F1 Macro: {val_f1_macro:.4f}\n"
               f"Label-wise Accuracies: {val_acc_label}\n"
               f"Average Label Accuracy: {val_acc_labelAve:.4f}\n")
    print(log_msg)
    log_file.write(log_msg)
    scheduler.step(val_loss)# 调整学习率
log_file.close()

# 绘制Loss曲线和Acc曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_accuracies_labelAve, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('D:\\PBert_element.png')
#plt.show()
#print(f"曲线图已保存")

# 保存模型
model_save_path = 'D:\\PBert_element.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'mlb': mlb  # 保存MultiLabelBinarizer对象
}, model_save_path)
#print(f"模型已保存到 {model_save_path}")