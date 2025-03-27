import torch
import pandas as pd
from transformers import AutoTokenizer, DistilBertModel
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
train_data = pd.read_excel('D:\\File_zyj\\2.论文文件\\4.小论文_政策知识图谱\\数据分析文件0929\\政策工具-训练.xlsx')

train_texts = train_data['政策段落'].tolist()
train_labels = train_data['政策工具'].apply(lambda x: x.split(',')).tolist()

# 标签编码
mlb = MultiLabelBinarizer(classes=['i1', 'i2', 'i3', 'i4', 'i5'])
train_labels_encoded = mlb.fit_transform(train_labels)
label_classes = mlb.classes_
# 创建增强器
syn_aug = naw.SynonymAug(aug_src='wordnet')  # 使用WordNet同义词替换
# 增加训练数据
augmented_texts = [syn_aug.augment(text) for text in train_texts]
train_texts.extend(augmented_texts)
train_labels_encoded = np.concatenate([train_labels_encoded, train_labels_encoded])

# 标签解释
label_descriptions = [
    "命令是管理个人或机构行为的规范以及违反准则的惩罚，从而保持行为的一致性，要求必须服从，具有强制性特点，例如学术道德规范，学术不端行为的调查规定、处理原则与处分等，常包含的关键词：不得、不准、应、应当、必须、惩戒、严厉打击、严禁、撤销、批评等强制性词语", 
    "激励是通过物质条件诱导目标群体快速做出期待的行为，包括补助、拨款、奖励、福利、提升待遇等，例如对优秀集体和个人的宣传、表彰和奖励，失范行为举报奖励等，常包含的关键词：激励、表彰、表扬、奖励、补助等",
    "劝告是以象征、呼吁、劝导等进行价值和信念传播，鼓励目标群体依据政策目标自觉调整行为，具有自愿性特点，例如对学术道德行为的鼓励与弘扬、对学术不端行为的调查处理建议等，常包含的关键词：引导、呼吁、宣传、建议、可根据、积极、倡导、弘扬、可以等",
    "能力建设是通过资金转移来投资未来的物质、知识或人力资源建设，如教育、培训等方式提升个体或机构素质和能力，以更好地适应和执行相关政策规定，例如教师岗位职业培训、师德专题教育，学生学术道德与规范类课程培训，科研诚信学术道德专项投入等为提升个人或机构的学术道德建设能力进行的资源投入，常包含的关键词：教育、培训、经费等",
    "权威重组是改变实施政策的体制结构，实现个人或机构之间的权力转移，例如弱化或取消个体或机构权力，形成新的职能机构，目的在于提高政策实施效率，例如科研诚信建设相关委员会或领导小组的建立或变革、人员组成设置、权力划分以及相关学术制度建立等，常包含的关键词：成立、设立、建立、出台、制定、负责、组织等"
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
        )

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# 创建数据集
# 加载 BERT-mini 的 tokenizer 和 模型
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model_name="distilbert-base-uncased"

# 划分训练集和验证集，80% 作为训练集，20% 作为验证集
train_texts, val_texts, train_labels_encoded, val_labels_encoded = train_test_split(
    train_texts, train_labels_encoded, test_size=0.2, random_state=42
)

train_dataset = PolicyDataset(train_texts, train_labels_encoded, tokenizer)
val_dataset = PolicyDataset(val_texts, val_labels_encoded, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义模型
class BertForMultiLabelClassification(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)  # 加载 DistilBERT 模型
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # DistilBERT 不提供 pooler_output，使用最后一层隐藏状态的第一个 token ([CLS])
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 选择 [CLS] token 的输出
        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)
        return logits

    
# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForMultiLabelClassification(model_name, num_labels=len(label_classes))
model = model.to(device)

# 优化器和损失函数
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)  # 降低学习率
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)  # 学习率调度器
# 根据标签频率计算权重
label_weights = torch.tensor([10, 15, 1, 8, 7]).to(device)  # 这里的权重根据实际情况设置
loss_fn = torch.nn.BCEWithLogitsLoss()  # 可以根据类别权重调整此损失函数
# 记录文件路径
log_file_path = 'D:\\Experiment_zyj\\Python_exp\\policy_map\\instrument_distilbert_迭代过程.txt'
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
'''
# 在训练和验证循环之前，初始化最佳验证损失
best_val_loss = float('inf')  # 初始化为正无穷
bestmodel_save_path = 'D:\\Experiment_zyj\\Python_exp\\policy_map\\instrument_distilbert_model_best.pth'  # 最佳模型保存路径
'''
# 训练和验证循环
num_epochs = 30  # 增加训练轮次
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc, val_f1_micro, val_acc_label, val_acc_labelAve, val_f1_macro = eval_model(model, val_loader, loss_fn, device)
    
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
    '''
    # 检查验证损失是否是最佳的
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #保存最佳模型的权重
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mlb': mlb 
        }, bestmodel_save_path)
        print(f"保存了最佳模型权重，当前最佳验证损失: {best_val_loss:.4f}")
        '''
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
plt.savefig('D:\\Experiment_zyj\\Python_exp\\policy_map\\instrument_distilbert_曲线图.png')
#plt.show()
print(f"曲线图已保存")

# 保存最终模型权重
model_save_path = 'D:\\Experiment_zyj\\Python_exp\\policy_map\\instrument_distilbert_model.pth'

# 保存模型、优化器、调度器和标签处理器
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'mlb': mlb  # 保存MultiLabelBinarizer对象
}, model_save_path)

print(f"政策工具模型已保存到 {model_save_path}")
