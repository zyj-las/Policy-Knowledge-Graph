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



# 读取Excel数据
train_data = pd.read_csv('data/train_dic.csv')
train_texts = train_data['政策条文'].tolist()
train_labels = train_data['政策目标'].apply(lambda x: x.split(',')).tolist()#可替换为相应的要素
# train_labels = train_data['政策措施'].apply(lambda x: x.split(',')).tolist()
# train_labels = train_data['政策工具'].apply(lambda x: x.split(',')).tolist()
# train_labels = train_data['作用行为'].apply(lambda x: x.split(',')).tolist()
# 标签编码
mlb = MultiLabelBinarizer(classes=['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])
train_labels_encoded = mlb.fit_transform(train_labels)
label_classes = mlb.classes_
# 创建增强器
syn_aug = naw.SynonymAug(aug_src='wordnet')  # 使用WordNet同义词替换
# 增加训练数据
augmented_texts = [syn_aug.augment(text) for text in train_texts]
train_texts.extend(augmented_texts)
train_labels_encoded = np.concatenate([train_labels_encoded, train_labels_encoded])
# 目标标签解释示例
label_descriptions = [
    "制度规则建设是实现科学规范、激励有效、惩处有力的科研诚信制度规则",
    "工作机制建设是实现职责清晰、协调有序、监管到位的科研诚信工作机制",
    "信息系统建设是实现覆盖全面、共享联动、动态管理的科研诚信信息系统",
    "意识精神建设是实现科研人员诚信意识显著增强与科学精神得到大力弘扬,弘扬科学精神、恪守诚信规范成为科技界共同理念和自觉行动",
    "社会生态建设是全社会的诚信基础和创新生态持续巩固发展"
]
# # 措施标签解释
# label_descriptions = [
#     "宣传教育是通过各种渠道和方式向科研人员普及科研诚信的重要性、原则和规范以及案件等相关信息；通过培训、教育活动等方式，加强科研人员的科研诚信意识、提升对科学规范认识",
#     "监举调查包括对失信行为的监督发现、调查与审查等，监督包括以诚信为基础的科技监督评估体系，发现包括举报（涉及举报流程、举报受理等）、监督机构自行发现、媒体曝光等；审查是对失信行为的审核与审查，包括对失信行为的认定、核实等；调查涉及对涉嫌失信行为的调查与取证等",
#     "行为惩处是对失信行为的惩罚与纠正措施以及处理的原则、标准、程序、结果等",
#     "救济保护是对受到不当对待或损害的科研人员提供的救济和保护措施，涉及对失信行为举报者、当事人等合法权益的保护；对举报中非失信行为的处理；科研人员的申诉、复查等",
#     "权责划分是对相关责任人、机构等的权力分配与责任划分，还包含工作职责、工作规章、工作流程等",
#     "标准界定是对科研不端、科研失信与相关行为的定义、标准、范畴、边界等进行划分"
# ]
# # 工具标签解释
# label_descriptions = [
#     "命令是管理个人或机构行为的规范以及违反准则的惩罚，从而保持行为的一致性，要求必须服从，具有强制性特点，例如学术道德规范，学术不端行为的调查规定、处理原则与处分等，常包含的关键词：不得、不准、应、应当、必须、惩戒、严厉打击、严禁、撤销、批评等强制性词语", 
#     "激励是通过物质条件诱导目标群体快速做出期待的行为，包括补助、拨款、奖励、福利、提升待遇等，例如对优秀集体和个人的宣传、表彰和奖励，失范行为举报奖励等，常包含的关键词：激励、表彰、表扬、奖励、补助等",
#     "劝告是以象征、呼吁、劝导等进行价值和信念传播，鼓励目标群体依据政策目标自觉调整行为，具有自愿性特点，例如对学术道德行为的鼓励与弘扬、对学术不端行为的调查处理建议等，常包含的关键词：引导、呼吁、宣传、建议、可根据、积极、倡导、弘扬、可以等",
#     "能力建设是通过资金转移来投资未来的物质、知识或人力资源建设，如教育、培训等方式提升个体或机构素质和能力，以更好地适应和执行相关政策规定，例如教师岗位职业培训、师德专题教育，学生学术道德与规范类课程培训，科研诚信学术道德专项投入等为提升个人或机构的学术道德建设能力进行的资源投入，常包含的关键词：教育、培训、经费等",
#     "权威重组是改变实施政策的体制结构，实现个人或机构之间的权力转移，例如弱化或取消个体或机构权力，形成新的职能机构，目的在于提高政策实施效率，例如科研诚信建设相关委员会或领导小组的建立或变革、人员组成设置、权力划分以及相关学术制度建立等，常包含的关键词：成立、设立、建立、出台、制定、负责、组织等"
# ]
# # 行为标签解释
# label_descriptions = [
#     "常规科研行为是针对科研人员从事科研活动中的各类合法、合规行为" ,
#     "科研不端行为是针对科研人员在科研活动中违反科研诚信原则的行为" ,
#     "研究监管行为是针对科研机构和有关部门对科研活动进行监督管理工作" ,
#     "举报调查行为是针对任何单位或个人对发现的失信行为进行举报的行为"
# ]

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
num_epochs = 50  # 增加训练轮次
best_val_f1_macro = 0
patience = 3 
patience_counter = 0
# 优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)  # 降低学习率
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)  # 学习率调度器
# label_weights = torch.tensor([10, 15, 1, 8, 7]).to(device) 
loss_fn = torch.nn.BCEWithLogitsLoss()  


# 记录训练过程
log_file_path = 'modles/PTinyBert_goal.txt'
# log_file_path = 'modles/PTinyBert_measure.txt'
# log_file_path = 'modles/PTinyBert_instrument.txt'
# log_file_path = 'modles/PTinyBert_actbe.txt'
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
    
    # 在第5个epoch时，解冻BERT的最后两层以进行微调
    if epoch == 5:
        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        print(f"Epoch {epoch + 1}: Unfreezing the last 2 layers of BERT for fine-tuning.")
    
    # 训练阶段
    train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
     # 验证阶段
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
plt.savefig('outputs/PTinyBert_goal.png')
# plt.savefig('modles/PTinyBert_measure.png')
# plt.savefig('modles/PTinyBert_intrument.png')
# plt.savefig('modles/PTinyBert_actbe.png')
#plt.show()
print(f"曲线图已保存")

# 保存模型权重
model_save_path = 'models/PTinyBert_goal.pth'
# model_save_path = 'modles/PTinyBert_measure.pth'
# model_save_path = 'modles/PTinyBert_instrument.pth'
# model_save_path = 'modles/PTinyBert_actbe.pth'
# 保存模型、优化器、调度器和标签处理器
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'mlb': mlb  # 保存MultiLabelBinarizer对象
}, model_save_path)

print(f"模型已保存到 {model_save_path}")