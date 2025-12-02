import torch
import pandas as pd
from transformers import AutoTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# 定义数据集类
class PolicyDataset(Dataset):
    def __init__(self, texts, tokenizer=None, max_len=128):
        self.texts = texts
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
        return item

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

# 加载保存的模型和MultiLabelBinarizer
model_save_path = 'models/PDistilBert_goal.pth'
# model_save_path = 'modles/PDistilBert_measure.pth'
# model_save_path = 'modles/PDistilBert_instrument.pth'
# model_save_path = 'modles/PDistilBert_actbe.pth'
checkpoint = torch.load(model_save_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_labels = len(checkpoint['mlb'].classes_)
model = BertForMultiLabelClassification('distilbert-base-uncased', num_labels)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
mlb = checkpoint['mlb']
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# 读取测试数据
test_data = pd.read_csv('data/paragraph.csv')
test_texts = test_data['政策条文'].tolist()

# 创建测试数据集和DataLoader
test_dataset = PolicyDataset(test_texts, tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 预测函数
def predict(model, data_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            all_preds.append(outputs.sigmoid().cpu().numpy())
    all_preds = np.vstack(all_preds)
    return (all_preds > threshold).astype(int)

# 调用预测函数进行预测
preds = predict(model, test_loader, device, threshold=0.5)
pred_labels = mlb.inverse_transform(preds)
# 保存预测结果到Excel文件
output_path = 'data/PDistilBert_goal.xlsx'
# output_path = 'data/PDistilBert_measure.xlsx'
# output_path = 'data/PDistilBert_instrument.xlsx'
# output_path = 'data/PDistilBert_actbe.xlsx'
test_data.to_excel(output_path, index=False)
print(f"结果已保存到 {output_path}")