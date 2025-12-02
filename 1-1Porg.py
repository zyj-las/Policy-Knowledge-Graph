import pandas as pd

# 读取Excel文件
file_path = 'data/policy_base.xlsx'
df = pd.read_excel(file_path)

# 提取“政策主体”列中的机构名称，并去重
df['政策主体'] = df['政策主体'].str.split('、')  # 按“、”分隔
df = df.explode('政策主体')  # 将多个机构分成多行
df['政策主体'] = df['政策主体'].str.strip()  # 去除空格
df = df.drop_duplicates(subset=['政策主体'])  # 去重

# 定义机构类型的分类规则
def classify_institution(institution):
    if institution in ["中国科学院", "中国社会科学院", "中国中医科学院", "中国工程院", "中央党校",\
                         "国家自然科学基金委员会","国家科学技术奖励工作办公室","国务院发展研究中心"]:
        return "科研管理机构"
    elif any(keyword in institution for keyword in ["全国人民代表大会","全国代表大会","中共", "中央", "国务院","国家"]) \
        or institution in ["最高人民法院", "最高人民检察院","中国人民银行"] \
            or institution.endswith(("部","局","署","委员会")):
        return "党政机关"
    elif institution.endswith(("所","中心","大学","公司","企业","研究院")):
        return "责任主体单位"
    elif institution.endswith(("协会","学会","工会","联合会","研究会")):
        return "科技/学术团体"
    else:
        return "其他"

# 为每个机构分配类别
df['主体类型'] = df['政策主体'].apply(classify_institution)

# 为每个类型的机构生成唯一的ID
id_counter = {
    '党政机关': 1,
    '科研管理机构': 1,
    '责任主体单位': 1,
    '科技/学术团体': 1
}

def generate_id(row):
    type_mapping = {
        '党政机关': 'o1-1-',
        '科研管理机构': 'o1-2-',
        '责任主体单位': 'o2-3-',
        '科技/学术团体': 'o1-4-'
    }
    
    inst_type = row['主体类型']
    if inst_type in type_mapping:
        id_str = f"{type_mapping[inst_type]}{id_counter[inst_type]:03d}"
        id_counter[inst_type] += 1
        return id_str
    else:
        return None

# 生成政策主体ID
df['政策主体ID'] = df.apply(generate_id, axis=1)

# 保存结果到新的Excel文件
output_file_path = 'org-result.xlsx'
df.to_excel(output_file_path, index=False, columns=['政策主体ID', '政策主体', '主体类型'])

print("文件已成功保存到:", output_file_path)