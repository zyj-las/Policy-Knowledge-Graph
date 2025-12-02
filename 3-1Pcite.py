import os
import re
import pandas as pd

# 清洗政策标题
def clean_policy_title(title):
    title = re.sub(r'关于印发(.+)的通知', r'\1', title)
    title = title.replace(' ', '').replace('、', '')
    department_patterns = [
        r'办公厅', r'中共中央', r'国务院', r'教育部', r'国家发改委', r'国家科教委',
        r'人民银行', r'科技部', r'财政部', r'自然科学基金委办公室', r'印发〈',
        r'〉的通知', r'转发<', r'>的通知', r'转发〈', r'印发', r'国家发展改革委',
    ]
    for pattern in department_patterns:
        title = re.sub(pattern, '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# 提取政策标题
def extract_policy_titles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    policy_title = lines[0].strip()# 提取主政策标题，第一行是政策标题
    referenced_policies = set()# 去重
    excluded_titles = set()
    exclude_patterns = [
        r'表', r'公报', r'书', r'要报', r'索引', r'综述', r'报告'
    ]# 排除
    for line in lines:
        # 查找书名号中的内容
        matches = re.findall(r'《([^》]+)》', line)
        for match in matches:
            # 检查标题前是否有“简称”或“本”
            if re.search(r'(简称|本)\s*《[^》]*' + re.escape(match) + r'》', line):
                excluded_titles.add(match.strip())
            else:
                # 检查标题是否匹配排除模式
                if any(re.search(pattern, match) for pattern in exclude_patterns):
                    excluded_titles.add(match.strip())
                else:
                    cleaned_title = clean_policy_title(match.strip())
                    # 去除标题是两个字的条目
                    if len(cleaned_title) > 2:
                        referenced_policies.add(cleaned_title)
    # 过滤掉排除的标题
    filtered_policies = {policy for policy in referenced_policies if policy not in excluded_titles}
    return policy_title, filtered_policies

# 处理政策文件并生成结果
def process_policies(folder_path, output_file, policy_id_file):
    # 读取政策ID和标题对应表
    policy_data = pd.read_excel(policy_id_file)
    policy_data['政策标题'] = policy_data['政策标题'].apply(clean_policy_title)
    all_data = []

    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            policy_title, referenced_policies = extract_policy_titles(file_path)

            # 清洗主政策标题
            cleaned_policy_title = clean_policy_title(policy_title)

            # 获取主政策的ID
            policy_id = policy_data[policy_data['政策标题'] == cleaned_policy_title]['政策ID']
            policy_id = policy_id.iloc[0] if not policy_id.empty else None

            # 对每个引用的政策标题进行匹配
            for ref_policy in referenced_policies:
                cleaned_ref_policy = clean_policy_title(ref_policy)
                
                # 获取被引政策的ID
                ref_policy_id = policy_data[policy_data['政策标题'] == cleaned_ref_policy]['政策ID']
                ref_policy_id = ref_policy_id.iloc[0] if not ref_policy_id.empty else None

                all_data.append([policy_title, ref_policy, policy_id, ref_policy_id])

    # 保存为CSV文件
    df = pd.DataFrame(all_data, columns=['政策标题', '被引政策标题', '政策ID', '被引政策ID'])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

# 设置文件夹路径和输出文件路径
folder_path = 'data/policy_txt'
output_file = 'data/cite.csv'
policy_id_file = 'data/policy_base.xlsx'

# 处理政策文件并输出结果
process_policies(folder_path, output_file, policy_id_file)
print("政策引用提取及ID匹配完成！")