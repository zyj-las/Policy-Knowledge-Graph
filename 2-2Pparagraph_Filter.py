import pandas as pd
import re
# 读取Excel文件
policy_file_path = 'data/policy_base.xlsx'
split_file_path = 'data/paragraph_base.xlsx'

policy_df = pd.read_excel(policy_file_path)
split_df = pd.read_excel(split_file_path)

# 创建一个空的DataFrame来存储筛选结果
result_df = pd.DataFrame(columns=['政策ID','政策标题', '政策条文'])

# 遍历政策文件的每一行
for _, policy_row in policy_df.iterrows():
    policy_id = policy_row['政策ID']
    policy_title = policy_row['政策标题']
    text_object = policy_row['文本分析对象']
    # 找到相同政策标题的段落
    matched_rows = split_df[split_df['政策标题'] == policy_title]
    if not matched_rows.empty:
        if text_object == '全文':
            # 保留所有段落
            filtered_rows = matched_rows
        elif '排除' in text_object:
            # 去除所有段落
            filtered_rows = pd.DataFrame(columns=matched_rows.columns)
        elif '节选' in text_object:
            # 筛选包含特定关键字的段落
            keywords = re.findall(r'“(.*?)”', text_object)
            keyword_pattern = '|'.join(keywords)
            filtered_rows = matched_rows[matched_rows['政策条文'].str.contains(keyword_pattern, na=False)]
        else:
            # 如果不符合任何条件，返回空DataFrame
            filtered_rows = pd.DataFrame(columns=matched_rows.columns)
        
        # 添加“政策ID”列到筛选结果中
        filtered_rows['政策ID'] = policy_id

        # 将筛选结果添加到结果DataFrame中
        result_df = pd.concat([result_df, filtered_rows], ignore_index=True)


# 输出筛选后的数据
print(result_df)
result_df.to_csv('data/paragraph.csv', index=False, encoding='utf-8-sig')