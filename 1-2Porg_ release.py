import pandas as pd
excel_file = 'data/policy_base.xlsx'
csv_file = 'data/org.csv'
output_file = 'data/relesae.csv'

df_excel = pd.read_excel(excel_file)
df_csv = pd.read_csv(csv_file)

# 预处理Excel中的“政策主体”列，将多个主体分开
df_excel['政策主体'] = df_excel['政策主体'].str.split('、')
matched_data = []
for index, row in df_excel.iterrows():
    policy_id = row['政策ID']
    policy_title = row['政策标题']
    subjects = row['政策主体']
    first_subject = True  # 标记第一个匹配的政策主体
    # 对每个“政策主体”进行匹配
    for subject in subjects:
        match = df_csv[df_csv['政策主体'] == subject]
        if not match.empty:
            # 对每个匹配项，将政策主体ID、政策主体、政策标题ID、政策标题存储
            for _, match_row in match.iterrows():
                # 判断发布类型：第一个匹配项为“牵头”，其余为“合作”
                release_type = '牵头发布' if first_subject else '合作发布'
                matched_data.append({
                    '政策主体ID': match_row['政策主体ID'],
                    '政策主体': match_row['政策主体'],
                    '政策标题ID': policy_id,
                    '政策标题': policy_title,
                    '发布类型': release_type
                })
                first_subject = False  # 第一个匹配后，将标志设置为False
df_result = pd.DataFrame(matched_data)
df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"匹配结果已保存到 {output_file}")