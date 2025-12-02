import pandas as pd
policy_citation_df = pd.read_csv('data/cite_base.csv', encoding='utf-8-sig')
policy_publish_df = pd.read_csv('data/release.csv', encoding='utf-8-sig')
policy_org_df = pd.read_csv('data/org.csv', encoding='utf-8-sig')

# 合并文件，基于政策ID匹配
matched_df = pd.merge(policy_citation_df, policy_publish_df, left_on='被引政策ID', right_on='政策ID')

# 选择发布类型为“牵头发布”的行
lead_publish_df = matched_df[matched_df['发布类型'] == '牵头发布']
# 将【政策主体】与【主体类型】对应
merged_df = pd.merge(lead_publish_df, policy_org_df, on='政策主体ID', how='left')
# 去重以确保每个被引政策ID只有一条记录
merged_df = merged_df.drop_duplicates(subset=['被引政策ID'])

# 根据主体类型确定引用类型
def get_reference_type(row):
    if row['主体类型'] == '党政机关':
        return '引用党政机关政策'
    elif row['主体类型'] == '科研管理机构':
        return '引用科研管理机构政策'
    elif row['主体类型'] == '责任主体单位':
        return '引用责任主体单位政策'
    elif row['主体类型'] == '科技/学术团体':
        return '引用科技/学术团体政策'
    else:
        return '其他引用类型'
merged_df['引用类型'] = merged_df.apply(get_reference_type, axis=1)

# 将【引用类型】合并回政策引用的原始数据框，并去重
policy_citation_df = policy_citation_df.merge(merged_df[['被引政策ID', '引用类型']], on='被引政策ID', how='left')
# 去重，确保没有重复的被引政策
policy_citation_df = policy_citation_df.drop_duplicates()

# 检查合并结果的行数是否匹配
print(f"原始政策引用行数: {len(policy_citation_df)}")
print(f"最终生成的行数: {len(policy_citation_df)}")

# 将结果保存到新文件
policy_citation_df.to_csv('data/cite.csv', index=False, encoding='utf-8-sig')