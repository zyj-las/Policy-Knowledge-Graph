import os
import re
import pandas as pd
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()
def process_text_lines(lines):
    policy_title = None
    chapter_title = None
    current_paragraph = []
    paragraphs = []
    chapter_re = re.compile(r'(第[\d〇一二三四五六七八九十百零]+章[^\n]{0,30}|[一二三四五六七八九十百零]+[、 ][^\n]{0,30})')
    for line in lines:
        line = line.strip()
        if not policy_title and line:
            policy_title = line
            continue
        chapter_match = chapter_re.match(line)
        if chapter_match:   
            if len(line) <= 40:
                if current_paragraph:
                    paragraphs.append((policy_title, chapter_title, '\n'.join(current_paragraph)))
                    current_paragraph = []
                chapter_title = line
            else:
                current_paragraph.append(line)
            continue
        if not line:
            if current_paragraph:
                paragraphs.append((policy_title, chapter_title, '\n'.join(current_paragraph)))
                current_paragraph = []
        else:
            current_paragraph.append(line)
    if current_paragraph:
        paragraphs.append((policy_title, chapter_title, '\n'.join(current_paragraph)))
    return [p for p in paragraphs if p[2]]

def process_files_in_directory(directory_path, output_file):
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在，请检查路径是否正确。")
        return
    
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    all_paragraphs = []

    for txt_file in txt_files:
        input_file_path = os.path.join(directory_path, txt_file)
        lines = read_text_file(input_file_path)
        processed_paragraphs = process_text_lines(lines)
        all_paragraphs.extend(processed_paragraphs)
    df = pd.DataFrame(all_paragraphs, columns=['政策标题', '章节标题', '政策条文'])
    df.to_excel(output_file, index=False)
    print(f"处理完成，所有文件的结果已保存至 {output_file}")

if __name__ == '__main__':
    directory_path = "data/policy_txt"
    output_file = "data/paragraph_base.xlsx"
    process_files_in_directory(directory_path, output_file)