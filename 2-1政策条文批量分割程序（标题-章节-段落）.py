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
    # Regular expression to match chapter titles
    chapter_re = re.compile(r'(第[\d〇一二三四五六七八九十百零]+章[^\n]{0,30}|[一二三四五六七八九十百零]+[、 ][^\n]{0,30})')
    for line in lines:
        line = line.strip()
        # Identify policy title
        if not policy_title and line:
            policy_title = line
            continue
        # Check if the current line is a potential chapter title
        chapter_match = chapter_re.match(line)
        if chapter_match:
            # Check if the entire paragraph (line) is short enough to be considered a chapter title
            if len(line) <= 40:
                # Save the previous paragraph before setting the new chapter title
                if current_paragraph:
                    paragraphs.append((policy_title, chapter_title, '\n'.join(current_paragraph)))
                    current_paragraph = []
                chapter_title = line
            else:
                # If it's too long, consider it part of the paragraph
                current_paragraph.append(line)
            continue
        # Paragraphs are separated by empty lines
        if not line:
            if current_paragraph:
                paragraphs.append((policy_title, chapter_title, '\n'.join(current_paragraph)))
                current_paragraph = []
        else:
            current_paragraph.append(line)
    # Add the last paragraph if exists
    if current_paragraph:
        paragraphs.append((policy_title, chapter_title, '\n'.join(current_paragraph)))
    return [p for p in paragraphs if p[2]]  # Remove entries where paragraph is empty

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
    
    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(all_paragraphs, columns=['政策标题', '章节标题', '政策条文'])

    # Write the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"处理完成，所有文件的结果已保存至 {output_file}")

if __name__ == '__main__':
    directory_path = r"D:\File_zyj\2.论文文件\4.小论文_政策知识图谱\科研诚信政策-国家-文本分析-20240912"
    output_file = r"D:\File_zyj\2.论文文件\4.小论文_政策知识图谱\数据分析文件0912\政策分段-新.xlsx"
    process_files_in_directory(directory_path, output_file)