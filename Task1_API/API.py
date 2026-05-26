import fitz
import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        # 仅提取前 5000 个字符（大部分病历核心信息在开头），防止超长
        text = "".join([page.get_text() for page in doc])
        return text[:5000] 
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_info_with_llm(text):
    if not text: return None
    
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 增加 System Role 定义，强化指令执行能力
    messages = [
        {"role": "system", "content": "你是一个严谨的医疗数据提取助手，请只输出标准的 JSON，不要包含 Markdown 标记。"},
        {"role": "user", "content": f"{prompt_template}\n\n病例文本: {text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            temperature=0.1 # 降低随机性，提高提取稳定性
        )
        content = response.choices[0].message.content
        # 清洗可能存在的 Markdown 符号
        return re.sub(r"```json|```", "", content).strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

# 定义 Prompt 模板（保持你原有的逻辑）
prompt_template = """ # Role
你是一位临床医学领域的数据挖掘专家，擅长从非结构化电子病历（EMR）中提取高质量的结构化临床实体。

# Task
请从提供的病例文本中提取以下关键实体，并严格按照指定的 JSON 格式输出。

# Extraction Targets
1. **患者基本信息**: (包括姓名、性别、年龄、职业等，若缺失则置为"null")
2. **主要症状**: (本次就诊的主要主诉及症状描述，以列表形式提取)
3. **既往史**: (过往患病情况、手术史、过敏史等)
4. **诊断结果**: (医生给出的最终临床诊断)
5. **治疗方案**: (药物、手术、物理治疗或后续随访建议)

# Constraints & Rules
1. **准确性优先**: 必须严格基于病例文本提取，严禁生成或推断原文中不存在的信息。
2. **格式规范**: 仅输出标准的 JSON 对象，不要包含任何前导文本（如“好的，这是提取结果...”）或后续评价。
3. **缺失处理**: 若某项实体在原文中未提及，请将其值设为 "null"。
4. **简洁性**: 提取的实体应保留医学专业术语，去除冗余描述。

# Few-Shot Example
Input: "患者张三，男，45岁，患高血压病史5年。近一周出现头晕、乏力，测量血压160/100mmHg。诊断为原发性高血压。建议口服硝苯地平控释片，低盐饮食。"
Output:
{
  "患者基本信息": {"姓名": "张三", "性别": "男", "年龄": "45岁", "职业": "null"},
  "主要症状": ["头晕", "乏力"],
  "既往史": ["高血压病史5年"],
  "诊断结果": ["原发性高血压"],
  "治疗方案": ["口服硝苯地平控释片", "低盐饮食"]
}

# Input Case
{病例内容} """

# 执行逻辑
result = extract_info_with_llm(extract_text_from_pdf(r"D:\Clone Repository\123\Task1_API\A case of portal vein recanalization and symptomatic heart failure.pdf"))
if result:
    try:
        data = json.loads(result)
        output_filename = "extracted_medical_info.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"成功！已将提取结果保存至: {os.path.abspath(output_filename)}")
    except json.JSONDecodeError:
        print("提取内容非合规 JSON:", result)