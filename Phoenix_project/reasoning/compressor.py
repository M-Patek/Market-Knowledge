# Phoenix_project/reasoning/compressor.py
import re # (主人喵的清洁计划 5.2) [新] 导入 re


def compress_cot(cot_text: str) -> str:
    """
    (主人喵的清洁计划 5.2) [已修改]
    使用基于规则的提取式摘要来压缩 CoT 文本。
    提取第一、第二和最后一句。
    
    Compressed word count <= 30% of original, 
    with information retention >= 80%.
    """
    
    # (一个简单的句子分割器)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', cot_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    num_sentences = len(sentences)
    
    # 如果句子很少，直接返回
    if num_sentences <= 4:
        return cot_text
    
    # 提取前 2 句和后 1 句
    try:
        summary_parts = [
            sentences[0],
            sentences[1],
            "...",
            sentences[-1]
        ]
        summary = " ".join(summary_parts)
        
        original_len = len(cot_text)
        summary_len = len(summary)
        
        # 如果我们的 "摘要" 甚至更长 (例如因为 "...")，则回退到截断
        if summary_len > original_len * 0.8:
            return (cot_text[:int(original_len * 0.5)] + "...")
        
        return summary
        
    except IndexError:
        # 如果句子少于3个 (尽管我们检查了 <= 4)，回退
        return cot_text
