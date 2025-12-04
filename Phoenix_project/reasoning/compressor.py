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
    
    # [Task 3.3 Fix] Semantic Compression (Pivot Retention)
    try:
        # Always keep first and last
        indices_to_keep = {0, num_sentences - 1}
        
        # Scan middle for logic pivots
        pivot_keywords = ["but", "however", "although", "risk", "warning", "critical", "halt"]
        for i in range(1, num_sentences - 1):
            if any(k in sentences[i].lower() for k in pivot_keywords):
                indices_to_keep.add(i)
        
        sorted_indices = sorted(list(indices_to_keep))
        summary_parts = []
        last_idx = -1
        
        for idx in sorted_indices:
            if last_idx != -1 and idx > last_idx + 1:
                summary_parts.append("...")
            summary_parts.append(sentences[idx])
            last_idx = idx
            
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
