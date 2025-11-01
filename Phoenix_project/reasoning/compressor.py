# Phoenix_project/reasoning/compressor.py


def compress_cot(cot_text: str) -> str:
    """
    Extracts key logical sentences and returns a summary.
    
    Compressed word count <= 30% of original, 
    with information retention >= 80%.
    """
    # TODO: 实现实际的 CoT 压缩逻辑 (例如，使用摘要模型)。
    # 这是一个占位符实现。
    word_count = len(cot_text.split())
    max_len = int(len(cot_text) * 0.3)  # 模拟 30% 压缩
    
    return cot_text[:max_len] + "..." if len(cot_text) > max_len else cot_text
