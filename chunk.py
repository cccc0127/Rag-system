import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

from privacy_judge import PrivacyScorer


DocumentSource = Union[str, Path, Iterable[Dict[str, str]]]

#先按段落切分，再按句子切分，最后按软分隔符切分
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])\s*|(?<=[A-Za-z0-9][.])\s+")
SOFT_SEPARATOR_RE = re.compile(r"(?<=[，,、：:])\s*|\s+")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """文本切分函数，将输入文本切分成多个片段，每个片段的长度不超过 chunk_size，并且相邻片段之间有 overlap 的重叠部分。"""
    _validate_chunk_args(chunk_size, overlap)
    if not text or not text.strip():
        return []

    units = list(_semantic_units(text, chunk_size))
    if not units:
        return []

    chunks: List[str] = []
    current_units: List[str] = []
    soft_limit = max(1, chunk_size - overlap)

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        candidate = _join_units([*current_units, unit])
        if not current_units or len(candidate) <= soft_limit:
            current_units.append(unit)
            continue

        chunks.append(_join_units(current_units))
        current_units = [unit]

    if current_units:
        chunks.append(_join_units(current_units))

    return [chunk for chunk in chunks if chunk]


def iter_chunk_documents(
    source: DocumentSource,
    chunk_size: int = 1000,
    overlap: int = 200,
    scorer: Optional[PrivacyScorer] = None,
) -> Iterator[Dict[str, object]]:
    """从已加载的文档或知识库文件夹中提取经过隐私评分处理的片段，负责把文档流变成带隐私评分的chunk流。"""
    _validate_chunk_args(chunk_size, overlap)
    # 初始化隐私评分器，如果外部已经传入PrivacyScorer就复用外部的，如果没有传就创建一个默认评分器
    scorer = scorer or PrivacyScorer()

    for doc in _iter_documents(source):
        filename = doc.get("filename", "")
        content = doc.get("content", "")

        for chunk_id, text_chunk in enumerate(chunk_text(content, chunk_size, overlap)):
            # 对每个文本块进行隐私评估
            profile = scorer.get_privacy_profile(text_chunk)
            yield {
                "filename": filename,
                "chunk_id": chunk_id,
                "content": text_chunk,
                "raw_sensitivity_score": profile["raw_sensitivity_score"],
                "privacy_epsilon": profile["privacy_epsilon"],
                "dynamic_delta": profile["dynamic_delta"],
            }


def chunk_documents(
    source: DocumentSource,
    chunk_size: int = 1000,
    overlap: int = 200,
    scorer: Optional[PrivacyScorer] = None,
) -> List[Dict[str, object]]:
    """Return privacy-scored chunk records for all documents in the source."""
    return list(iter_chunk_documents(source, chunk_size, overlap, scorer))


def _iter_documents(source: DocumentSource) -> Iterator[Dict[str, str]]:
    """递归遍历文档源，返回所有文档。如果 source 是路径，就调用 loader.py 读取文件
        如果 source 已经是文档列表/生成器，就直接遍历"""
    if isinstance(source, (str, Path)):
        from loader import load_documents

        yield from load_documents(source)
        return

    yield from source


def _semantic_units(text: str, chunk_size: int) -> Iterator[str]:
    for paragraph in PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _split_sentences(paragraph, chunk_size):
            yield sentence


def _split_sentences(text: str, chunk_size: int) -> Iterator[str]:
    for sentence in SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= chunk_size:
            yield sentence
        else:
            yield from _split_oversized_unit(sentence, chunk_size)


def _split_oversized_unit(text: str, chunk_size: int) -> List[str]:
    """Split oversized text by soft separators; hard wrapping is the last resort."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    parts = [part.strip() for part in SOFT_SEPARATOR_RE.split(text) if part.strip()]
    if len(parts) > 1:
        packed: List[str] = []
        current = ""
        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    packed.extend(_split_oversized_unit(current, chunk_size))
                current = part
        if current:
            packed.extend(_split_oversized_unit(current, chunk_size))
        return packed

    # 如果软分隔符切分失败，就按硬换行切分
    return textwrap.wrap(
        text,
        width=chunk_size,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=True,
    )


def _join_units(units: Iterable[str]) -> str:
    return "\n".join(unit.strip() for unit in units if unit and unit.strip()).strip()


def _validate_chunk_args(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
