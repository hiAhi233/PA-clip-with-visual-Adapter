"""三层提示词扩展，形成多属性文本锚点（brain MRI）。

文档原文：
    将一对 normal/abnormal 提示扩展为三层或者多层提示词，形成多属性文本锚点。
    如原来是：
        a normal brain MRI
        an abnormal brain MRI
    现在可以在三个层级上进行扩展。

    三层提示词各自独立地与图像特征做匹配，输出各自的判别结果，最后再融合。

默认 brain_mri_sentence 与胸腺瘤 sentence 版采用相同的分层与句式设计：
    Level 1：整体正常/异常，使用简短模态句。
    Level 2：脑部结构/局灶病变，使用 "brain MRI showing ..."。
    Level 3：信号、组织对比与边界，使用属性前置、模态后置的句式。

未指定 T1/T2/FLAIR 序列，因此不把异常固定写成高信号或低信号。
旧 brain_mri 词表保留给历史 checkpoint；新训练默认使用 brain_mri_sentence。
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class ThreeLevelPrompts:
    """三层提示词模板，每层 normal / abnormal 各一组（可多条，做均值池化）。"""

    normal: Dict[str, List[str]]
    abnormal: Dict[str, List[str]]

    @property
    def levels(self) -> List[str]:
        return list(self.normal.keys())

    def all_texts(self) -> List[str]:
        texts: List[str] = []
        for lvl in self.levels:
            texts.extend(self.normal[lvl])
            texts.extend(self.abnormal[lvl])
        return texts


LEGACY_BRAIN_MRI_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal brain MRI"],
        "2": ["a normal brain MRI with intact anatomy and clear ventricles"],
        "3": ["a normal brain MRI with homogeneous tissue texture and sharp boundaries"],
    },
    abnormal={
        "1": ["an abnormal brain MRI"],
        "2": ["an abnormal brain MRI with a focal lesion"],
        "3": ["an abnormal brain MRI with irregular texture and blurred boundaries"],
    },
)


BRAIN_MRI_SENTENCE_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal brain MRI"],
        "2": ["brain MRI showing preserved brain anatomy and symmetric ventricles"],
        "3": ["preserved gray-white matter contrast with well-defined tissue boundaries on brain MRI"],
    },
    abnormal={
        "1": ["an abnormal brain MRI"],
        "2": ["brain MRI showing a focal lesion in the brain parenchyma"],
        "3": ["heterogeneous lesion signal with irregular indistinct margins on brain MRI"],
    },
)

# checkpoint 的旧名称必须继续指向旧词表，不能随默认版本改变而重新解释旧权重。
BRAIN_MRI_PROMPT_SETS = {
    "brain_mri": LEGACY_BRAIN_MRI_PROMPTS,
    "brain_mri_sentence": BRAIN_MRI_SENTENCE_PROMPTS,
}
DEFAULT_BRAIN_MRI_PROMPT_SET = "brain_mri_sentence"
DEFAULT_BRAIN_MRI_PROMPTS = BRAIN_MRI_PROMPT_SETS[DEFAULT_BRAIN_MRI_PROMPT_SET]


def resolve_prompt_set(requested, saved, registry, default, *, loading=False):
    """新实验用默认版本；加载权重继承版本，缺元数据时必须明确指定。"""
    if loading and not saved and requested is None:
        raise ValueError("checkpoint 未记录 prompt_set，请显式指定 --prompt-set 为该权重训练时的版本")
    if saved and saved not in registry:
        raise ValueError(f"checkpoint 的提示词 {saved!r} 不属于当前任务")
    if requested is not None and saved and requested != saved:
        raise ValueError(f"提示词不一致：checkpoint={saved}，--prompt-set={requested}；不能按原实验续训/评估")
    name = requested or saved or default
    if name not in registry:
        raise ValueError(f"未知提示词版本：{name!r}")
    return name


def build_text_anchors(
    texts: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """tokenize 字符串列表，返回 input_ids / attention_mask（CPU 长整型）。"""
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
