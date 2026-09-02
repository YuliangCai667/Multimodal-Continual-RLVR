from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def _longest_common_prefix(first: torch.Tensor, second: torch.Tensor) -> int:
    limit = min(first.numel(), second.numel())
    mismatch = (first[:limit] != second[:limit]).nonzero(as_tuple=False)
    return int(mismatch[0].item()) if mismatch.numel() else limit


def _navigation_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.replace("{question}", question).replace("\n", " ")


def _multimodal_user_content(question: str, prompt_template: str, image) -> list[dict[str, Any]]:
    # This is the Figure-B construction, including its pixel bounds and image placement.
    text = _navigation_prompt(question, prompt_template)
    image_item = {
        "type": "image",
        "image": image,
        "min_pixels": 8 * 32 * 32,
        "max_pixels": 512 * 32 * 32,
    }
    pattern = r"(<image(?:[\s_]+\d+)?>)"
    if not re.search(pattern, text):
        return [image_item, {"type": "text", "text": text}]
    content: list[dict[str, Any]] = []
    image_added = False
    for part in re.split(pattern, text):
        if not part:
            continue
        if re.fullmatch(pattern, part):
            if not image_added:
                content.append(image_item)
                image_added = True
        else:
            content.append({"type": "text", "text": part})
    return content


def prepare_teacher_forced_inputs(
    processor,
    *,
    question: str,
    target_completion: str,
    image_path: str | Path,
    prompt_template: str,
    device: torch.device | str,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Figure-B teacher forcing: only assistant target tokens receive labels."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    user_content = _multimodal_user_content(question, prompt_template, image)
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": [{"type": "text", "text": target_completion}]},
    ]
    prompt_rendered = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_rendered = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_images, prompt_videos = process_vision_info(prompt_messages)
    full_images, full_videos = process_vision_info(full_messages)
    prompt_inputs = processor(
        text=[prompt_rendered], images=prompt_images, videos=prompt_videos,
        padding=True, return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_rendered], images=full_images, videos=full_videos,
        padding=True, return_tensors="pt",
    )
    prompt_length = _longest_common_prefix(prompt_inputs["input_ids"][0], full_inputs["input_ids"][0])
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if int((labels != -100).sum()) < 1:
        raise RuntimeError("Navigation probe has no assistant target tokens")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in full_inputs.items()}
    return inputs, labels.to(device)


def navigation_geometry_loss(model, processor, probe: dict[str, Any], prompt_template: str) -> torch.Tensor:
    inputs, labels = prepare_teacher_forced_inputs(
        processor,
        question=probe["question"],
        target_completion=probe["target_completion"],
        image_path=probe["image_path"],
        prompt_template=prompt_template,
        device=next(model.parameters()).device,
    )
    # Qwen's causal-LM loss is the mean over labels != -100, so each probe is
    # normalized by its valid assistant target-token count, exactly as Figure B.
    output = model(**inputs, labels=labels, use_cache=False, return_dict=True)
    return output.loss
