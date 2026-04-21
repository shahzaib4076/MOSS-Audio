"""HuggingFace inference wrapper for MOSS-Audio."""

from __future__ import annotations

import os
from typing import Optional

import torch

from src.audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor
from transformers import BitsAndBytesConfig

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Audio"


def read_env_model_id() -> str:
    return os.environ.get("MOSS_AUDIO_MODEL_ID", DEFAULT_MODEL_ID)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class MossAudioHFInference:
    """Thin wrapper that loads model + processor and exposes a single
    ``generate`` method for both audio-grounded and text-only queries."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
        torch_dtype: str = "auto",
        enable_time_marker: bool = True,
        quantization_bits: int = 4,
        num_gpus: int = None,
    ):
        self.device = device

        if quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization_bits == 16:
            quantization_config = None
        else:
            raise ValueError(f"quantization_bits must be 4, 8, or 16. Got {quantization_bits}")

        # Build device_map and max_memory based on num_gpus
        if num_gpus == 1:
            # Pin everything to a single GPU — avoids PCI-e overhead
            device_map = "cuda:0"
        elif num_gpus is not None and num_gpus >= 2:
            # Restrict accelerate to only the first N GPUs
            available = torch.cuda.device_count()
            use_count = min(num_gpus, available)
            max_memory = {i: torch.cuda.get_device_properties(i).total_mem for i in range(use_count)}
            device_map = "auto"
        else:
            # Default: let accelerate decide across all visible GPUs
            device_map = "auto"
            max_memory = None

        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            quantization_config=quantization_config,
        )
        if num_gpus is not None and num_gpus >= 2:
            load_kwargs["max_memory"] = max_memory

        self.model = MossAudioModel.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        self.model.eval()
        self.processor = MossAudioProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            enable_time_marker=enable_time_marker,
        )

    @torch.no_grad()
    def generate(
        self,
        question: str,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 1024,
        num_beams: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> str:
        if audio_path is not None:
            raw_audio = load_audio(audio_path, sample_rate=self.processor.config.mel_sr)
            inputs = self.processor(text=question, audios=[raw_audio], return_tensors="pt")
        else:
            inputs = self.processor(text=question, return_tensors="pt")

        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)

        audio_input_mask = inputs["input_ids"] == self.processor.audio_token_id
        inputs["audio_input_mask"] = audio_input_mask

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            gen_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(
            generated_ids[0, input_len:], skip_special_tokens=True
        )


__all__ = ["MossAudioHFInference", "read_env_model_id", "resolve_device"]
