from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from robot_memory_vla.runtime.models import SegmentationResult


class GeneralVLAAdapter:
    def __init__(
        self,
        generalvla_root: str,
        vis_save_path: Path,
        lisa_version: str = "",
        segagent_version: str = "",
        simpleclick_checkpoint: str = "",
        grounding_model: str = "qwen-full",
        seg_model: str = "simple_click",
        precision: str = "fp16",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        lisa_factory: Callable[[], object] | None = None,
        segagent_factory: Callable[[], object] | None = None,
    ) -> None:
        self.generalvla_root = Path(generalvla_root)
        self.vis_save_path = Path(vis_save_path)
        self.lisa_version = lisa_version
        self.segagent_version = segagent_version
        self.simpleclick_checkpoint = simpleclick_checkpoint
        self.grounding_model = grounding_model
        self.seg_model = seg_model
        self.precision = precision
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.vis_save_path.mkdir(parents=True, exist_ok=True)
        self._lisa_factory = lisa_factory
        self._segagent_factory = segagent_factory
        self._lisa = None
        self._segagent = None

    @staticmethod
    def _ensure_numpy_compat() -> None:
        if hasattr(np, "sctypes"):
            return
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.str_, np.bytes_],
        }

    def _default_lisa_version(self) -> Path:
        return self.generalvla_root / "pretrain_model" / "LISA-7B-v1-explanatory"

    def _default_segagent_version(self) -> Path:
        return self.generalvla_root / "pretrain_model" / "segagent" / "SegAgent-Model"

    def _default_simpleclick_checkpoint(self) -> Path:
        return self.generalvla_root / "pretrain_model" / "simpleclick" / "cocolvis_vit_large.pth"

    def _default_clip_vision_tower(self) -> Path:
        return self.generalvla_root / "pretrain_model" / "clip-vit-large-patch14"

    def _module(self):
        module_path = self.generalvla_root / "demo.py"
        simpleclick_root = self.generalvla_root / "third_party" / "SimpleClick"
        model_root = self.generalvla_root / "model"
        self._ensure_numpy_compat()
        self._ensure_cuda_allocator_settings()
        for search_path in (self.generalvla_root, simpleclick_root, model_root):
            if str(search_path) not in sys.path:
                sys.path.insert(0, str(search_path))
        spec = importlib.util.spec_from_file_location("generalvla_demo", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load GeneralVLA demo module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _ensure_cuda_allocator_settings(self) -> None:
        if not self.device.startswith("cuda"):
            return
        current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in current:
            return
        if current:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{current},expandable_segments:True"
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def _cuda_index(self) -> int:
        if not self.device.startswith("cuda"):
            return 0
        _, _, suffix = self.device.partition(":")
        if not suffix:
            return 0
        try:
            return int(suffix)
        except ValueError:
            return 0

    def _grounding_offload_dir(self) -> Path:
        path = self.vis_save_path / "offload" / "qwen_full"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _lisa_offload_dir(self) -> Path:
        path = self.vis_save_path / "offload" / "lisa"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _grounding_max_memory(self) -> dict[object, str]:
        try:
            import torch
        except ImportError:
            return {"cpu": self._cpu_max_memory()}
        if not torch.cuda.is_available() or not self.device.startswith("cuda"):
            return {"cpu": self._cpu_max_memory()}
        index = self._cuda_index()
        free_gib = self._gpu_free_gib(torch, index)
        gpu_budget_gib = max(2, free_gib - 4)
        return {
            index: f"{gpu_budget_gib}GiB",
            "cpu": self._cpu_max_memory(),
        }

    @staticmethod
    def _gpu_free_gib(torch_module, index: int) -> int:
        try:
            free_bytes, _ = torch_module.cuda.mem_get_info(index)
            return math.ceil(free_bytes / 1024**3)
        except Exception:
            total_bytes = torch_module.cuda.get_device_properties(index).total_memory
            return math.ceil(total_bytes / 1024**3)

    def _lisa_gpu_max_memory(self) -> str:
        try:
            import torch
        except ImportError:
            return "3GiB"
        if not torch.cuda.is_available() or not self.device.startswith("cuda"):
            return "3GiB"
        free_gib = self._gpu_free_gib(torch, self._cuda_index())
        budget_gib = min(5, max(3, free_gib - 2))
        return f"{budget_gib}GiB"

    def _lisa_cpu_max_memory(self) -> str:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            available_gib = math.floor((page_size * available_pages) / 1024**3)
        except (AttributeError, OSError, ValueError):
            return "10GiB"
        budget_gib = min(10, max(4, available_gib - 8))
        return f"{budget_gib}GiB"

    def _cpu_max_memory(self) -> str:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return "24GiB"
        available_gib = math.floor((page_size * available_pages) / 1024**3)
        budget_gib = max(4, available_gib - 4)
        return f"{budget_gib}GiB"

    def _build_qwen_load_kwargs(self) -> dict[str, object]:
        if not self.device.startswith("cuda"):
            return {
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
            }
        kwargs: dict[str, object] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "offload_folder": str(self._grounding_offload_dir()),
            "max_memory": self._grounding_max_memory(),
        }
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        return kwargs

    def _build_lisa_load_kwargs(self) -> dict[str, object]:
        if not self.device.startswith("cuda"):
            return {
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
            }
        kwargs: dict[str, object] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "offload_folder": str(self._lisa_offload_dir()),
            "offload_state_dict": True,
            "max_memory": {
                self._cuda_index(): self._lisa_gpu_max_memory(),
                "cpu": self._lisa_cpu_max_memory(),
            },
        }
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        return kwargs

    def _build_lisa_retry_load_kwargs(self) -> dict[str, object]:
        if not self.device.startswith("cuda"):
            return {
                "low_cpu_mem_usage": True,
            }
        return {
            "low_cpu_mem_usage": True,
            "device_map": {
                "": self._cuda_index(),
                "lm_head": "cpu",
                "model.text_hidden_fcs": "cpu",
                "model.model.embed_tokens": "cpu",
            },
            "offload_folder": str(self._lisa_offload_dir()),
            "offload_state_dict": True,
        }

    @staticmethod
    def _build_qwen_cpu_fallback_kwargs() -> dict[str, object]:
        return {
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
        }

    def _build_lisa_cpu_fallback_kwargs(self) -> dict[str, object]:
        return {
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
            "offload_state_dict": True,
            "offload_folder": str(self._lisa_offload_dir()),
        }

    def _build_args(self, module):
        lisa_version = self.lisa_version or str(self._default_lisa_version())
        segagent_version = self.segagent_version or str(self._default_segagent_version())
        simpleclick_checkpoint = self.simpleclick_checkpoint or str(self._default_simpleclick_checkpoint())
        args = [
            "--lisa_version",
            lisa_version,
            "--segagent_version",
            segagent_version,
            "--checkpoint",
            simpleclick_checkpoint,
            "--grounding_model",
            self.grounding_model,
            "--seg_model",
            self.seg_model,
            "--precision",
            self.precision,
            "--device",
            self.device,
        ]
        if self.load_in_4bit:
            args.append("--load_in_4bit")
        parsed_args = module.parse_args(args)
        clip_vision_tower = self._default_clip_vision_tower()
        if clip_vision_tower.exists():
            parsed_args.vision_tower = str(clip_vision_tower)
        parsed_args.robot_memory_vla_qwen_load_kwargs = self._build_qwen_load_kwargs()
        parsed_args.robot_memory_vla_qwen_cpu_fallback_kwargs = self._build_qwen_cpu_fallback_kwargs()
        parsed_args.robot_memory_vla_lisa_load_kwargs = self._build_lisa_load_kwargs()
        parsed_args.robot_memory_vla_lisa_retry_load_kwargs = self._build_lisa_retry_load_kwargs()
        parsed_args.robot_memory_vla_lisa_cpu_fallback_kwargs = self._build_lisa_cpu_fallback_kwargs()
        return parsed_args

    def _get_lisa(self):
        if self._lisa is None:
            if self._lisa_factory is not None:
                self._lisa = self._lisa_factory()
            else:
                module = self._module()
                args = self._build_args(module)
                self._lisa = module.LISASegmenter(args)
        return self._lisa

    def _get_segagent(self):
        if self._segagent is None:
            if self._segagent_factory is not None:
                self._segagent = self._segagent_factory()
            else:
                module = self._module()
                args = self._build_args(module)
                args.model = args.segagent_version
                grounding_model, seg_model = module.load_segagent_models(args)
                self._segagent = module.SegAgentInteractiveSegmenter(grounding_model, seg_model, args)
        return self._segagent

    def _release_runtime_component(self, attr_name: str) -> None:
        component = getattr(self, attr_name)
        if component is None:
            return
        release = getattr(component, "release_resources", None)
        if callable(release):
            release()
        setattr(self, attr_name, None)
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _write_input_image(self, image_bgr: np.ndarray, stem: str) -> Path:
        path = self.vis_save_path / f"{stem}.jpg"
        cv2.imwrite(str(path), image_bgr)
        return path

    @staticmethod
    def _normalize_mask(mask_like) -> np.ndarray:
        mask = np.asarray(mask_like)
        if mask.ndim == 3:
            mask = mask[0]
        return (mask > 0).astype(np.uint8)

    def segment_pick_object(
        self,
        image_bgr: np.ndarray,
        task_text: str,
        pick_target_text: str,
    ) -> SegmentationResult:
        image_path = self._write_input_image(image_bgr, "pick_object_input")
        lisa = self._get_lisa()
        try:
            result = lisa.segment(str(image_path), pick_target_text or task_text)
        finally:
            self._release_runtime_component("_lisa")
        mask = self._normalize_mask(result["masks"][0])
        return SegmentationResult(
            mask=mask,
            score=None,
            text_response=result.get("text"),
            debug_images={"input": str(image_path)},
        )

    def segment_grasp_region(
        self,
        image_bgr: np.ndarray,
        object_mask: np.ndarray,
        pick_part_text: str | None,
    ) -> SegmentationResult:
        self._release_runtime_component("_lisa")
        image_path = self._write_input_image(image_bgr, "grasp_region_input")
        prompt = pick_part_text or "Segment the graspable part of the target object."
        segagent = self._get_segagent()
        try:
            result = segagent.segment(str(image_path), initial_mask=object_mask, prompt=prompt)
        finally:
            self._release_runtime_component("_segagent")
        mask = self._normalize_mask(result["mask"])
        return SegmentationResult(
            mask=mask,
            score=None,
            text_response=result.get("outputs"),
            debug_images={"input": str(image_path)},
        )
