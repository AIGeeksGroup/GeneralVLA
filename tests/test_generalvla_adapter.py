from pathlib import Path

import numpy as np
import pytest

from robot_memory_vla.adapters.generalvla_adapter import GeneralVLAAdapter


class FakeLISA:
    def segment(self, image_path: str, prompt: str) -> dict:
        return {
            "text": f"segment:{prompt}",
            "masks": [np.array([[[0, 1], [1, 1]]], dtype=np.uint8)],
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
        }


class FakeSegAgent:
    def segment(self, image_path: str, initial_mask=None, prompt=None) -> dict:
        return {
            "mask": np.array([[[1, 1], [0, 0]]], dtype=np.uint8),
            "outputs": f"grasp:{prompt}",
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "clicks": [],
        }


class ReleasableFakeLISA(FakeLISA):
    def __init__(self) -> None:
        self.released = False

    def release_resources(self) -> None:
        self.released = True


class ReleasableFakeSegAgent(FakeSegAgent):
    def __init__(self) -> None:
        self.released = False

    def release_resources(self) -> None:
        self.released = True


def test_generalvla_adapter_returns_numpy_masks(tmp_path: Path) -> None:
    adapter = GeneralVLAAdapter(
        generalvla_root="/home/why/GeneralVLA",
        vis_save_path=tmp_path,
        lisa_factory=lambda: FakeLISA(),
        segagent_factory=lambda: FakeSegAgent(),
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    object_result = adapter.segment_pick_object(image, "抓起瓶盖", "瓶盖")
    grasp_result = adapter.segment_grasp_region(image, object_result.mask, "瓶盖上沿")

    assert object_result.mask.shape == (2, 2)
    assert grasp_result.mask.shape == (2, 2)
    assert object_result.text_response == "segment:瓶盖"
    assert grasp_result.text_response == "grasp:瓶盖上沿"


def test_generalvla_adapter_can_import_demo_with_local_sibling_module(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    root.mkdir()
    (root / "helper_mod.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "demo.py").write_text(
        "from types import SimpleNamespace\n"
        "from helper_mod import VALUE\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': VALUE + ':' + prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        pass\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / "vis",
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    result = adapter.segment_pick_object(image, "抓起瓶盖", "瓶盖")
    assert result.text_response == "ok:瓶盖"


def test_generalvla_adapter_can_import_demo_with_simpleclick_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    simpleclick = root / "third_party" / "SimpleClick" / "isegm"
    simpleclick.mkdir(parents=True)
    (simpleclick / "__init__.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "demo.py").write_text(
        "import sys\n"
        "from types import SimpleNamespace\n"
        "sys.path.append('./third_party/SimpleClick')\n"
        "from isegm import VALUE\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': VALUE + ':' + prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        pass\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / "vis",
    )

    result = adapter.segment_pick_object(np.zeros((2, 2, 3), dtype=np.uint8), "抓起瓶盖", "瓶盖")
    assert result.text_response == "ok:瓶盖"


def test_generalvla_adapter_can_import_demo_with_model_subpackage_path(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    segment_anything = root / "model" / "segment_anything"
    segment_anything.mkdir(parents=True)
    (segment_anything / "__init__.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "demo.py").write_text(
        "from types import SimpleNamespace\n"
        "from segment_anything import VALUE\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': VALUE + ':' + prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        pass\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / "vis",
    )

    result = adapter.segment_pick_object(np.zeros((2, 2, 3), dtype=np.uint8), "抓起瓶盖", "瓶盖")
    assert result.text_response == "ok:瓶盖"


def test_generalvla_adapter_restores_numpy_sctypes_for_legacy_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "fake_generalvla"
    root.mkdir()
    monkeypatch.delattr(np, "sctypes", raising=False)
    (root / "demo.py").write_text(
        "import numpy as np\n"
        "from types import SimpleNamespace\n"
        "assert hasattr(np, 'sctypes')\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        pass\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / "vis",
    )

    result = adapter.segment_pick_object(np.zeros((2, 2, 3), dtype=np.uint8), "抓起瓶盖", "瓶盖")
    assert result.text_response == "瓶盖"


def test_generalvla_adapter_releases_stage_models_after_use(tmp_path: Path) -> None:
    lisa = ReleasableFakeLISA()
    segagent = ReleasableFakeSegAgent()
    adapter = GeneralVLAAdapter(
        generalvla_root="/home/why/GeneralVLA",
        vis_save_path=tmp_path,
        lisa_factory=lambda: lisa,
        segagent_factory=lambda: segagent,
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    object_result = adapter.segment_pick_object(image, "抓起瓶盖", "瓶盖")
    grasp_result = adapter.segment_grasp_region(image, object_result.mask, "瓶盖上沿")

    assert object_result.text_response == "segment:瓶盖"
    assert grasp_result.text_response == "grasp:瓶盖上沿"
    assert lisa.released is True
    assert segagent.released is True
    assert adapter._lisa is None
    assert adapter._segagent is None


def test_generalvla_adapter_sets_low_vram_qwen_load_kwargs(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    root.mkdir()
    (root / "demo.py").write_text(
        "from types import SimpleNamespace\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    kwargs = args.robot_memory_vla_qwen_load_kwargs\n"
        "    cpu_fallback = args.robot_memory_vla_qwen_cpu_fallback_kwargs\n"
        "    assert kwargs['device_map'] == 'auto'\n"
        "    assert kwargs['load_in_4bit'] is True\n"
        "    assert kwargs['low_cpu_mem_usage'] is True\n"
        "    assert kwargs['offload_folder'].endswith('/offload/qwen_full')\n"
        "    assert kwargs['max_memory'][0].endswith('GiB')\n"
        "    assert kwargs['max_memory']['cpu'].endswith('GiB')\n"
        "    assert cpu_fallback['device_map'] == 'cpu'\n"
        "    assert cpu_fallback['low_cpu_mem_usage'] is True\n"
        "    assert cpu_fallback['torch_dtype'] == 'auto'\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / 'vis',
        grounding_model='qwen-full',
        load_in_4bit=True,
    )

    adapter._get_segagent()


def test_generalvla_adapter_sets_low_vram_lisa_load_kwargs(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    root.mkdir()
    (root / "demo.py").write_text(
        "from types import SimpleNamespace\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        kwargs = args.robot_memory_vla_lisa_load_kwargs\n"
        "        cpu_fallback = args.robot_memory_vla_lisa_cpu_fallback_kwargs\n"
        "        assert kwargs['device_map'] == 'auto'\n"
        "        assert kwargs['load_in_4bit'] is True\n"
        "        assert kwargs['low_cpu_mem_usage'] is True\n"
        "        assert kwargs['offload_state_dict'] is True\n"
        "        assert kwargs['offload_folder'].endswith('/offload/lisa')\n"
        "        assert kwargs['max_memory'][0].endswith('GiB')\n"
        "        assert kwargs['max_memory']['cpu'].endswith('GiB')\n"
        "        assert cpu_fallback['device_map'] == 'cpu'\n"
        "        assert cpu_fallback['low_cpu_mem_usage'] is True\n"
        "        assert cpu_fallback['torch_dtype'] == 'auto'\n"
        "        assert cpu_fallback['offload_state_dict'] is True\n"
        "        assert cpu_fallback['offload_folder'].endswith('/offload/lisa')\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / "vis",
        grounding_model="qwen-full",
        load_in_4bit=True,
    )

    adapter._get_lisa()


def test_generalvla_adapter_sets_lisa_single_gpu_retry_kwargs(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    root.mkdir()
    (root / "demo.py").write_text(
        "from types import SimpleNamespace\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        retry_kwargs = args.robot_memory_vla_lisa_retry_load_kwargs\n"
        "        assert retry_kwargs['low_cpu_mem_usage'] is True\n"
        "        assert retry_kwargs['device_map'][''] == 0\n"
        "        assert retry_kwargs['device_map']['lm_head'] == 'cpu'\n"
        "        assert retry_kwargs['device_map']['model.text_hidden_fcs'] == 'cpu'\n"
        "        assert retry_kwargs['device_map']['model.model.embed_tokens'] == 'cpu'\n"
        "        assert retry_kwargs['offload_state_dict'] is True\n"
        "        assert retry_kwargs['offload_folder'].endswith('/offload/lisa')\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / "vis",
        grounding_model="qwen-full",
        load_in_4bit=True,
    )

    adapter._get_lisa()


def test_generalvla_adapter_uses_dynamic_cpu_budget_for_qwen_low_vram_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = GeneralVLAAdapter(
        generalvla_root="/home/why/GeneralVLA",
        vis_save_path=tmp_path / "vis",
        device="cuda:0",
        load_in_4bit=True,
    )
    monkeypatch.setattr(adapter, "_cpu_max_memory", lambda: "23GiB", raising=False)

    qwen_kwargs = adapter._build_qwen_load_kwargs()

    assert qwen_kwargs["max_memory"]["cpu"] == "23GiB"


def test_generalvla_adapter_uses_conservative_lisa_memory_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = GeneralVLAAdapter(
        generalvla_root="/home/why/GeneralVLA",
        vis_save_path=tmp_path / "vis",
        device="cuda:0",
        load_in_4bit=True,
    )
    monkeypatch.setattr(adapter, "_lisa_gpu_max_memory", lambda: "5GiB", raising=False)
    monkeypatch.setattr(adapter, "_lisa_cpu_max_memory", lambda: "10GiB", raising=False)

    lisa_kwargs = adapter._build_lisa_load_kwargs()

    assert lisa_kwargs["max_memory"][0] == "5GiB"
    assert lisa_kwargs["max_memory"]["cpu"] == "10GiB"


def test_generalvla_adapter_prefers_local_clip_vision_tower(tmp_path: Path) -> None:
    root = tmp_path / "fake_generalvla"
    clip_dir = root / "pretrain_model" / "clip-vit-large-patch14"
    clip_dir.mkdir(parents=True)
    (root / "demo.py").write_text(
        "from types import SimpleNamespace\n"
        "import numpy as np\n"
        "class LISASegmenter:\n"
        "    def __init__(self, args):\n"
        "        assert args.vision_tower.endswith('/pretrain_model/clip-vit-large-patch14')\n"
        "        self.args = args\n"
        "    def segment(self, image_path, prompt):\n"
        "        return {'text': prompt, 'masks': [np.array([[[1, 0],[0, 1]]], dtype=np.uint8)], 'image': np.zeros((2,2,3), dtype=np.uint8)}\n"
        "class SegAgentInteractiveSegmenter:\n"
        "    def __init__(self, grounding_model, seg_model, args):\n"
        "        self.args = args\n"
        "    def segment(self, image_path, initial_mask=None, prompt=None):\n"
        "        return {'mask': np.array([[[1, 1],[1, 0]]], dtype=np.uint8), 'outputs': prompt, 'image': np.zeros((2,2,3), dtype=np.uint8), 'clicks': []}\n"
        "def load_segagent_models(args):\n"
        "    return object(), object()\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(segagent_version='unused')\n",
        encoding="utf-8",
    )

    adapter = GeneralVLAAdapter(
        generalvla_root=str(root),
        vis_save_path=tmp_path / 'vis',
    )

    result = adapter.segment_pick_object(np.zeros((2, 2, 3), dtype=np.uint8), "抓起瓶盖", "瓶盖")
    assert result.text_response == "瓶盖"


def test_generalvla_adapter_default_lisa_version_uses_7b_explanatory(tmp_path: Path) -> None:
    adapter = GeneralVLAAdapter(
        generalvla_root=str(tmp_path / "GeneralVLA"),
        vis_save_path=tmp_path / "vis",
    )

    assert str(adapter._default_lisa_version()).endswith("/LISA-7B-v1-explanatory")


def test_generalvla_adapter_sets_expandable_cuda_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    adapter = GeneralVLAAdapter(
        generalvla_root="/home/why/GeneralVLA",
        vis_save_path=Path("/tmp/vis"),
        device="cuda:0",
    )

    adapter._ensure_cuda_allocator_settings()

    assert "expandable_segments:True" in __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"]
