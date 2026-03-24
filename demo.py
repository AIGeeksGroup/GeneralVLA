import argparse
import os
import sys
import traceback

sys.path.append("./third_party/SimpleClick")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor



from evaltools.model_loader import load_model

# LISA imports
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

# SegAgent / Interactive segmentation imports
import pycocotools.mask as mask_util
from pycocotools import mask as maskUtils


from evaltools.visual_utils import visualize_mask_and_point
from third_party.SimpleClick.isegm.inference.clicker import Click, Clicker
HAS_INTERACTIVE = True



# ============================================================================
# Configuration Constants
# ============================================================================

# Fixed prompt for Stage 2 - graspable region segmentation
GRASPABLE_PROMPT = "Segment the graspable part of the object in the image (e.g., the upper part of the bottle)"


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args(args):
    """Parse command line arguments for both LISA and SegAgent models."""
    parser = argparse.ArgumentParser(
        description="Combined LISA + SegAgent Pipeline"
    )
    
    # LISA model configuration
    parser.add_argument("--lisa_version", 
                        default="/root/autodl-tmp/LISA/pretrain_model/LISA-13B-llama2-v1-explanatory",
                        help="LISA model path")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str,
                        help="Output directory for visualizations")
    parser.add_argument("--precision", default="fp16", type=str,
                        choices=["fp32", "bf16", "fp16"],
                        help="Model precision for inference")
    parser.add_argument("--image_size", default=1024, type=int,
                        help="Input image size for SAM encoder")
    parser.add_argument("--model_max_length", default=512, type=int,
                        help="Maximum token length")
    parser.add_argument("--lora_r", default=8, type=int,
                        help="LoRA rank parameter")
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14",
                        type=str, help="CLIP vision encoder model")
    parser.add_argument("--local-rank", default=0, type=int,
                        help="GPU device rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Enable 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Enable 4-bit quantization")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True,
                        help="Use multimodal start/end tokens")
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"],
                        help="Conversation template type")
    
    # SegAgent model configuration
    parser.add_argument("--segagent_version",
                        default="/root/autodl-tmp/LISA/pretrain_model/segagent/SegAgent-Model",
                        help="SegAgent model path")
    parser.add_argument("--n_clicks", default=5, type=int,
                        help="Number of clicks for interactive segmentation")
    parser.add_argument("--use_previous_mask", action="store_true", default=True,
                        help="Use previous mask as prior for refinement")
    parser.add_argument("--undo_radius", default=10, type=int,
                        help="Radius for click undo operation")
    parser.add_argument("--skip_stage2", action="store_true", default=False,
                        help="Skip interactive segmentation stage")
    
    # SegAgent specific args (from run_eval.sh)
    parser.add_argument("--grounding_model", default="point", type=str,
                        help="Grounding model type")
    parser.add_argument("--use_gt_box", action="store_true", default=False,
                        help="Use ground truth box")
    parser.add_argument("--only_use_gt_box", action="store_true", default=False,
                        help="Only use ground truth box")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SimpleClick ")

    parser.add_argument("--seg_model", type=str, default="simple_click",
                        help="Segmentation model type")
    parser.add_argument("--model", type=str, default=None,
                        help="Grounding model name or path")

    parser.add_argument("--config_path", type=str, 
                        default="./third_party/SimpleClick/config.yml",
                        help="SimpleClick config file path")

    parser.add_argument("--exp_path", type=str, default="",
                        help="")
    
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="")
    
    parser.add_argument("--eval_ritm", action="store_true", default=False,
                        help="")
    parser.add_argument("--clicks_limit", type=int, default="7", 
                        help="")
    parser.add_argument("--eval_mode", type=str, default="cvpr",
                        choices=["cvpr", "fixed400", "fixed600"],
                        help="Evaluation mode (cvpr is recommended)")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Click probability threshold")
    parser.add_argument("--mode", type=str, default="NoBRS",
                        help="Inference mode (NoBRS, RGB-BRS, etc.)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Enable visualization in predictor")
    parser.add_argument("--record_trace", action="store_true", default=False,
                        help="Record click trace")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for processing")
    parser.add_argument("--end_index", type=int, default=-1,
                        help="End index for processing")
    
    parser.add_argument("--use_mask_module", action="store_true", default=False,
                        help="")
    
    return parser.parse_args(args)


# ============================================================================
# Utility Functions
# ============================================================================

def preprocess_for_sam(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """
    Normalize and pad image to square for SAM encoder.
    
    Args:
        x: Input tensor (C, H, W)
        pixel_mean: Normalization mean
        pixel_std: Normalization std
        img_size: Target square size
    
    Returns:
        Preprocessed tensor
    """
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def convert_mask_to_coco(mask):
    """
    Convert binary mask to COCO RLE format.
    
    Args:
        mask: Binary mask (numpy array or torch tensor)
    
    Returns:
        COCO RLE dictionary
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.astype(np.uint8)
    rle = mask_util.encode(np.array(mask, order="F"))
    if isinstance(rle, list):
        rle = mask_util.merge(rle)
    if isinstance(rle, dict):
        rle = {"counts": rle["counts"].decode("utf-8"), "size": rle["size"]}
    return rle


def create_overlay_visualization(image_rgb, mask, color=(255, 0, 0), alpha=0.5):
    """
    Create overlay visualization of mask on image.
    
    Args:
        image_rgb: RGB image numpy array
        mask: Binary mask
        color: Overlay color (R, G, B)
        alpha: Transparency factor
    
    Returns:
        Overlay image (RGB)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    
    mask_bool = mask > 0
    overlay = image_rgb.copy()
    
    # Apply colored overlay
    overlay[mask_bool] = (
        image_rgb[mask_bool] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay


# ============================================================================
# LISA Model Class
# ============================================================================

class LISASegmenter:
    """
    LISA reasoning segmentation model wrapper.
    """
    
    def __init__(self, args):
        """Initialize LISA model."""
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")
        
        print("[LISA] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.lisa_version,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.seg_token_idx = self.tokenizer(
            "[SEG]", add_special_tokens=False
        ).input_ids[0]
        
        # Determine dtype
        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.half
        
        kwargs = {"torch_dtype": torch_dtype}
        
        # Quantization configuration
        if args.load_in_4bit:
            kwargs.update({
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            })
        elif args.load_in_8bit:
            kwargs.update({
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            })
        
        print("[LISA] Loading model...")
        self.model = LISAForCausalLM.from_pretrained(
            args.lisa_version,
            low_cpu_mem_usage=True,
            vision_tower=args.vision_tower,
            seg_token_idx=self.seg_token_idx,
            **kwargs
        )
        
        # Set special token ids
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Initialize vision modules
        self.model.get_model().initialize_vision_modules(
            self.model.get_model().config
        )
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
        
        # Move to device with appropriate precision
        if args.precision == "bf16":
            self.model = self.model.bfloat16().cuda()
        elif args.precision == "fp16" and not args.load_in_4bit and not args.load_in_8bit:
            vision_tower = self.model.get_model().get_vision_tower()
            self.model.model.vision_tower = None
            import deepspeed
            model_engine = deepspeed.init_inference(
                model=self.model,
                dtype=torch.half,
                replace_with_kernel_inject=True,
                replace_method="auto",
            )
            self.model = model_engine.module
            self.model.model.vision_tower = vision_tower.half().cuda()
        elif args.precision == "fp32":
            self.model = self.model.float().cuda()
        
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(device=self.device)
        
        # Image processors
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            self.model.config.vision_tower
        )
        self.sam_transform = ResizeLongestSide(args.image_size)
        
        self.model.eval()
        print("[LISA] Model loaded successfully.")
    
    def segment(self, image_path, prompt):
        """
        Run LISA segmentation on image with text prompt.
        
        Args:
            image_path: Path to input image
            prompt: Text description of region to segment
        
        Returns:
            dict with text output, masks, and original image
        """
        args = self.args
        
        # Build conversation prompt
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        
        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            full_prompt = full_prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], "")
        full_prompt = conv.get_prompt()
        
        # Load and preprocess image
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_size = [image_rgb.shape[:2]]
        
        # CLIP preprocessing
        clip_input = self.clip_processor.preprocess(
            image_rgb, return_tensors="pt"
        )["pixel_values"][0].unsqueeze(0).cuda()
        
        if args.precision == "bf16":
            clip_input = clip_input.bfloat16()
        elif args.precision == "fp16":
            clip_input = clip_input.half()
        else:
            clip_input = clip_input.float()
        
        # SAM preprocessing
        sam_image = self.sam_transform.apply_image(image_rgb)
        resize_list = [sam_image.shape[:2]]
        
        sam_input = preprocess_for_sam(
            torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        ).unsqueeze(0).cuda()
        
        if args.precision == "bf16":
            sam_input = sam_input.bfloat16()
        elif args.precision == "fp16":
            sam_input = sam_input.half()
        else:
            sam_input = sam_input.float()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, return_tensors="pt"
        ).unsqueeze(0).cuda()
        
        # Inference
        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(
                clip_input,
                sam_input,
                input_ids,
                resize_list,
                original_size,
                max_new_tokens=512,
                tokenizer=self.tokenizer,
            )
        
        # Decode text output
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        
        return {
            'text': text_output,
            'masks': pred_masks,
            'image': image_rgb
        }


# ============================================================================
# SegAgent Interactive Segmentation Class (from code2)
# ============================================================================

class SegAgentInteractiveSegmenter:
    """
    SegAgent-based interactive click segmentation.
    Adapted from REFCOCOG_EVAL in code2.
    """
    
    def __init__(self, grounding_model, seg_model, args):
        """
        Initialize SegAgent interactive segmenter.
        
        Args:
            grounding_model: Model for processing prompts and predicting clicks
            seg_model: Segmentation model (SimpleClick)
            args: Configuration arguments
        """
        self.begin_str = "<image>\nThis provides an overview of the picture.\n"
        self.grounding_model = grounding_model
        self.seg_model = seg_model
        self.args = args
        self.workspace = os.environ.get("VIS_DIR", os.getcwd())
        self.image_processor = (
            self.grounding_model.image_processor
            if hasattr(self.grounding_model, "image_processor")
            else None
        )
        self.tokenizer = (
            self.grounding_model.tokenizer
            if hasattr(self.grounding_model, "tokenizer")
            else None
        )
    
    def segment(self, image_path, initial_mask=None, prompt=None):
        """
        Run interactive segmentation with click-based refinement.
        
        Args:
            image_path: Path to input image
            initial_mask: Optional initial mask from LISA stage
            prompt: Text prompt (uses GRASPABLE_PROMPT if None)
        
        Returns:
            dict with final mask, clicks info, and image
        """
        if prompt is None:
            prompt = GRASPABLE_PROMPT
        
        # Load image
        image_cv = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        h, w = image_cv.shape[:2]
        
        # Prepare initial mask
        if initial_mask is not None:
            if isinstance(initial_mask, torch.Tensor):
                gt_mask = initial_mask.cpu().numpy()
            else:
                gt_mask = initial_mask.copy()
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Build data item structure expected by grounding model
        data_item = {
            "img_path": image_path,
            "height": h,
            "width": w,
            "caption": [prompt],
            "gt_mask": gt_mask,
            "masks": None,
            "annotation": [{"segmentation": None, "id": 0, "click_id": 0}],
            "relative_coor": [0, 0, 0, 0],
            "pred_list": []
        }
        
        # Process image for SimpleClick model
        simple_click_image = self.seg_model.image_process(img_path=image_path)
        
        clicks_info = []
        
        with torch.no_grad():
            self.seg_model.set_input_image(simple_click_image)
            
            clicker = Clicker()
            previous_mask = None
            if initial_mask is not None:
                previous_mask = torch.from_numpy(gt_mask).to(torch.uint8).unsqueeze(0)
            pred_logits = None
            last_ref_box_str = None
            gt_box2 = None
            
            for i in range(self.args.n_clicks):
                click_id = i
                last_pred_logits = pred_logits if self.args.use_previous_mask else None
                
                # Build prompt and get response from grounding model
                prompt_text, conv = self.grounding_model.build_prompt(
                    data_item, last_ref_box_str
                )
                outputs = self.grounding_model.generate_response(
                    prompt_text, image_path, 
                    previous_mask.numpy() if previous_mask is not None else None, 
                    conv
                )
                
                if last_ref_box_str is not None:
                    outputs = last_ref_box_str + outputs
                
                data_item["outputs"] = outputs
                
                # Parse response to get click coordinates
                if "box" in self.args.grounding_model:
                    is_positive, points, gt_box = self.grounding_model.process_response(outputs)
                    scale = 999
                    gt_box = [b / scale for b in gt_box]
                    gt_box2 = [
                        int(gt_box[0] * w), int(gt_box[1] * h),
                        int(gt_box[2] * w), int(gt_box[3] * h),
                    ]
                else:
                    is_positive, points = self.grounding_model.process_response(outputs)
                
                # Convert relative coords to absolute
                abs_coords = (round(points[0] * h), round(points[1] * w))
                
                # Add click to clicker
                click = Click(is_positive=is_positive, coords=abs_coords)
                clicker.add_click(click, self.args.undo_radius)
                
                # Get segmentation prediction
                pred_result = self.seg_model.get_prediction(
                    clicker, box=gt_box2, mask=last_pred_logits
                )
                
                if isinstance(pred_result, tuple):
                    pred_mask, pred_logits = pred_result
                else:
                    pred_mask = pred_result
                
                previous_mask = torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                
                # Record click info
                clicks_info.append({
                    "click_id": click_id,
                    "is_positive": is_positive,
                    "coords": abs_coords,
                    "used_box": gt_box2,
                    "outputs": outputs
                })
                
                # Update pred_list for next iteration
                click_info_dict = {
                    "click_id": click_id,
                    "clicks": {"is_positive": is_positive, "coords": abs_coords},
                    "used_box": gt_box2,
                    "mask": convert_mask_to_coco(previous_mask),
                    "outputs": outputs
                }
                data_item["pred_list"].append(click_info_dict)
                
                print(f"  Click {i+1}/{self.args.n_clicks}: "
                      f"{'Positive' if is_positive else 'Negative'} at {abs_coords}")
        
        return {
            'mask': previous_mask,
            'clicks': clicks_info,
            'outputs': outputs,
            'image': image_rgb
        }


# ============================================================================
# Output Saving Functions
# ============================================================================

def save_stage1_outputs(result, save_dir, image_name):
    """Save LISA (Stage 1) outputs."""
    os.makedirs(save_dir, exist_ok=True)
    
    pred_masks = result['masks']
    image_rgb = result['image']
    
    if len(pred_masks) == 0 or pred_masks[0].shape[0] == 0:
        print("[Stage 1] No valid mask generated.")
        return None, None, None
    
    # Get first mask
    mask = pred_masks[0].detach().cpu().numpy()[0]
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Save binary mask
    mask_path = os.path.join(save_dir, f"{image_name}_stage1_mask.jpg")
    cv2.imwrite(mask_path, binary_mask * 255)
    print(f"[Stage 1] Binary mask saved: {mask_path}")
    
    # Save overlay visualization (red overlay)
    overlay = create_overlay_visualization(image_rgb, binary_mask, color=(255, 0, 0))
    overlay_path = os.path.join(save_dir, f"{image_name}_stage1_overlay.jpg")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[Stage 1] Overlay saved: {overlay_path}")
    
    return mask_path, overlay_path, binary_mask


def save_stage2_outputs(result, save_dir, image_name, prompt):
    """Save interactive segmentation (Stage 2) outputs."""
    os.makedirs(save_dir, exist_ok=True)
    
    mask = result['mask']
    image_rgb = result['image']
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Save binary mask
    mask_path = os.path.join(save_dir, f"{image_name}_stage2_mask.jpg")
    cv2.imwrite(mask_path, binary_mask * 255)
    print(f"[Stage 2] Binary mask saved: {mask_path}")
    
    # Save overlay visualization (green overlay)
    overlay = create_overlay_visualization(image_rgb, binary_mask, color=(0, 255, 0))
    overlay_path = os.path.join(save_dir, f"{image_name}_stage2_overlay.jpg")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[Stage 2] Overlay saved: {overlay_path}")
    
    # Save combined visualization with click markers
    combined = overlay.copy()
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    # Add prompt text
    prompt_display = prompt[:80] + "..." if len(prompt) > 80 else prompt
    cv2.putText(
        combined_bgr, f"Prompt: {prompt_display}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    
    # Add click markers
    for i, click in enumerate(result['clicks']):
        y, x = click['coords']
        color = (0, 255, 0) if click['is_positive'] else (0, 0, 255)
        cv2.circle(combined_bgr, (x, y), 5, color, -1)
        cv2.putText(
            combined_bgr, str(i+1),
            (x+8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    
    combined_path = os.path.join(save_dir, f"{image_name}_stage2_combined.jpg")
    cv2.imwrite(combined_path, combined_bgr)
    print(f"[Stage 2] Combined visualization saved: {combined_path}")
    
    return mask_path, overlay_path


# ============================================================================
# Model Loading Helper (for SegAgent)
# ============================================================================

def load_segagent_models(args):
    """
    Load SegAgent grounding model and segmentation model.
    
    This function should be adapted based on your actual SegAgent implementation.
    The loading logic is based on run_eval.sh patterns.
    
    Args:
        args: Parsed arguments containing segagent_version path
    
    Returns:
        tuple: (grounding_model, seg_model)
    """
    # Import SegAgent-specific modules
    # These imports depend on your actual project structure

    seg_model, grounding_model = load_model(args)
    return grounding_model, seg_model



# ============================================================================
# Main Function
# ============================================================================

def main(args):
    """
    Main pipeline execution.
    
    Stage 1: LISA reasoning segmentation with user prompt
    Stage 2: SegAgent interactive click-based segmentation for graspable regions
    """
    args = parse_args(args)
    args.model = args.segagent_version
    os.makedirs(args.vis_save_path, exist_ok=True)
    
    # ========================================================================
    # Initialize LISA Model (Stage 1)
    # ========================================================================
    print("=" * 60)
    print("Initializing LISA Model (Stage 1)")
    print("=" * 60)
    lisa = LISASegmenter(args)
    
    # ========================================================================
    # Initialize SegAgent Models (Stage 2)
    # ========================================================================
    segagent_segmenter = None
    if HAS_INTERACTIVE and not args.skip_stage2:
        print("\n" + "=" * 60)
        print("Initializing SegAgent Models (Stage 2)")
        print("=" * 60)
        
        grounding_model, seg_model = load_segagent_models(args)
        
        if grounding_model is not None and seg_model is not None:
            segagent_segmenter = SegAgentInteractiveSegmenter(
                grounding_model, seg_model, args
            )
        else:
            print("[Warning] SegAgent models not loaded. Stage 2 will be skipped.")
    
    # ========================================================================
    # Main Interaction Loop
    # ========================================================================
    while True:
        print("\n" + "=" * 60)
        print("LISA + SegAgent Combined Pipeline")
        print("=" * 60)
        
        # ====================================================================
        # Stage 1: LISA Segmentation
        # ====================================================================
        print("\n[Stage 1] LISA Reasoning Segmentation")
        print("-" * 50)
        
        prompt = input("Enter prompt (or 'q' to quit): ").strip()
        if prompt.lower() == 'q':
            print("Exiting...")
            break
        
        image_path = input("Enter image path: ").strip()
        image_path = image_path.replace("'", "").replace('"', "")
        
        if not os.path.exists(image_path):
            print(f"[Error] File not found: {image_path}")
            continue
        
        try:
            # Run LISA
            print("\n[Stage 1] Running LISA inference...")
            lisa_result = lisa.segment(image_path, prompt)
            print(f"[Stage 1] Text output: {lisa_result['text']}")
            
            # Save Stage 1 outputs
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            stage1_dir = os.path.join(args.vis_save_path, "stage1_lisa")
            mask_path, overlay_path, binary_mask = save_stage1_outputs(
                lisa_result, stage1_dir, image_name
            )
            
            if binary_mask is None:
                print("[Stage 1] No valid mask. Skipping Stage 2.")
                continue
            
            # ================================================================
            # Stage 2: SegAgent Interactive Segmentation
            # ================================================================
            if args.skip_stage2:
                print("\n[Stage 2] Skipped (--skip_stage2 flag)")
                continue
            
            if segagent_segmenter is None:
                print("\n[Stage 2] SegAgent not available. Only Stage 1 outputs saved.")
                continue
            
            print("\n[Stage 2] SegAgent Interactive Segmentation")
            print("-" * 50)
            # print(f"Fixed prompt: {GRASPABLE_PROMPT}")
            print(f"Prompt: {prompt}")
            
            
            # Choose input source for Stage 2
            print("\nSelect input for Stage 2:")
            print("  1. Original image")
            print("  2. Stage 1 masked overlay")
            print("  3. Stage 1 binary mask")
            choice = input("Enter choice (1/2/3) [default=1]: ").strip() or "1"
            
            if choice == '2' and overlay_path:
                stage2_input = overlay_path
            elif choice == '3' and mask_path:
                stage2_input = mask_path
            else:
                stage2_input = image_path
            
            # Use LISA mask as initialization?
            use_lisa_init = input(
                "Use LISA mask as initial mask? (y/n) [default=n]: "
            ).strip().lower() == 'y'
            
            initial_mask = binary_mask if use_lisa_init else None
            
            # Run SegAgent interactive segmentation
            print(f"\n[Stage 2] Running with input: {stage2_input}")
            interactive_result = segagent_segmenter.segment(
                stage2_input,
                initial_mask=initial_mask,
                prompt=prompt
            )
            
            # Save Stage 2 outputs
            stage2_dir = os.path.join(args.vis_save_path, "stage2_segagent")
            save_stage2_outputs(
                interactive_result, stage2_dir, image_name, prompt
            )
            
        except Exception as e:
            print(f"[Error] {str(e)}")
            traceback.print_exc()
            continue
        
        print("\n" + "-" * 60)
        print(f"All outputs saved to: {args.vis_save_path}")
        print(f"  Stage 1 (LISA): {args.vis_save_path}/stage1_lisa/")
        print(f"  Stage 2 (SegAgent): {args.vis_save_path}/stage2_segagent/")


if __name__ == "__main__":
    main(sys.argv[1:])
