"""Coordinate restoration for gradients of Qwen's packed visual inputs."""

def restore_qwen_video_grid(scores, grid_shape, processor):
    """Undo [T,H/m,W/m,m,m] packing before any spatial pooling.

    The processor that actually packs video pixels owns merge_size. Older Qwen
    processors use image_processor for videos. Do not silently guess a layout.
    """
    visual_processor = getattr(processor, "video_processor", None)
    if visual_processor is None:
        visual_processor = getattr(processor, "image_processor", None)
    merge_size = getattr(visual_processor, "merge_size", None)
    if not isinstance(merge_size, int) or isinstance(merge_size, bool) or merge_size < 1:
        raise ValueError("Qwen video credit requires the processor's positive integer merge_size")
    t, h, w = map(int, grid_shape)
    if min(t, h, w) < 1 or h % merge_size or w % merge_size:
        raise ValueError(f"Invalid Qwen video grid {grid_shape} for merge_size={merge_size}")
    if scores.numel() != t * h * w:
        raise ValueError(f"Expected {t*h*w} patch scores, got {scores.numel()}")
    return (
        scores.reshape(t, h // merge_size, w // merge_size, merge_size, merge_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(t, h, w)
    )
