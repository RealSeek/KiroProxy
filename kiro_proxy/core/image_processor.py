"""图片处理模块 - 压缩、缩放、GIF 抽帧

1:1 对应 kiro.rs src/image.rs 的逻辑：
- 静态图：缩放 + 大文件重编码(>200KB)
- GIF：抽帧（两遍扫描，≤20帧 ≤5fps，转 JPEG）
"""
import io
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ==================== 常量 ====================

MAX_IMAGE_BYTES = 200_000       # 200KB，超过此大小触发重编码
GIF_MAX_OUTPUT_FRAMES = 20      # GIF 最大输出帧数
GIF_MAX_FPS = 5                 # GIF 最大帧率
GIF_MIN_FRAME_DELAY = 10        # GIF 最小帧延迟 (ms)


# ==================== 数据结构 ====================

@dataclass
class ImageConfig:
    """图片处理配置"""
    enabled: bool = True
    image_max_long_edge: int = 4000
    image_max_pixels_single: int = 4_000_000
    image_max_pixels_multi: int = 4_000_000
    image_multi_threshold: int = 20

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "image_max_long_edge": self.image_max_long_edge,
            "image_max_pixels_single": self.image_max_pixels_single,
            "image_max_pixels_multi": self.image_max_pixels_multi,
            "image_multi_threshold": self.image_multi_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageConfig":
        return cls(
            enabled=data.get("enabled", True),
            image_max_long_edge=data.get("image_max_long_edge", 4000),
            image_max_pixels_single=data.get("image_max_pixels_single", 4_000_000),
            image_max_pixels_multi=data.get("image_max_pixels_multi", 4_000_000),
            image_multi_threshold=data.get("image_multi_threshold", 20),
        )

@dataclass
class ImageProcessResult:
    """图片处理结果"""
    data: str                # base64
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    tokens: int
    was_resized: bool
    was_reencoded: bool
    original_bytes_len: int
    final_bytes_len: int


@dataclass
class GifSamplingResult:
    """GIF 抽帧结果"""
    frames: List[ImageProcessResult]
    duration_ms: int
    source_frames: int
    sampling_interval_ms: int
    output_format: str = "jpeg"


# ==================== 全局配置 ====================

_image_config = ImageConfig()


def get_image_config() -> ImageConfig:
    """获取图片处理配置"""
    return _image_config


def set_image_config(config: ImageConfig):
    """设置图片处理配置"""
    global _image_config
    _image_config = config


def update_image_config(data: dict):
    """更新图片处理配置"""
    global _image_config
    _image_config = ImageConfig.from_dict(data)


# ==================== 核心函数 ====================

def calculate_tokens(width: int, height: int) -> int:
    """计算图片 token 数量"""
    return (width * height + 375) // 750


def apply_scaling_rules(w: int, h: int, max_long_edge: int, max_pixels: int) -> Tuple[int, int]:
    """两级缩放：长边限制 → 总像素限制

    Returns:
        (new_w, new_h) 缩放后的尺寸，如果不需要缩放则返回原尺寸
    """
    new_w, new_h = w, h

    # 第一级：长边限制
    long_edge = max(new_w, new_h)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        new_w = max(1, int(new_w * scale))
        new_h = max(1, int(new_h * scale))

    # 第二级：总像素限制
    pixels = new_w * new_h
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        new_w = max(1, int(new_w * scale))
        new_h = max(1, int(new_h * scale))

    return new_w, new_h


def encode_image(img, fmt: str) -> str:
    """PIL Image → base64 字符串"""
    buf = io.BytesIO()
    save_fmt = fmt.upper()
    if save_fmt == "JPG":
        save_fmt = "JPEG"

    save_kwargs = {}
    if save_fmt == "JPEG":
        save_kwargs["quality"] = 85
        # JPEG 不支持 alpha 通道
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
    elif save_fmt == "PNG":
        save_kwargs["optimize"] = True

    img.save(buf, format=save_fmt, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def process_image(
    base64_data: str,
    fmt: str,
    config: ImageConfig,
    image_count: int = 1,
) -> Optional[ImageProcessResult]:
    """处理静态图片：缩放 + 大文件重编码

    对应 kiro.rs process_image()
    """
    from PIL import Image

    try:
        raw_bytes = base64.b64decode(base64_data)
        original_bytes_len = len(raw_bytes)
        img = Image.open(io.BytesIO(raw_bytes))
        original_w, original_h = img.size
    except Exception as e:
        print(f"[ImageProcessor] Failed to decode image: {e}")
        return None

    # 选择像素上限：多图模式 vs 单图模式
    max_pixels = (
        config.image_max_pixels_multi
        if image_count >= config.image_multi_threshold
        else config.image_max_pixels_single
    )

    # 计算缩放目标
    target_w, target_h = apply_scaling_rules(
        original_w, original_h,
        config.image_max_long_edge, max_pixels,
    )
    needs_resize = (target_w != original_w or target_h != original_h)

    # GIF 格式强制重编码为 PNG（静态帧）
    output_fmt = "png" if fmt == "gif" else fmt
    is_gif = fmt == "gif"

    # 判断是否需要处理
    needs_reencode = (
        original_bytes_len > MAX_IMAGE_BYTES
        or is_gif
    )

    if not needs_resize and not needs_reencode:
        # 不需要任何处理，直接返回原始数据
        tokens = calculate_tokens(original_w, original_h)
        return ImageProcessResult(
            data=base64_data,
            original_size=(original_w, original_h),
            final_size=(original_w, original_h),
            tokens=tokens,
            was_resized=False,
            was_reencoded=False,
            original_bytes_len=original_bytes_len,
            final_bytes_len=original_bytes_len,
        )

    try:
        # 需要处理：加载像素数据
        img.load()

        if needs_resize:
            img = img.resize((target_w, target_h), Image.LANCZOS)
            print(f"[ImageProcessor] Resized: {original_w}x{original_h} -> {target_w}x{target_h}")

        final_w, final_h = img.size
        encoded = encode_image(img, output_fmt)
        final_bytes_len = len(base64.b64decode(encoded))
        tokens = calculate_tokens(final_w, final_h)

        if needs_reencode and not needs_resize:
            ratio = final_bytes_len / original_bytes_len * 100 if original_bytes_len > 0 else 0
            print(f"[ImageProcessor] Re-encoded: {original_bytes_len} -> {final_bytes_len} bytes ({ratio:.0f}%)")

        return ImageProcessResult(
            data=encoded,
            original_size=(original_w, original_h),
            final_size=(final_w, final_h),
            tokens=tokens,
            was_resized=needs_resize,
            was_reencoded=True,
            original_bytes_len=original_bytes_len,
            final_bytes_len=final_bytes_len,
        )
    except Exception as e:
        print(f"[ImageProcessor] Failed to process image: {e}")
        return None


def process_gif_frames(
    base64_data: str,
    config: ImageConfig,
    image_count: int = 1,
) -> Optional[GifSamplingResult]:
    """GIF 抽帧：两遍扫描，≤20帧 ≤5fps，转 JPEG

    对应 kiro.rs process_gif_frames()

    第一遍：扫描所有帧获取总时长和帧数
    第二遍：按采样间隔提取帧并处理
    """
    from PIL import Image, ImageSequence

    try:
        raw_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        print(f"[ImageProcessor] Failed to decode GIF: {e}")
        return None

    if not hasattr(img, "n_frames") or img.n_frames <= 1:
        # 单帧 GIF，当作静态图处理
        return None

    try:
        # 第一遍：扫描帧信息
        total_duration_ms = 0
        source_frames = 0
        frame_durations = []

        for frame in ImageSequence.Iterator(img):
            duration = frame.info.get("duration", 100)
            if duration < GIF_MIN_FRAME_DELAY:
                duration = 100  # 异常值修正
            frame_durations.append(duration)
            total_duration_ms += duration
            source_frames += 1

        if source_frames == 0:
            return None

        # 计算采样间隔
        # 目标：最多 GIF_MAX_OUTPUT_FRAMES 帧，最多 GIF_MAX_FPS fps
        min_interval_by_count = (
            total_duration_ms / GIF_MAX_OUTPUT_FRAMES
            if GIF_MAX_OUTPUT_FRAMES > 0 else 0
        )
        min_interval_by_fps = 1000.0 / GIF_MAX_FPS
        sampling_interval_ms = max(min_interval_by_count, min_interval_by_fps)
        sampling_interval_ms = max(sampling_interval_ms, 1.0)

        # 第二遍：按采样间隔提取帧
        img.seek(0)
        frames = []
        elapsed_ms = 0.0
        next_sample_at = 0.0

        max_pixels = (
            config.image_max_pixels_multi
            if image_count >= config.image_multi_threshold
            else config.image_max_pixels_single
        )

        for idx, frame in enumerate(ImageSequence.Iterator(img)):
            duration = frame_durations[idx] if idx < len(frame_durations) else 100

            if elapsed_ms >= next_sample_at:
                # 采样此帧
                frame_rgb = frame.convert("RGB")
                fw, fh = frame_rgb.size

                target_w, target_h = apply_scaling_rules(
                    fw, fh, config.image_max_long_edge, max_pixels,
                )
                was_resized = (target_w != fw or target_h != fh)
                if was_resized:
                    frame_rgb = frame_rgb.resize((target_w, target_h), Image.LANCZOS)

                encoded = encode_image(frame_rgb, "jpeg")
                final_bytes_len = len(base64.b64decode(encoded))
                final_w, final_h = target_w, target_h
                tokens = calculate_tokens(final_w, final_h)

                frames.append(ImageProcessResult(
                    data=encoded,
                    original_size=(fw, fh),
                    final_size=(final_w, final_h),
                    tokens=tokens,
                    was_resized=was_resized,
                    was_reencoded=True,
                    original_bytes_len=0,  # GIF 帧无独立字节数
                    final_bytes_len=final_bytes_len,
                ))
                next_sample_at += sampling_interval_ms

                if len(frames) >= GIF_MAX_OUTPUT_FRAMES:
                    break

            elapsed_ms += duration

        total_final_bytes = sum(f.final_bytes_len for f in frames)
        print(
            f"[ImageProcessor] GIF sampled: {source_frames} frames -> {len(frames)} frames, "
            f"duration={total_duration_ms}ms, interval={sampling_interval_ms:.0f}ms, "
            f"total_bytes={total_final_bytes}"
        )

        return GifSamplingResult(
            frames=frames,
            duration_ms=total_duration_ms,
            source_frames=source_frames,
            sampling_interval_ms=int(sampling_interval_ms),
            output_format="jpeg",
        )
    except Exception as e:
        print(f"[ImageProcessor] Failed to process GIF frames: {e}")
        return None


def process_image_to_format(
    base64_data: str,
    output_format: str,
    config: ImageConfig,
    image_count: int = 1,
) -> Optional[ImageProcessResult]:
    """强制转格式处理（GIF fallback 用）

    对应 kiro.rs process_image_to_format()
    """
    from PIL import Image

    try:
        raw_bytes = base64.b64decode(base64_data)
        original_bytes_len = len(raw_bytes)
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
        original_w, original_h = img.size
    except Exception as e:
        print(f"[ImageProcessor] Failed to decode image for format conversion: {e}")
        return None

    max_pixels = (
        config.image_max_pixels_multi
        if image_count >= config.image_multi_threshold
        else config.image_max_pixels_single
    )

    target_w, target_h = apply_scaling_rules(
        original_w, original_h,
        config.image_max_long_edge, max_pixels,
    )
    needs_resize = (target_w != original_w or target_h != original_h)

    try:
        if needs_resize:
            img = img.resize((target_w, target_h), Image.LANCZOS)

        if output_format == "jpeg":
            img = img.convert("RGB")

        final_w, final_h = img.size
        encoded = encode_image(img, output_format)
        final_bytes_len = len(base64.b64decode(encoded))
        tokens = calculate_tokens(final_w, final_h)

        return ImageProcessResult(
            data=encoded,
            original_size=(original_w, original_h),
            final_size=(final_w, final_h),
            tokens=tokens,
            was_resized=needs_resize,
            was_reencoded=True,
            original_bytes_len=original_bytes_len,
            final_bytes_len=final_bytes_len,
        )
    except Exception as e:
        print(f"[ImageProcessor] Failed to convert image format: {e}")
        return None