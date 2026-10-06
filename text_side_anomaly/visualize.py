"""异常图的可视化：稳健标定 + 引导滤波上采样。

**纯显示层，不参与任何指标计算。**
前面已实测：逐图 z-score / min-max 会破坏跨图可比性（Dice 0.5535 → 0.4012 / 0.3730），
所以标定只能用在出图上，绝不能回灌到评估口径里。

两件事：
1. 标定 —— 把 amap 映射到 [0,1]。默认用「在一批正常切片上拟合出的全局分位数」，
   保证正常切片不会因为逐图拉伸而凭空发亮（医学场景这点很重要）。
2. 上采样 —— 14×14 双线性放大到 224 必然糊。改用引导滤波（以原图为引导），
   让异常边界贴着解剖结构走，这是 paclip 的做法。
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from .postprocess import guided_filter


def upsample_guided(amap: np.ndarray, guide: np.ndarray,
                    radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """14×14 异常图 → 引导滤波上采样到 guide 的尺寸。

    先双线性放大（补空间连续性），再用原图作引导做保边平滑（把边界贴回解剖结构）。
    """
    from PIL import Image

    h, w = guide.shape
    up = np.asarray(
        Image.fromarray(amap.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )
    g = np.asarray(guide, dtype=np.float32)
    lo, hi = float(g.min()), float(g.max())
    g = (g - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(g)
    return guided_filter(g, up, radius=radius, eps=eps)


# ---------------------------------------------------------------------- #
# 标定
# ---------------------------------------------------------------------- #
class AmapCalibration:
    """异常图 → [0, 1] 的稳健标定。

    用分位数而不是 min/max：局部极值不会再把整张图的色阶吃光。
    fit() 在正常切片上拟合即可，部署时不需要病灶标注。
    """

    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        self.lo, self.hi = lo, hi

    @classmethod
    def fit_from_normals(cls, amaps: np.ndarray,
                         lo_pct: float = 50.0, hi_pct: float = 99.5) -> "AmapCalibration":
        """在一批正常切片（或任意参考图）的 amap 上标定。

        取 P50 作下界（正常组织对应色阶 0）、P99.5 作上界，留出对异常的高动态范围。
        """
        flat = np.asarray(amaps, dtype=np.float64).reshape(-1)
        lo = float(np.percentile(flat, lo_pct))
        hi = float(np.percentile(flat, hi_pct))
        return cls(lo, hi if hi - lo > 1e-6 else lo + 1e-6)

    @classmethod
    def fit_from_stats(cls, amaps: np.ndarray,
                       lo_pct: float = 50.0, hi_pct: float = 99.5) -> "AmapCalibration":
        """无正常切片时：在混合分布上用分位数标定，同样稳健。

        lo_pct/hi_pct 可调，便于按数据集调「多冷/多热」；默认值与首版实现一致。
        """
        flat = np.asarray(amaps, dtype=np.float64).reshape(-1)
        lo = float(np.percentile(flat, lo_pct))
        hi = float(np.percentile(flat, hi_pct))
        return cls(lo, hi if hi - lo > 1e-6 else lo + 1e-6)

    def __call__(self, amap: np.ndarray) -> np.ndarray:
        """amap → [0, 1]（截断，不逐图归一化）。"""
        return np.clip((np.asarray(amap, dtype=np.float32) - self.lo)
                       / (self.hi - self.lo), 0.0, 1.0)


# ---------------------------------------------------------------------- #
# 上色
# ---------------------------------------------------------------------- #
_CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


def load_font(size: int = 14):
    """加载一个含中文字形的字体。

    坑：PIL 的 ImageFont.load_default() 不含中文字形，直接画中文会全变成方块。
    """
    from PIL import ImageFont

    for p in _CJK_FONTS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_labels(canvas_img, labels, panel_w: int, header: int = 22, gap: int = 8,
                font_size: int = 14):
    """在拼图顶部画面板标题（支持中文）。canvas 尺寸需为 (len(labels)*(panel_w+gap), H+header)。"""
    from PIL import ImageDraw

    dr = ImageDraw.Draw(canvas_img)
    font = load_font(font_size)
    for j, lab in enumerate(labels):
        dr.text((j * (panel_w + gap) + 4, 3), lab, fill=(0, 0, 0), font=font)
    return canvas_img


def colorize(norm01: np.ndarray, gray: np.ndarray, gt: Optional[np.ndarray] = None,
             alpha: float = 0.5) -> np.ndarray:
    """[0,1] 的异常图叠到灰度原图上，可选画出 GT 轮廓（绿）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    h, w = gray.shape
    heat = (cm.jet(np.clip(norm01, 0, 1))[..., :3] * 255).astype(np.float32)
    g = np.stack([gray] * 3, axis=-1).astype(np.float32) * 255.0
    out = ((1 - alpha) * g + alpha * heat).astype(np.uint8)

    if gt is not None:
        from PIL import Image, ImageFilter
        e = np.asarray(Image.fromarray(gt.astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
                       dtype=np.float32)
        out[e > 30] = [0, 255, 0]
        # 哨兵：传了 gt 却一条边都没画出来，多半是值域不对。FIND_EDGES 之后用 e>30 取边，
        # 要求输入是 0/255 的 uint8；喂 0/1 的 mask 时 e 最大只有 8，绿轮廓会**静默消失**
        # （脑线 make_heatmaps.py / eval_final.py 传的正是 0/1）。只提示，不改任何像素。
        if not np.any(e > 30):
            import sys
            print("[visualize.colorize] 传入了 gt 但没有任何边缘超过阈值：绿轮廓为空。"
                  "请确认 gt 是 0/255 的 uint8 —— 0/1 的 mask 会静默画不出线。",
                  file=sys.stderr)
    return out


def as_binary_mask(mask) -> np.ndarray:
    """0/1 和 0/255 都变成 bool。不能用 >127，否则 0/1 掩码会消失。"""
    values = np.asarray(mask)
    if values.size == 0:
        return values.astype(bool)
    return values > 0


def binary_contour(mask) -> np.ndarray:
    """由 bool 掩码得到一圈边界。全零掩码返回全零，不把它当成绘图错误。"""
    binary = as_binary_mask(mask)
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)
    padded = np.pad(binary, 1, constant_values=False)
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return binary & ~neighbors


def paint_contour(rgb: np.ndarray, mask, color=(0, 255, 0)) -> np.ndarray:
    edge = binary_contour(mask)
    if edge.shape != rgb.shape[:2]:
        raise ValueError("轮廓掩码必须与图像同尺寸")
    out = np.array(rgb, copy=True)
    out[edge] = color
    return out


def _resize(image: np.ndarray, size: int, resample: str) -> np.ndarray:
    from PIL import Image

    mode = "F" if image.dtype != np.uint8 and image.ndim == 2 else None
    if image.ndim == 2 and image.dtype != np.uint8:
        source = Image.fromarray(image.astype(np.float32), mode="F")
        resized = source.resize((size, size), Image.BILINEAR if resample == "bilinear" else Image.NEAREST)
        return np.asarray(resized, dtype=np.float32)
    source = Image.fromarray(image.astype(np.uint8) if image.dtype == np.uint8 or image.ndim == 3 else image)
    method = Image.BILINEAR if resample == "bilinear" else Image.NEAREST
    return np.asarray(source.resize((size, size), method))


def _heat_rgb(gray01: np.ndarray, score01: np.ndarray, alpha: float, vmin: float, vmax: float) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    span = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((np.asarray(score01, dtype=np.float32) - float(vmin)) / span, 0.0, 1.0)
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.float32)
    gray = np.clip(np.asarray(gray01, dtype=np.float32), 0.0, 1.0)
    base = np.stack([gray, gray, gray], axis=-1) * 255.0
    return ((1.0 - alpha) * base + alpha * heat).astype(np.uint8)


def render_four_panel(gray01, score_map_high, pred_mask, gt_mask, gt_available: bool,
                      panel_size: int = 256, vmin: float = 0.0, vmax: float = 1.0,
                      mask_status: str = "predicted", sample_title: str = "") -> np.ndarray:
    """四列：原图、热图、热图+GT、预测掩码+GT。不重新推理，也不按 GT 改分数。"""
    from PIL import Image, ImageDraw

    gray = np.clip(np.asarray(gray01, dtype=np.float32), 0.0, 1.0)
    score = np.asarray(score_map_high, dtype=np.float32)
    if gray.shape != score.shape:
        raise ValueError("原图与热图尺寸必须一致")
    gray_show = _resize(gray, panel_size, "bilinear")
    score_show = _resize(score, panel_size, "bilinear")
    heat = _heat_rgb(gray_show, score_show, alpha=0.45, vmin=vmin, vmax=vmax)
    original = (np.stack([gray_show] * 3, axis=-1) * 255).astype(np.uint8)
    heat_gt = heat.copy()
    mask_panel = original.copy()
    gt_bool = None if gt_mask is None or not gt_available else as_binary_mask(gt_mask)
    if gt_bool is not None and gt_bool.shape != gray.shape:
        gt_bool = as_binary_mask(_resize(gt_bool.astype(np.uint8) * 255, panel_size, "nearest"))
    elif gt_bool is not None:
        gt_bool = as_binary_mask(_resize(gt_bool.astype(np.uint8) * 255, panel_size, "nearest"))
    if gt_available and gt_bool is not None and gt_bool.any():
        heat_gt = paint_contour(heat_gt, gt_bool)
    if pred_mask is None or mask_status == "unavailable":
        mask_panel = np.zeros_like(original)
    else:
        pred = as_binary_mask(pred_mask)
        pred = as_binary_mask(_resize(pred.astype(np.uint8) * 255, panel_size, "nearest"))
        red = mask_panel.astype(np.float32)
        red[pred] = red[pred] * 0.65 + np.array([255, 0, 0], dtype=np.float32) * 0.35
        mask_panel = red.astype(np.uint8)
        if gt_available and gt_bool is not None and gt_bool.any():
            mask_panel = paint_contour(mask_panel, gt_bool)
    gap = 8
    header = 36
    width = panel_size * 4 + gap * 3
    canvas = Image.new("RGB", (width, panel_size + header), (0, 0, 0))
    frames = [original, heat, heat_gt, mask_panel]
    for index, frame in enumerate(frames):
        canvas.paste(Image.fromarray(frame), (index * (panel_size + gap), header))
    draw = ImageDraw.Draw(canvas)
    font = load_font(13)
    titles = ["Original", "Heatmap", "Heatmap + GT", "Refined mask + GT"]
    for index, title in enumerate(titles):
        draw.text((index * (panel_size + gap) + 4, 2), title, fill=(230, 230, 230), font=font)
    note_font = load_font(12)
    if sample_title:
        draw.text((4, 16), sample_title[:80], fill=(180, 180, 180), font=note_font)
    if not gt_available:
        draw.text((2 * (panel_size + gap) + 4, header + 4), "GT unavailable", fill=(255, 255, 0), font=note_font)
        draw.text((3 * (panel_size + gap) + 4, header + 4), "GT unavailable", fill=(255, 255, 0), font=note_font)
    elif gt_bool is not None and not gt_bool.any():
        draw.text((2 * (panel_size + gap) + 4, header + 4), "GT empty", fill=(180, 255, 180), font=note_font)
        draw.text((3 * (panel_size + gap) + 4, header + 4), "GT empty", fill=(180, 255, 180), font=note_font)
    if pred_mask is None or mask_status == "unavailable":
        draw.text((3 * (panel_size + gap) + 4, header + 20), "Mask unavailable: threshold missing", fill=(255, 180, 180), font=note_font)
    return np.asarray(canvas)


def save_contact_sheet(panels: Sequence[np.ndarray], path: str, header: str, footer: str,
                       rows_per_page: int = 8) -> list:
    """把已经画好的四列图分页拼起来。不重新推理，也不改预测。"""
    from PIL import Image, ImageDraw

    if rows_per_page < 1:
        raise ValueError("rows_per_page 必须为正")
    os_makedirs = __import__("os").makedirs
    os_makedirs(__import__("os").path.dirname(path) or ".", exist_ok=True)
    written = []
    font = load_font(16)
    small = load_font(13)
    pages = [panels[start:start + rows_per_page] for start in range(0, max(len(panels), 1), rows_per_page)]
    if not panels:
        pages = [[]]
    for page_index, page in enumerate(pages, start=1):
        row_h = 0 if not page else page[0].shape[0]
        row_w = 0 if not page else page[0].shape[1]
        margin = 28
        height = margin * 2 + max(len(page), 1) * (row_h + 10)
        canvas = Image.new("RGB", (max(row_w, 640), height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), header[:180], fill=(230, 230, 230), font=font)
        for row_index, panel in enumerate(page):
            canvas.paste(Image.fromarray(panel), (0, margin + row_index * (row_h + 10)))
        draw.text((8, height - 22), f"{footer[:140]}  {page_index}/{len(pages)}", fill=(180, 180, 180), font=small)
        stem, ext = __import__("os").path.splitext(path)
        page_path = path if len(pages) == 1 else f"{stem}_{page_index:03d}{ext or '.png'}"
        canvas.save(page_path)
        written.append(page_path)
    return written


def render(amap: np.ndarray, gray: np.ndarray, calib: AmapCalibration,
           gt: Optional[np.ndarray] = None, guided: bool = True,
           radius: int = 4, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """一站式出图。返回 (叠加图, 归一化后的异常图)。

    顺序很关键：**先在 14×14 原生分辨率上标定到 [0,1]**（分位数就是在这个分辨率上
    测出来的），再上采样。掉过来的话，引导滤波会改变数值范围，标定就失准了。
    """
    from PIL import Image

    g = np.asarray(gray, dtype=np.float32)
    if g.max() > 1.0:
        g = g / 255.0

    norm01_low = calib(amap)                                   # (h, w) 原生分辨率
    if guided:
        up = upsample_guided(norm01_low, g, radius=radius, eps=eps)
    else:
        up = np.asarray(
            Image.fromarray(norm01_low.astype(np.float32), mode="F").resize(
                (g.shape[1], g.shape[0]), Image.BILINEAR), dtype=np.float32)
    up = np.clip(up, 0.0, 1.0)
    return colorize(up, g, gt), up
