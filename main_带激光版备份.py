# ============================================================
# K230 万能视觉平台
# 适配：庐山派 K230 + ST7701 800x480 触摸屏 + CanMV
#
# 平台原则：
# 1) 屏幕界面全部中文，不显示版本号。
# 2) YOLO 支持 YOLOv5 / YOLOv8 / YOLO11 / YOLO26 多模型触屏切换。
# 3) 形状、颜色、码类、YOLO 支持独立“识别区域”；YOLO 采用逻辑区域过滤。
# 4) 所有视觉模块统一发布识别结果，供显示、串口、测量等公共模块复用。
# 5) 支持三路独立串口 UART1 / UART2 / UART4，可同时启用。
# 6) 支持三种发送协议：二进制、ASCII十六进制、字符串；帧头 0x55，帧尾 0x66。
# 7) 支持单目标/多目标、目标选择规则、发送频率、绝对/相对坐标。
# 8) 支持两点标定、像素/mm/cm 测量叠加。
# 9) 支持四个场景槽，一键保存/恢复整套比赛参数。
# 10) 串口可选开启远程控制：切页面、切模型、加载/保存场景。
# 11) 增加几何视觉：直线、线段、圆、矩形，统一支持识别区域与结果输出。
# 12) 增加模板识别：4 个模板槽，支持触屏框选模板、保存、匹配与搜索区域。
# 13) 增加相机控制：水平镜像、垂直翻转，统一保存到平台配置。
# 14) 不启用网络模块，优先保证比赛现场的流畅、稳定和可维护性。
# 15) 首页默认使用800x480不透明控制台，视频层继续运行但不露出，切入算法页立即显示。
# 16) 运行中改变相机方向采用 stop->set_hmirror/set_vflip->run，确保多通道真正生效。
# ============================================================

import os
import time
import gc
import ujson
import image
import math

from machine import TOUCH, UART, FPIOA
from media.sensor import *
from media.display import *
from media.media import *

try:
    from libs.YOLO import YOLOv5
except Exception:
    YOLOv5 = None

try:
    from libs.YOLO import YOLO26
except Exception:
    YOLO26 = None

try:
    from libs.YOLO import YOLOv8
except Exception:
    YOLOv8 = None

try:
    from libs.YOLO import YOLO11
except Exception:
    YOLO11 = None

try:
    import cv2
    from ulab import numpy as np
    CV2_OK = True
except Exception:
    cv2 = None
    np = None
    CV2_OK = False


# ============================================================
# 0. 用户只需要改这里：YOLO 多模型配置
# ============================================================
# 使用说明：
# 1) 每增加一个模型，就复制一个字典块；无需再改 YoloEngine / UI / default_config。
# 2) name         : 屏幕上显示的模型名。
# 3) family       : 当前支持 "yolov5" / "yolov8" / "yolo11" / "yolo26"。
#                    也兼容简写："v5" / "v8" / "11" / "26"。
# 4) model_path   : kmodel 在 K230 上的绝对路径。
# 5) labels       : 必须与训练类别顺序完全一致。
# 6) input_size   : 训练/转换模型时使用的 imgsz，例如 [320, 320]。
# 7) conf / nms   : 该模型第一次使用时的默认阈值；屏幕调节并保存后会记住。
# 8) 官方接口差异：YOLOv5/YOLOv8/YOLO11 detect 传 nms_thresh；
#    YOLO26 detect 不包含 nms_thresh，因此 family="yolo26" 时自动省略。
#
# ★ 以后换模型，优先只改本区。

YOLO_DEFAULT_MODEL_INDEX = 0

# 摄像头给 YOLO 的 RGBP888 推理帧尺寸。一般不需要随模型 imgsz 改动。
YOLO_INFERENCE_FRAME = [640, 360]

YOLO_MODEL_PROFILES = [
    # ===== 狐狸检测（K230 模型工厂生成 2026-09-25）=====
    {
        "name": "狐狸",
        "family": "yolo11",
        "model_path": "/sdcard/kmodel/yolo11n_320_20260925_0000/best.kmodel",
        "labels": ["狐狸"],
        "input_size": [320, 320],
        "conf": 0.30,
        "nms": 0.45,
        "max_boxes": 20,
    },

{
        "name": "可乐",
        "family": "yolo11",
        "model_path": "/sdcard/kmodel/yolo11n_640_20260926_1242/best.kmodel",
        "labels": ["可乐"],
        "input_size": [640, 640],
        "conf": 0.30,
        "nms": 0.45,
        "max_boxes": 20,
    },


    # 例如同一套 labels/imgsz 已经写好，只换成 YOLOv5 模型时：
    # 把 family 改成 "yolov5"，把 model_path 改成对应 v5 kmodel 即可。
    # YOLO26 同理：family="yolo26"；代码会自动停止传 nms_thresh。

    # ===== 示例：新增第二个 YOLOv8 模型时，取消下面注释并修改 =====
     {
         "name": "模型2",
         "family": "yolo11",
         "model_path": "/sdcard/examples/kmodel/fruit_det_yolo11n_320.kmodel",
         "labels": ["apple","banana","orange"],
         "input_size": [320, 320],
         "conf": 0.30,
         "nms": 0.45,
         "max_boxes": 20,
     },

    # ===== 示例：新增第三个模型 =====
     {
         "name": "模型3",
         "family": "yolo26",
         "model_path": "/sdcard/examples/kmodel/fruit_det_yolo26n_320.kmodel",
         "labels": ["apple","banana","orange"],
         "input_size": [320, 320],
         "conf": 0.25,
         "nms": 0.45,
         "max_boxes": 20,
     },
]


# ============================================================
# 用户可改：三路串口硬件与协议默认值
# ============================================================
# FPIOA.set_function() 使用的是芯片 GPIO/PAD 编号，不是 40Pin 排针序号。
# 以下默认映射按庐山派/CanMV 官方资料中的可用 UART 功能设置：
#   UART1: GPIO3  TX / GPIO4  RX（40Pin：8/10）
#   UART2: GPIO5  TX / GPIO6  RX（40Pin：11/13；板载GH1.25串口2另可用GPIO11/12）
#   UART4: GPIO36 TX / GPIO37 RX（40Pin：29/31，和部分音频功能复用）
# 若你的扩展板已经占用了这些引脚，只改这里即可，后续通信代码不用改。
#
# enabled_default 默认关闭：第一次上板不会抢占你现有外设引脚。
# 用户确认接线后，可在触摸“串口通信”页开启，也可以直接把默认值改为 True。

SERIAL_FRAME_HEAD = 0x55
SERIAL_FRAME_TAIL = 0x66
SERIAL_DEFAULT_BAUD = 115200
SERIAL_MAX_TARGETS = 8
SERIAL_RESULT_FRESH_MS = 700

SERIAL_PORT_PROFILES = [
    {
        "name": "串口1",
        "uart_no": 1,
        "tx_pin": 3,
        "rx_pin": 4,
        "enabled_default": False,
    },
    {
        "name": "串口2",
        "uart_no": 2,
        "tx_pin": 5,
        "rx_pin": 6,
        "enabled_default": False,
    },
    {
        "name": "串口3",
        "uart_no": 4,
        "tx_pin": 36,
        "rx_pin": 37,
        "enabled_default": False,
    },
]

SERIAL_BAUD_OPTIONS = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
SERIAL_HZ_OPTIONS = [5, 10, 20, 30, 50]
SERIAL_PROTOCOL_OPTIONS = ["binary", "ascii_hex", "string"]
SERIAL_SOURCE_OPTIONS = ["current", "yolo", "color", "shape", "line", "code", "geometry", "template"]
SERIAL_TARGET_MODE_OPTIONS = ["single", "multi"]
SERIAL_SINGLE_RULE_OPTIONS = ["center", "score", "area"]
SERIAL_COORD_MODE_OPTIONS = ["absolute", "screen_center", "region_center", "custom"]

# 测量/标定公共选项。界面全部使用中文，内部键保持短英文便于维护。
MEASURE_SOURCE_OPTIONS = ["current", "yolo", "color", "shape", "line", "code", "geometry", "template"]
MEASURE_RULE_OPTIONS = ["center", "score", "area"]
MEASURE_REF_OPTIONS = ["screen_center", "region_center", "custom"]
MEASURE_UNIT_OPTIONS = ["pixel", "mm", "cm"]

# 场景不是“只保存一份参数”，还必须明确绑定一个执行功能。
# 加载场景后会自动跳到该识别页，所以“场景1=第一题颜色识别”可以真正一键执行。
SCENE_SLOT_COUNT = 4
SCENE_PATH_FMT = "/sdcard/k230_scene_%d.json"
SCENE_PAGE_OPTIONS = ["color", "yolo", "shape", "line", "code", "geometry", "template"]

APRILTAG_FAMILY_OPTIONS = [
    "all", "16h5", "25h7", "25h9", "36h10", "36h11", "artoolkit"
]
APRILTAG_FAMILY_UI_NAMES = {
    "all": "全部6类",
    "16h5": "TAG16H5",
    "25h7": "TAG25H7",
    "25h9": "TAG25H9",
    "36h10": "TAG36H10",
    "36h11": "TAG36H11",
    "artoolkit": "ARTOOLKIT",
}
SCENE_PAGE_NAMES = {
    "color": "颜色识别",
    "yolo": "YOLO识别",
    "shape": "形状识别",
    "line": "巡线导航",
    "code": "码类识别",
    "geometry": "几何视觉",
    "template": "模板识别",
}
VISION_PAGE_OPTIONS = tuple(SCENE_PAGE_OPTIONS)

# 几何视觉公共选项。
GEOMETRY_MODE_OPTIONS = ["line", "segment", "circle", "rect"]

# 内存安全策略：几何/LSD/Hough/NCC 都会使用图像算法临时工作区。
# 传统视觉原始通道仍保持 400x240，不影响颜色/巡线/形状；
# 只有进入几何视觉或模板识别时，才对“当前这一帧”原地降采样。
# mean_pool() 是原地缩小，不创建另一张大图，优先避免 fast frame buffer 溢出。
GEOMETRY_WORK_DIV = 2          # 400x240 -> 200x120；若仍不足会自动再缩小一次
TEMPLATE_WORK_DIV = 2          # 400x240 -> 200x120，保存与匹配始终使用同一比例
RELEASE_YOLO_ON_HEAVY_PAGE = True  # 进入几何/模板页时释放暂时不用的YOLO资源

# 模板识别：4 个模板槽。新模板保存为灰度 PGM，避免加载后再次做大图格式转换。
TEMPLATE_SLOT_COUNT = 4
TEMPLATE_DIR = "/sdcard/k230_templates"
TEMPLATE_PATH_FMT = TEMPLATE_DIR + "/template_%d.pgm"
TEMPLATE_LEGACY_PATH_FMT = TEMPLATE_DIR + "/template_%d.bmp"
TEMPLATE_SEARCH_OPTIONS = ["exhaustive", "fast"]



# ============================================================
# 1. 全局常量（一般无需手改）
# ============================================================

DISPLAY_W = ALIGN_UP(800, 16)
DISPLAY_H = 480

# 传统视觉使用 400x240：与 800x480 显示保持完全相同的 5:3 宽高比，
# 坐标映射为严格 2x，避免形状/巡线叠加框因非等比缩放变形。
CV_W = ALIGN_UP(400, 16)
CV_H = 240

# YOLO 保留独立 RGBP888 通道。
AI_W = int(YOLO_INFERENCE_FRAME[0])
AI_H = int(YOLO_INFERENCE_FRAME[1])

# 右侧 UI 面板宽度保持 16 对齐，避免大尺寸第二张全屏 ARGB8888。
PANEL_W = 240
PANEL_X = DISPLAY_W - PANEL_W
PANEL_H = DISPLAY_H
PREVIEW_W = PANEL_X

CONFIG_PATH = "/sdcard/k230_vision_platform.json"
LEGACY_CONFIG_PATHS = ()
FULL_LAB = [0, 100, -128, 127, -128, 127]

# Touch 行为参数。
TOUCH_RELEASE_GAP_MS = 180     # 连续空读达到该时间才认为一次接触结束，避免偶发漏点重复触发。
TOUCH_REPEAT_DELAY_MS = 480   # REPEAT 长按开始连发延迟。
TOUCH_REPEAT_INTERVAL_MS = 130
TOUCH_HOLD_MS = 1000          # HOLD 危险按钮长按确认时间。
TOUCH_REARM_MS = 320          # 普通 TAP 完成后的短防抖；略加长以压制一次物理触摸被拆成两次。
TOUCH_TAP_DEDUP_MS = 280       # 同一 TAP 动作的最短重复间隔，防止触摸尾部重复提交。
TOUCH_CONFIRM_TIMEOUT_MS = 2800 # 危险操作改为“两次单击确认”，避免长按依赖触摸事件持续上报。
TOUCH_CONFIRM_MIN_GAP_MS = 650  # 两次确认之间必须真正松手并间隔一段时间，防止同一次物理长按被拆成两次。
TOUCH_YOLO_REARM_MS = 520     # YOLO 模型重建结束后的专用防抖。
TOUCH_YOLO_RELEASE_STABLE_MS = 220  # YOLO模型按钮：连续无触点达到该时间才认定真正松手。
TOUCH_YOLO_UNLOCK_QUIET_MS = 650    # 模型重建结束后还需保持无触点，才允许下一次切模。
TOUCH_MOVE_TOL = 36           # TAP 允许的最大移动像素。

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK = (24, 28, 34)
COLOR_DARK2 = (35, 42, 52)
COLOR_GRAY = (90, 100, 115)
COLOR_LIGHT = (205, 215, 225)
COLOR_BLUE = (45, 125, 255)
COLOR_GREEN = (0, 220, 95)
COLOR_RED = (255, 75, 75)
COLOR_YELLOW = (255, 218, 40)
COLOR_CYAN = (30, 220, 220)
COLOR_ORANGE = (255, 155, 45)
COLOR_PURPLE = (190, 95, 255)

PALETTE = [
    (255, 80, 80),
    (80, 255, 120),
    (80, 160, 255),
    (255, 220, 80),
    (220, 80, 255),
    (80, 255, 255),
]

PRESS_TAP = "tap"
PRESS_REPEAT = "repeat"
PRESS_HOLD = "hold"
PRESS_CONFIRM = "confirm"


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def copy_lab(v):
    return [int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), int(v[5])]


def is_full_lab(v):
    try:
        return list(v) == FULL_LAB
    except Exception:
        return True


def deep_update(base, src):
    if not isinstance(src, dict):
        return base
    for k in src:
        if k in base and isinstance(base[k], dict) and isinstance(src[k], dict):
            deep_update(base[k], src[k])
        else:
            base[k] = src[k]
    return base


def safe_ticks_diff(now, before):
    try:
        return time.ticks_diff(now, before)
    except Exception:
        return now - before


def is_fast_fb_error(exc):
    """判断 CanMV 图像算法常见的快速帧缓冲区不足异常。"""
    try:
        text = str(exc).lower()
    except Exception:
        return False
    return ("fast frame buffer" in text or
            "frame buffer stack" in text or
            "fb stack" in text)


def scale_roi_between_images(roi, src_w, src_h, dst_w, dst_h):
    """把一个图像坐标 ROI 等比例映射到另一尺寸，并保证至少 2x2。"""
    if roi is None:
        return None
    try:
        x, y, w, h = [int(v) for v in roi]
        sx = float(dst_w) / float(max(1, src_w))
        sy = float(dst_h) / float(max(1, src_h))
        nx = int(clamp(round(x * sx), 0, max(0, dst_w - 2)))
        ny = int(clamp(round(y * sy), 0, max(0, dst_h - 2)))
        nw = int(clamp(round(w * sx), 2, max(2, dst_w - nx)))
        nh = int(clamp(round(h * sy), 2, max(2, dst_h - ny)))
        return (nx, ny, nw, nh)
    except Exception:
        return None


def pool_image_in_place(img_obj, div):
    """使用官方 mean_pool 原地缩小当前算法帧，不制造第二张大图。"""
    d = max(1, int(div))
    if d > 1:
        img_obj.mean_pool(d, d)
    return img_obj


def grayscale_heap_copy(img_obj):
    """
    生成灰度工作图时优先明确放到系统 HEAP。
    不使用默认 to_grayscale()，避免临时图走 fast frame buffer。
    不同正式固件的关键字支持略有差别，所以保留官方 copy=True 兼容回退。
    """
    alloc_heap = getattr(image, "ALLOC_HEAP", None)
    if alloc_heap is not None:
        try:
            return img_obj.to_grayscale(copy=True, alloc=alloc_heap)
        except TypeError:
            pass
    # 官方 OpenMV 兼容接口：copy=True 在堆上创建新图像副本。
    return img_obj.to_grayscale(copy=True)


def safe_label_y(y, box_h=0, text_h=22):
    # 目标靠近顶部时把标签放框内/框下，避免全部堆在屏幕最上方。
    if y >= text_h + 2:
        return y - text_h - 2
    return y + 3 if box_h <= 0 else min(DISPLAY_H - text_h - 2, y + 3)


def draw_cross_lines(img, x, y, color=COLOR_GREEN, size=8, thickness=2):
    img.draw_line(x - size, y, x + size, y, color=color, thickness=thickness)
    img.draw_line(x, y - size, x, y + size, color=color, thickness=thickness)


def draw_short_status(img, text, color=COLOR_YELLOW):
    # 状态只占左下角，不再长期占用左上视觉区域。
    if not text:
        return
    if len(text) > 34:
        text = text[:34]
    img.draw_string_advanced(8, DISPLAY_H - 30, 18, text, color=color)


def normalize_yolo_family(value):
    """把用户填写的 family 统一成 libs.YOLO 中对应的标准类名键。"""
    f = str(value if value is not None else "yolov8").strip().lower()
    aliases = {
        "5": "yolov5", "v5": "yolov5", "yolo5": "yolov5", "yolov5": "yolov5",
        "8": "yolov8", "v8": "yolov8", "yolo8": "yolov8", "yolov8": "yolov8",
        "11": "yolo11", "v11": "yolo11", "yolo11": "yolo11", "yolov11": "yolo11",
        "26": "yolo26", "v26": "yolo26", "yolo26": "yolo26", "yolov26": "yolo26",
    }
    return aliases.get(f, f)


def yolo_profile_count():
    return len(YOLO_MODEL_PROFILES)


def get_yolo_profile(index):
    if not YOLO_MODEL_PROFILES:
        raise ValueError("YOLO_MODEL_PROFILES 不能为空")
    idx = int(clamp(int(index), 0, len(YOLO_MODEL_PROFILES) - 1))
    return idx, YOLO_MODEL_PROFILES[idx]


def _safe_profile_float(profile, key, default):
    try:
        return float(profile.get(key, default))
    except Exception:
        return float(default)


def remember_yolo_runtime(y):
    """把当前模型在触屏上调过的 CONF/NMS 保存到每模型独立槽位。"""
    try:
        idx, profile = get_yolo_profile(y.get("model_index", YOLO_DEFAULT_MODEL_INDEX))
    except Exception:
        return
    params = y.get("model_params", {})
    if not isinstance(params, dict):
        params = {}
    params[str(idx)] = {
        "conf": float(y.get("conf", _safe_profile_float(profile, "conf", 0.25))),
        "nms": float(y.get("nms", _safe_profile_float(profile, "nms", 0.45))),
    }
    y["model_params"] = params


def apply_yolo_profile_to_cfg(y, index, use_saved=True):
    """让源码最前面的 profile 成为模型路径/标签/imgsz 的唯一配置入口。"""
    idx, p = get_yolo_profile(index)
    family = normalize_yolo_family(p.get("family", "yolov8"))

    labels = p.get("labels", ["object"])
    if not isinstance(labels, list) or len(labels) == 0:
        labels = ["object"]
    input_size = p.get("input_size", [320, 320])
    if not isinstance(input_size, list) or len(input_size) < 2:
        input_size = [320, 320]

    y["model_index"] = idx
    y["profile_name"] = str(p.get("name", "模型%d" % (idx + 1)))
    y["family"] = family
    y["model_path"] = str(p.get("model_path", ""))
    y["labels"] = [str(v) for v in labels]
    y["model_input_size"] = [int(input_size[0]), int(input_size[1])]
    y["max_boxes"] = int(p.get("max_boxes", 20))

    conf = _safe_profile_float(p, "conf", 0.25)
    nms = _safe_profile_float(p, "nms", 0.45)
    if use_saved:
        params = y.get("model_params", {})
        if isinstance(params, dict):
            saved = params.get(str(idx), {})
            if isinstance(saved, dict):
                try:
                    conf = float(saved.get("conf", conf))
                    nms = float(saved.get("nms", nms))
                except Exception:
                    pass
    y["conf"] = float(clamp(conf, 0.05, 0.95))
    y["nms"] = float(clamp(nms, 0.05, 0.95))
    y["official_draw"] = False
    return idx, p


def default_profile(index):
    return {
        "name": "目标%d" % (index + 1),
        "trained": False,
        "enabled": False,
        "auto": False,
        "lab": copy_lab(FULL_LAB),
    }


def default_region():
    return {
        "enabled": False,
        "rect": [0, 0, DISPLAY_W, DISPLAY_H],
    }


def default_serial_port(index):
    profile = SERIAL_PORT_PROFILES[index]
    return {
        "enabled": bool(profile.get("enabled_default", False)),
        "baudrate": SERIAL_DEFAULT_BAUD,
        "bits": 8,
        "parity": "none",
        "stop": 1,
        "protocol": "binary",
        "source": "current",
        "target_mode": "single",
        "single_rule": "center",
        "coord_mode": "absolute",
        "custom_x": DISPLAY_W // 2,
        "custom_y": DISPLAY_H // 2,
        "send_hz": 20,
        "max_targets": SERIAL_MAX_TARGETS,
        "send_empty": True,
        "string_token": "T",
        # 是否允许该串口接收到命令后控制页面/模型/场景。默认关闭，避免误触发。
        "rx_control": False,
    }


def default_config():
    yi, yp = get_yolo_profile(YOLO_DEFAULT_MODEL_INDEX)
    return {
        "system": {
            "show_fps": True,
            "touch_debug": False,
            "ide_mirror": True,
            "ide_quality": 70,
        },
        "shape": {
            "mode": "auto",           # auto / triangle / rect / circle
            "canny_low": 50,
            "canny_high": 145,
            "min_area": 450,
            "max_area_ratio": 0.72,
            "epsilon_ratio": 0.020,
            "min_solidity": 0.88,
            "rect_min_fill": 0.70,
            "circle_min_circularity": 0.72,
            "circle_min_fill": 0.68,
            "stable_frames": 2,
            "stable_dist": 32,
            "max_results": 8,
            # ---- 激光点识别（mode = "laser"，或在其它形状模式下叠加）----
            "laser_threshold": 245,      # 亮度门限 0~255：比它亮的才算激光
            "laser_min_pixels": 4,       # 最小像素数（滤掉噪点）
            "laser_max_pixels": 1200,    # 最大像素数（滤掉白纸/灯光/反光）
            "laser_min_density": 0.35,   # 最小密度（圆点约 0.785）
            "laser_min_chroma": 40,      # 光晕彩度下限：激光有颜色，白高光是灰的
            "laser_bright_margin": 8,    # 只认最亮的：比最亮点暗这么多的丢掉
            "laser_overlay": 1,          # 1 = 其它形状模式下也顺便找激光点
            "param_idx": 0,
        },
        "color": {
            "current": 0,
            "x_stride": 4,
            "y_stride": 2,
            "area_threshold": 100,
            "pixels_threshold": 70,
            "min_density": 0.16,
            "max_results": 12,
            "adapt_interval": 12,
            "adapt_step": 2,
            "param_idx": 0,
            "profiles": [default_profile(i) for i in range(4)],
        },
        "line": {
            "mode": "roi",            # roi / regression / eight
            "trained": False,
            "lab": copy_lab(FULL_LAB),
            "invert": False,
            "robust": True,
            "area_threshold": 55,
            "pixels_threshold": 35,
            "min_density": 0.08,
            "roi_count": 7,
            "corridor_half_width": 72,
            "continuity_px": 42,
            "eight_max_gap": 4,
            "eight_half_width": 48,
            "eight_min_points": 32,
            "eight_max_band_width": 110,
            "smooth_alpha": 0.45,
            "param_idx": 0,
        },
        "yolo": {
            "model_index": yi,
            "profile_name": str(yp.get("name", "模型%d" % (yi + 1))),
            "family": normalize_yolo_family(yp.get("family", "yolov8")),
            "model_path": str(yp.get("model_path", "")),
            "labels": [str(v) for v in yp.get("labels", ["object"])],
            "model_input_size": [int(yp.get("input_size", [320, 320])[0]), int(yp.get("input_size", [320, 320])[1])],
            "conf": _safe_profile_float(yp, "conf", 0.25),
            "nms": _safe_profile_float(yp, "nms", 0.45),
            "max_boxes": int(yp.get("max_boxes", 20)),
            "model_params": {},
            "show_box": True,
            "show_label": True,
            "show_conf": True,
            "show_center": False,
            "official_draw": False,
        },
        "code": {
            "mode": "qr",
            "max_results": 6,
            "dm_effort": 200,
            "apriltag_family": "all",
            "barcode_highres": True,
        },
        "geometry": {
            "mode": "line",
            # 直线：官方 Hough 参数 + 平台后处理最小线长。
            "line_threshold": 1800,
            "line_x_stride": 2,
            "line_y_stride": 1,
            "theta_margin": 20,
            "rho_margin": 20,
            "line_min_length": 90,          # 显示坐标像素

            # 线段：LSD 原始结果很多，先让官方接口合并，再做长线过滤/NMS。
            "segment_merge_distance": 20,
            "segment_theta_difference": 10,
            "segment_min_length": 90,       # 显示坐标像素
            "segment_nms_angle": 8,         # 度
            "segment_nms_distance": 24,     # 显示坐标像素

            # 圆：提高默认 Hough 阈值，并对同心/近同心圆做二次合并。
            "circle_threshold": 6000,
            "circle_x_stride": 2,
            "circle_y_stride": 1,
            "circle_x_margin": 14,
            "circle_y_margin": 14,
            "circle_r_margin": 18,
            "circle_min_radius": 20,        # 显示坐标像素
            "circle_max_radius": 240,       # 显示坐标像素
            "circle_merge_center": 36,      # 显示坐标像素
            "circle_merge_radius": 36,      # 显示坐标像素
            # 圆周真实性验证：霍夫圆只是候选，必须检查整圈是否真的存在连续边缘。
            # 这能压掉电线、桌面纹理、键盘缝隙等“若干弧段碰巧投票成大圆”的误检。
            "circle_verify": True,
            "circle_edge_contrast": 24,     # 圆周内外像素差阈值
            "circle_edge_support": 50,      # 整圈有效边缘支持率%
            "circle_quadrant_support": 28,  # 四个象限各自最低支持率%

            # 矩形：find_rects 是四边形检测器，不是语义分类器；
            # 再用四角角度 + 四条边的真实像素对比度验证，压掉圆形外接菱形误检。
            "rect_threshold": 12000,
            "rect_min_side": 55,            # 显示坐标像素
            "rect_max_angle_error": 28,     # 每个角距90度的最大误差
            "rect_edge_contrast": 28,       # RGB/灰度跨边缘平均差
            "rect_edge_support": 55,        # 0~100，四边采样命中比例
            # 四角真实性验证：真矩形的四个角本身应落在明显边缘上；
            # 圆形被 find_rects 外接成菱形时，菱形四个虚假角通常没有真实边缘。
            "rect_corner_support": 75,      # 0~100，默认至少 3/4 角点通过
            # 矩形/圆形互斥：仅对接近正方形的矩形候选额外跑一次霍夫圆确认。
            # 这样不会影响普通长方形，又能强力压掉“圆 -> 菱形矩形”。
            "rect_circle_veto": True,
            "rect_circle_aspect_max": 145,  # 长宽比 <= 1.45 时才做排圆检查（百分数）
            "rect_circle_center_ratio": 28, # 圆心距 <= 短边 28% 视为同一目标
            "rect_circle_diam_min": 58,     # 圆直径/矩形短边 >= 0.58
            "rect_circle_diam_max": 118,    # 圆直径/矩形短边 <= 1.18

            "smart_filter": True,
            "max_results": 6,
            "param_idx": 0,
            "show_center": True,
            "show_angle": True,
        },
        "template": {
            "selected": 0,
            "threshold": 0.72,
            "step": 2,
            "search": "exhaustive",
            "show_center": True,
            "show_box": True,
        },
        "camera": {
            "hmirror": False,
            "vflip": False,
        },
        "regions": {
            "shape": default_region(),
            "color": default_region(),
            "yolo": default_region(),
            "code": default_region(),
            "geometry": default_region(),
            "template_search": default_region(),
            "template_capture": default_region(),
        },
        "measurement": {
            "enabled": False,
            "source": "current",
            "target_rule": "center",
            "reference": "screen_center",
            "custom_x": DISPLAY_W // 2,
            "custom_y": DISPLAY_H // 2,
            "unit": "pixel",
            "known_mm": 100.0,
            "mm_per_pixel": 0.0,
            "calibration_p1": [0, 0],
            "calibration_p2": [0, 0],
            "show_center": True,
            "show_offset": True,
            "show_size": True,
            "show_area": False,
        },
        "scene": {
            "selected": 0,
            # 加载场景后自动进入的识别功能。
            "entry_page": "color",
        },
        "serial": {
            "selected": 0,
            "ports": [default_serial_port(i) for i in range(len(SERIAL_PORT_PROFILES))],
        },
    }


# ============================================================
# 1. 配置读写
# ============================================================

class ConfigStore:
    def __init__(self, path=CONFIG_PATH):
        self.path = path

    def _normalize(self, cfg):
        # 强制恢复颜色/巡线结构，不允许颜色目标互相共享 list 引用。
        c = cfg["color"]
        profiles = c.get("profiles", [])
        fixed = []
        for i in range(4):
            src = profiles[i] if i < len(profiles) and isinstance(profiles[i], dict) else {}
            p = default_profile(i)
            deep_update(p, src)
            p["name"] = str(p.get("name", "目标%d" % (i + 1)))
            try:
                lab = copy_lab(p.get("lab", FULL_LAB))
            except Exception:
                lab = copy_lab(FULL_LAB)
            p["lab"] = lab
            p["trained"] = bool(p.get("trained", False))
            if not p["trained"]:
                p["lab"] = copy_lab(FULL_LAB)
                p["enabled"] = False
                p["auto"] = False
            fixed.append(p)
        c["profiles"] = fixed
        c["current"] = int(clamp(int(c.get("current", 0)), 0, 3))
        c["param_idx"] = int(clamp(int(c.get("param_idx", 0)), 0, 5))

        ln = cfg["line"]
        try:
            ln["lab"] = copy_lab(ln.get("lab", FULL_LAB))
        except Exception:
            ln["lab"] = copy_lab(FULL_LAB)
        ln["trained"] = bool(ln.get("trained", False))
        if not ln["trained"]:
            ln["lab"] = copy_lab(FULL_LAB)

        # 模型路径/标签/imgsz/family 一律以前面的 YOLO_MODEL_PROFILES 为准。
        # JSON 只保存“当前选中哪个模型”和各模型触屏调过的 CONF/NMS。
        y = cfg["yolo"]
        try:
            idx = int(y.get("model_index", YOLO_DEFAULT_MODEL_INDEX))
        except Exception:
            idx = YOLO_DEFAULT_MODEL_INDEX
        idx = int(clamp(idx, 0, max(0, yolo_profile_count() - 1)))

        # 兼容旧配置：没有 model_params 时，把原来的 CONF/NMS 收进当前模型独立槽。
        params = y.get("model_params", {})
        if not isinstance(params, dict):
            params = {}
        if str(idx) not in params:
            try:
                params[str(idx)] = {
                    "conf": float(y.get("conf", get_yolo_profile(idx)[1].get("conf", 0.25))),
                    "nms": float(y.get("nms", get_yolo_profile(idx)[1].get("nms", 0.45))),
                }
            except Exception:
                pass
        y["model_params"] = params
        apply_yolo_profile_to_cfg(y, idx, use_saved=True)

        y["show_box"] = bool(y.get("show_box", True))
        y["show_label"] = bool(y.get("show_label", True))
        y["show_conf"] = bool(y.get("show_conf", True))
        y["show_center"] = bool(y.get("show_center", False))

        # 仅标准化码类识别新增配置。
        code_cfg = cfg.get("code", {})
        code_default = default_config()["code"]
        if isinstance(code_cfg, dict):
            deep_update(code_default, code_cfg)
        if code_default.get("mode") not in ("qr", "dm", "barcode", "apriltag"):
            code_default["mode"] = "qr"
        fam = str(code_default.get("apriltag_family", "all")).lower()
        if fam not in APRILTAG_FAMILY_OPTIONS:
            fam = "all"
        code_default["apriltag_family"] = fam

        # 一维条码已经实机验证：只有官方原生 640x480 GRAYSCALE 链路稳定。
        # 因此这里固定为 True，不再允许旧配置把它恢复成 False。
        code_default["barcode_highres"] = True

        try:
            code_default["max_results"] = int(clamp(int(code_default.get("max_results", 6)), 1, 20))
            code_default["dm_effort"] = int(clamp(int(code_default.get("dm_effort", 200)), 160, 240))
        except Exception:
            code_default["max_results"] = 6
            code_default["dm_effort"] = 200
        cfg["code"] = code_default

        # 识别区域：统一保存显示坐标，算法内部再转换到各自图像坐标。
        regions = cfg.get("regions", {})
        if not isinstance(regions, dict):
            regions = {}
        fixed_regions = {}
        for key in ("shape", "color", "yolo", "code", "geometry", "template_search", "template_capture"):
            src = regions.get(key, {}) if isinstance(regions.get(key, {}), dict) else {}
            r = default_region()
            deep_update(r, src)
            rect = r.get("rect", [0, 0, DISPLAY_W, DISPLAY_H])
            try:
                x, y0, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            except Exception:
                x, y0, w, h = 0, 0, DISPLAY_W, DISPLAY_H
            x = int(clamp(x, 0, DISPLAY_W - 4))
            y0 = int(clamp(y0, 0, DISPLAY_H - 4))
            w = int(clamp(w, 4, DISPLAY_W - x))
            h = int(clamp(h, 4, DISPLAY_H - y0))
            r["rect"] = [x, y0, w, h]
            r["enabled"] = bool(r.get("enabled", False))
            fixed_regions[key] = r
        cfg["regions"] = fixed_regions

        # 场景配置：每个场景必须绑定一个“加载后执行的识别功能”。
        scene = cfg.get("scene", {})
        if not isinstance(scene, dict):
            scene = {}
        try:
            scene_selected = int(scene.get("selected", 0))
        except Exception:
            scene_selected = 0
        scene_selected = int(clamp(scene_selected, 0, SCENE_SLOT_COUNT - 1))
        entry_page = str(scene.get("entry_page", "color"))
        if entry_page not in SCENE_PAGE_OPTIONS:
            entry_page = "color"
        cfg["scene"] = {"selected": scene_selected, "entry_page": entry_page}

        # 三路串口配置：旧配置缺失时自动补齐，用户硬件映射仍以前部 profile 为准。
        serial = cfg.get("serial", {})
        if not isinstance(serial, dict):
            serial = {}
        try:
            selected = int(serial.get("selected", 0))
        except Exception:
            selected = 0
        selected = int(clamp(selected, 0, len(SERIAL_PORT_PROFILES) - 1))
        old_ports = serial.get("ports", [])
        if not isinstance(old_ports, list):
            old_ports = []
        fixed_ports = []
        for i in range(len(SERIAL_PORT_PROFILES)):
            port = default_serial_port(i)
            if i < len(old_ports) and isinstance(old_ports[i], dict):
                deep_update(port, old_ports[i])
            port["enabled"] = bool(port.get("enabled", False))
            try:
                port["baudrate"] = int(port.get("baudrate", SERIAL_DEFAULT_BAUD))
            except Exception:
                port["baudrate"] = SERIAL_DEFAULT_BAUD
            if port["baudrate"] not in SERIAL_BAUD_OPTIONS:
                # 保留用户手动写入的合法自定义波特率。
                port["baudrate"] = int(clamp(port["baudrate"], 1200, 1500000))
            port["protocol"] = port.get("protocol", "binary")
            if port["protocol"] not in SERIAL_PROTOCOL_OPTIONS:
                port["protocol"] = "binary"
            port["source"] = port.get("source", "current")
            if port["source"] not in SERIAL_SOURCE_OPTIONS:
                port["source"] = "current"
            port["target_mode"] = port.get("target_mode", "single")
            if port["target_mode"] not in SERIAL_TARGET_MODE_OPTIONS:
                port["target_mode"] = "single"
            port["single_rule"] = port.get("single_rule", "center")
            if port["single_rule"] not in SERIAL_SINGLE_RULE_OPTIONS:
                port["single_rule"] = "center"
            port["coord_mode"] = port.get("coord_mode", "absolute")
            if port["coord_mode"] not in SERIAL_COORD_MODE_OPTIONS:
                port["coord_mode"] = "absolute"
            try:
                port["send_hz"] = int(port.get("send_hz", 20))
            except Exception:
                port["send_hz"] = 20
            port["send_hz"] = int(clamp(port["send_hz"], 1, 50))
            try:
                port["max_targets"] = int(port.get("max_targets", SERIAL_MAX_TARGETS))
            except Exception:
                port["max_targets"] = SERIAL_MAX_TARGETS
            port["max_targets"] = int(clamp(port["max_targets"], 1, SERIAL_MAX_TARGETS))
            port["send_empty"] = bool(port.get("send_empty", True))
            port["string_token"] = str(port.get("string_token", "T"))[:12]
            port["rx_control"] = bool(port.get("rx_control", False))
            fixed_ports.append(port)
        cfg["serial"] = {"selected": selected, "ports": fixed_ports}

        # 几何视觉配置。
        g = cfg.get("geometry", {})
        gd = default_config()["geometry"]
        if isinstance(g, dict):
            deep_update(gd, g)
        if gd.get("mode") not in GEOMETRY_MODE_OPTIONS:
            gd["mode"] = "line"
        int_ranges = {
            "line_threshold": (100, 30000), "line_x_stride": (1, 8), "line_y_stride": (1, 8),
            "theta_margin": (0, 90), "rho_margin": (0, 120), "line_min_length": (10, 700),
            "segment_merge_distance": (0, 160), "segment_theta_difference": (0, 90),
            "segment_min_length": (10, 700), "segment_nms_angle": (0, 45),
            "segment_nms_distance": (0, 160),
            "circle_threshold": (100, 50000), "circle_x_stride": (1, 8), "circle_y_stride": (1, 8),
            "circle_x_margin": (0, 120), "circle_y_margin": (0, 120), "circle_r_margin": (0, 160),
            "circle_min_radius": (2, 350), "circle_max_radius": (4, 600),
            "circle_merge_center": (0, 200), "circle_merge_radius": (0, 250),
            "circle_edge_contrast": (1, 180), "circle_edge_support": (0, 100),
            "circle_quadrant_support": (0, 100),
            "rect_threshold": (100, 80000), "rect_min_side": (10, 700),
            "rect_max_angle_error": (5, 60), "rect_edge_contrast": (1, 180),
            "rect_edge_support": (0, 100), "rect_corner_support": (0, 100),
            "rect_circle_aspect_max": (100, 250), "rect_circle_center_ratio": (5, 80),
            "rect_circle_diam_min": (20, 120), "rect_circle_diam_max": (60, 180),
            "max_results": (1, 20), "param_idx": (0, 20),
        }
        for k, rg in int_ranges.items():
            try:
                gd[k] = int(clamp(int(gd.get(k, gd[k])), rg[0], rg[1]))
            except Exception:
                pass
        if gd["circle_max_radius"] < gd["circle_min_radius"]:
            gd["circle_max_radius"] = gd["circle_min_radius"]
        gd["smart_filter"] = bool(gd.get("smart_filter", True))
        gd["circle_verify"] = bool(gd.get("circle_verify", True))
        gd["rect_circle_veto"] = bool(gd.get("rect_circle_veto", True))
        gd["show_center"] = bool(gd.get("show_center", True))
        gd["show_angle"] = bool(gd.get("show_angle", True))
        cfg["geometry"] = gd

        # 模板识别配置。
        t = cfg.get("template", {})
        td = default_config()["template"]
        if isinstance(t, dict):
            deep_update(td, t)
        try:
            td["selected"] = int(clamp(int(td.get("selected", 0)), 0, TEMPLATE_SLOT_COUNT - 1))
            td["threshold"] = float(clamp(float(td.get("threshold", 0.72)), 0.10, 0.99))
            td["step"] = int(clamp(int(td.get("step", 2)), 1, 8))
        except Exception:
            td["selected"], td["threshold"], td["step"] = 0, 0.72, 2
        if td.get("search") not in TEMPLATE_SEARCH_OPTIONS:
            td["search"] = "exhaustive"
        td["show_center"] = bool(td.get("show_center", True))
        td["show_box"] = bool(td.get("show_box", True))
        cfg["template"] = td

        # 相机方向配置。只使用两边官方资料都明确给出的镜像/翻转接口。
        cam = cfg.get("camera", {})
        cd = default_config()["camera"]
        if isinstance(cam, dict):
            deep_update(cd, cam)
        cd["hmirror"] = bool(cd.get("hmirror", False))
        cd["vflip"] = bool(cd.get("vflip", False))
        cfg["camera"] = cd

        # 测量/标定配置。标定比例统一保存为 mm/像素，显示时再换算成 mm 或 cm。
        m = cfg.get("measurement", {})
        if not isinstance(m, dict):
            m = {}
        md = default_config()["measurement"]
        deep_update(md, m)
        md["enabled"] = bool(md.get("enabled", False))
        if md.get("source") not in MEASURE_SOURCE_OPTIONS:
            md["source"] = "current"
        if md.get("target_rule") not in MEASURE_RULE_OPTIONS:
            md["target_rule"] = "center"
        if md.get("reference") not in MEASURE_REF_OPTIONS:
            md["reference"] = "screen_center"
        if md.get("unit") not in MEASURE_UNIT_OPTIONS:
            md["unit"] = "pixel"
        try:
            md["custom_x"] = int(clamp(int(md.get("custom_x", DISPLAY_W // 2)), 0, DISPLAY_W - 1))
            md["custom_y"] = int(clamp(int(md.get("custom_y", DISPLAY_H // 2)), 0, DISPLAY_H - 1))
            md["known_mm"] = float(clamp(float(md.get("known_mm", 100.0)), 1.0, 10000.0))
            md["mm_per_pixel"] = float(max(0.0, float(md.get("mm_per_pixel", 0.0))))
        except Exception:
            md["custom_x"], md["custom_y"] = DISPLAY_W // 2, DISPLAY_H // 2
            md["known_mm"], md["mm_per_pixel"] = 100.0, 0.0
        for k in ("show_center", "show_offset", "show_size", "show_area"):
            md[k] = bool(md.get(k, md.get(k, False)))
        for pk in ("calibration_p1", "calibration_p2"):
            p = md.get(pk, [0, 0])
            try:
                md[pk] = [int(clamp(int(p[0]), 0, DISPLAY_W - 1)),
                          int(clamp(int(p[1]), 0, DISPLAY_H - 1))]
            except Exception:
                md[pk] = [0, 0]
        cfg["measurement"] = md

        # 场景配置必须同时保留“槽位”和“加载后执行功能”。
        # 旧逻辑只保留 selected，会导致场景文件虽然写入 entry_page，
        # 经过 normalize 后却又丢失，最终所有场景加载都退回颜色识别。
        scene = cfg.get("scene", {})
        if not isinstance(scene, dict):
            scene = {}
        try:
            scene_selected = int(scene.get("selected", 0))
        except Exception:
            scene_selected = 0
        entry_page = str(scene.get("entry_page", "color"))
        if entry_page not in SCENE_PAGE_OPTIONS:
            entry_page = "color"
        cfg["scene"] = {
            "selected": int(clamp(scene_selected, 0, SCENE_SLOT_COUNT - 1)),
            "entry_page": entry_page,
        }
        return cfg

    def load(self):
        cfg = default_config()
        loaded_from = None
        last_error = None

        for path in (self.path,) + LEGACY_CONFIG_PATHS:
            try:
                with open(path, "r") as f:
                    loaded = ujson.loads(f.read())
                deep_update(cfg, loaded)
                loaded_from = path
                break
            except Exception as e:
                last_error = e

        cfg = self._normalize(cfg)

        if loaded_from == self.path:
            print("[CFG] loaded:", self.path)
        elif loaded_from is not None:
            print("[CFG] 已迁移旧配置:", loaded_from)
        else:
            print("[CFG] 使用默认配置:", last_error)
        return cfg

    def save(self, cfg):
        tmp = self.path + ".tmp"
        try:
            remember_yolo_runtime(cfg["yolo"])
            with open(tmp, "w") as f:
                f.write(ujson.dumps(cfg))
            try:
                os.remove(self.path)
            except Exception:
                pass
            os.rename(tmp, self.path)
            try:
                os.sync()
            except Exception:
                pass
            print("[CFG] saved:", self.path)
            return True
        except Exception as e:
            print("[CFG] save failed:", e)
            return False


# ============================================================
# 2. 多帧 LAB 训练器
# ============================================================

class LABTrainer:
    """
    多帧 LAB 训练器：
    - 使用多帧分位数，而不是单帧/单像素；
    - 对每个边界做截尾均值，压掉偶发曝光变化；
    - 普通颜色和巡线可使用不同分位点/余量。
    """
    def __init__(self, frame_count=6, low_q=0.15, high_q=0.85,
                 l_margin=4, ab_margin=7):
        self.frame_count = int(frame_count)
        self.low_q = float(low_q)
        self.high_q = float(high_q)
        self.l_margin = int(l_margin)
        self.ab_margin = int(ab_margin)
        self.active = False
        self.roi = None
        self.token = None
        self.count = 0
        self.samples = []
        self.last_error = ""

    def cancel(self):
        self.active = False
        self.roi = None
        self.token = None
        self.count = 0
        self.samples = []

    def start_from_display(self, x, y, token=None, roi_size=36):
        if x < 0 or x >= PANEL_X or y < 0 or y >= DISPLAY_H:
            return False
        cx = int(x * CV_W / DISPLAY_W)
        cy = int(y * CV_H / DISPLAY_H)
        half = max(4, int(roi_size) // 2)
        rx = int(clamp(cx - half, 0, CV_W - 4))
        ry = int(clamp(cy - half, 0, CV_H - 4))
        rw = int(min(int(roi_size), CV_W - rx))
        rh = int(min(int(roi_size), CV_H - ry))
        if rw < 8 or rh < 8:
            return False
        self.active = True
        self.roi = (rx, ry, rw, rh)
        self.token = token
        self.count = 0
        self.samples = []
        self.last_error = ""
        return True

    def _trimmed_value(self, values):
        if not values:
            return 0
        vals = list(values)
        vals.sort()
        if len(vals) >= 5:
            vals = vals[1:-1]
        return int(round(sum(vals) / float(len(vals))))

    def feed(self, img):
        if not self.active or self.roi is None:
            return None
        try:
            hist = img.get_histogram(roi=self.roi)
            plo = hist.get_percentile(self.low_q)
            phi = hist.get_percentile(self.high_q)
            vals = [
                int(plo.l_value()), int(phi.l_value()),
                int(plo.a_value()), int(phi.a_value()),
                int(plo.b_value()), int(phi.b_value()),
            ]
            self.samples.append(vals)
            self.count += 1
        except Exception as e:
            self.last_error = str(e)
            self.cancel()
            return (False, None, None, self.last_error)

        if self.count < self.frame_count:
            return None

        cols = [[], [], [], [], [], []]
        for s in self.samples:
            for i in range(6):
                cols[i].append(s[i])
        base = [self._trimmed_value(c) for c in cols]

        lab = [
            int(clamp(base[0] - self.l_margin, 0, 100)),
            int(clamp(base[1] + self.l_margin, 0, 100)),
            int(clamp(base[2] - self.ab_margin, -128, 127)),
            int(clamp(base[3] + self.ab_margin, -128, 127)),
            int(clamp(base[4] - self.ab_margin, -128, 127)),
            int(clamp(base[5] + self.ab_margin, -128, 127)),
        ]

        # 防止训练结果退化成过宽阈值。过宽通常意味着 ROI 同时采到了赛道和大块背景。
        if (lab[1] - lab[0] > 88 or lab[3] - lab[2] > 210 or lab[5] - lab[4] > 210):
            token = self.token
            self.cancel()
            return (False, token, None, "采样区域颜色过杂，请点目标中心并缩小取样区域")

        token = self.token
        roi = self.roi
        self.cancel()
        return (True, token, lab, roi)

    def draw_roi(self, osd, color=COLOR_YELLOW):
        if not self.active or self.roi is None:
            return
        x, y, w, h = self.roi
        dx = int(x * DISPLAY_W / CV_W)
        dy = int(y * DISPLAY_H / CV_H)
        dw = max(1, int(w * DISPLAY_W / CV_W))
        dh = max(1, int(h * DISPLAY_H / CV_H))
        osd.draw_rectangle(dx, dy, dw, dh, color=color, thickness=3)


# ============================================================
# 3. UI Button / ButtonPool
# ============================================================

class Button:
    def __init__(self):
        self.active = False

    def set(self, x, y, w, h, text, action, payload=None,
            press_mode=PRESS_TAP, bg=COLOR_DARK2, fg=COLOR_WHITE,
            border=COLOR_GRAY, hold_ms=TOUCH_HOLD_MS):
        self.active = True
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.text = text
        self.action = action
        self.payload = payload
        self.press_mode = press_mode
        self.bg, self.fg, self.border = bg, fg, border
        self.hold_ms = hold_ms
        return self

    def contains(self, x, y):
        return (self.active and x >= self.x and x < self.x + self.w and
                y >= self.y and y < self.y + self.h)

    def snapshot(self):
        # 保存不可变触摸语义，UI 重绘复用 Button 对象时不会改变当前手势。
        return (self.x, self.y, self.w, self.h, self.text,
                self.action, self.payload, self.press_mode, self.hold_ms)

    def draw(self, img, offset_x=0, offset_y=0):
        if not self.active:
            return
        x = self.x - offset_x
        y = self.y - offset_y
        img.draw_rectangle(x, y, self.w, self.h, color=self.bg, fill=True)
        img.draw_rectangle(x, y, self.w, self.h, color=self.border, thickness=2)
        fs = 19 if self.w < 120 else 20
        img.draw_string_advanced(x + 6, y + max(5, (self.h - fs) // 2),
                                 fs, self.text, color=self.fg)


class ButtonPool:
    def __init__(self, size=72):
        self.pool = [Button() for _ in range(size)]
        self.used = 0

    def reset(self):
        self.used = 0
        for b in self.pool:
            b.active = False

    def add(self, x, y, w, h, text, action, payload=None,
            press_mode=PRESS_TAP, bg=COLOR_DARK2, fg=COLOR_WHITE,
            border=COLOR_GRAY, hold_ms=TOUCH_HOLD_MS):
        if self.used >= len(self.pool):
            print("[UI] button pool full")
            return None
        b = self.pool[self.used]
        self.used += 1
        return b.set(x, y, w, h, text, action, payload, press_mode,
                     bg, fg, border, hold_ms)

    def hit_snapshot(self, x, y):
        for i in range(self.used):
            b = self.pool[i]
            if b.contains(x, y):
                return b.snapshot()
        return None

    def draw_all(self, img, offset_x=0, offset_y=0):
        for i in range(self.used):
            self.pool[i].draw(img, offset_x, offset_y)


# ============================================================
# 4. Touch Gesture Controller
# ============================================================

class TouchController:
    """
    触摸状态机：
    - TAP：只在“抬起”时提交一次，绝不在 DOWN/MOVE 阶段反复执行；
    - REPEAT：按下先执行一次，持续按住后连发；
    - HOLD：达到长按阈值执行一次；
    - EVENT_UP 为首选释放依据；空元组仅作为兼容兜底，并要求较长稳定空窗；
    - TAP 完成后设置 re-arm 时间，抑制电容屏同一次接触的尾部抖动。
    """
    def __init__(self, app, touch):
        self.app = app
        self.touch = touch
        self.active = False
        self.button = None
        self.canvas_page = None
        self.down_x = 0
        self.down_y = 0
        self.last_x = 0
        self.last_y = 0
        self.down_ms = 0
        self.last_repeat_ms = 0
        self.repeat_started = False
        self.hold_fired = False
        self.empty_since = 0
        self.ignore_until = 0
        self.error_count = 0
        self.last_tap_key = None
        self.last_tap_ms = 0

        # YOLO模型导航专用物理触摸锁。
        # 一次切模提交后，在“模型重建完成 + 屏幕持续无触点”之前，
        # 所有新的模型切换触摸都被忽略，防止同一次物理接触被拆成两次。
        self.yolo_nav_locked = False
        self.yolo_unlock_empty_since = 0

    def _inside_snapshot(self, snap, x, y):
        if snap is None:
            return False
        bx, by, bw, bh = snap[0], snap[1], snap[2], snap[3]
        return x >= bx and x < bx + bw and y >= by and y < by + bh

    def _debug(self, *args):
        if self.app.cfg["system"].get("touch_debug", False):
            print(*args)

    def _ignored(self, now):
        return safe_ticks_diff(self.ignore_until, now) > 0

    def force_rearm(self, duration_ms=TOUCH_REARM_MS):
        # 必须使用“当前真实时间”重新武装。YOLO rebuild 可能耗时数百毫秒，
        # 不能继续使用进入触摸回调前保存的旧 now。
        self.ignore_until = time.ticks_ms() + int(duration_ms)

    def _is_yolo_model_button(self, snap=None):
        s = self.button if snap is None else snap
        return (s is not None and s[5] == "yolo_model_set")

    def _lock_yolo_navigation(self):
        self.yolo_nav_locked = True
        self.yolo_unlock_empty_since = 0
        self._debug("[TOUCH] YOLO nav locked")

    def _service_yolo_navigation_lock(self, points, now):
        """
        返回 True 表示当前触摸应被 YOLO 导航锁吞掉。

        解锁条件不是固定毫秒，而是：
        1) 模型重建已经结束；
        2) 当前没有其它延迟重任务；
        3) 屏幕连续 TOUCH_YOLO_UNLOCK_QUIET_MS 毫秒没有任何触点。

        所以即使同一次物理点击被驱动拆成 DOWN/UP/DOWN/UP，
        只要手指尾部还在或模型仍在重建，都绝不可能再次触发切模。
        """
        if not self.yolo_nav_locked:
            return False

        if self.app.pending_yolo_rebuild or self.app.deferred_busy:
            self.yolo_unlock_empty_since = 0
            return True

        if points and len(points) > 0:
            self.yolo_unlock_empty_since = 0
            return True

        if self.yolo_unlock_empty_since == 0:
            self.yolo_unlock_empty_since = now
            return True

        if safe_ticks_diff(now, self.yolo_unlock_empty_since) >= TOUCH_YOLO_UNLOCK_QUIET_MS:
            self.yolo_nav_locked = False
            self.yolo_unlock_empty_since = 0
            self.ignore_until = now + 80
            self._debug("[TOUCH] YOLO nav unlocked")
            return True

        return True

    def _begin(self, x, y, event, now):
        if self._ignored(now):
            return
        self.active = True
        self.down_x = self.last_x = int(x)
        self.down_y = self.last_y = int(y)
        self.down_ms = now
        self.last_repeat_ms = now
        self.repeat_started = False
        self.hold_fired = False
        self.empty_since = 0
        self.button = self.app.ui.hit_snapshot(x, y)
        self.canvas_page = None if self.button is not None else self.app.page

        # YOLO 模型切换不依赖 event。
        # 庐山派3.1寸屏的 event 在当前文档中并不可靠，只依赖触点存在和坐标。
        if self._is_yolo_model_button() and self.yolo_nav_locked:
            self.active = False
            self.button = None
            self.canvas_page = None
            self.empty_since = 0
            return

        if self.button is not None:
            text = self.button[4]
            action = self.button[5]
            payload = self.button[6]
            mode = self.button[7]
            self._debug("[TOUCH] begin", text, mode, action, payload)
            if mode == PRESS_REPEAT:
                # 数值按钮：短按一次就应立即变化。
                self.app.on_action(action, payload)
            # TAP 等 UP 再提交，HOLD 等时间到。
        else:
            self._debug("[TOUCH] canvas begin", self.canvas_page, x, y)

    def _update(self, x, y, event, now):
        self.last_x = int(x)
        self.last_y = int(y)
        self.empty_since = 0
        if self.button is None:
            return
        if not self._inside_snapshot(self.button, x, y):
            return

        action = self.button[5]
        payload = self.button[6]
        mode = self.button[7]
        hold_ms = self.button[8]
        elapsed = safe_ticks_diff(now, self.down_ms)

        if mode == PRESS_REPEAT:
            if elapsed >= TOUCH_REPEAT_DELAY_MS:
                if (not self.repeat_started or
                        safe_ticks_diff(now, self.last_repeat_ms) >= TOUCH_REPEAT_INTERVAL_MS):
                    self.repeat_started = True
                    self.last_repeat_ms = now
                    self.app.on_action(action, payload)

        elif mode == PRESS_HOLD:
            if not self.hold_fired and elapsed >= hold_ms:
                self.hold_fired = True
                self.app.on_action(action, payload)
                self.app.notice("长按操作已执行", COLOR_ORANGE, 1200)

    def _movement_ok(self):
        dx = self.last_x - self.down_x
        dy = self.last_y - self.down_y
        return (dx * dx + dy * dy) <= TOUCH_MOVE_TOL * TOUCH_MOVE_TOL

    def _tap_is_duplicate(self, action, payload, now):
        # 部分触摸固件会把一次接触尾部拆成第二次短触摸。
        # 这里只去重“同一个动作+同一个参数”的极短重复。
        key = (str(action), repr(payload))
        if self.last_tap_key == key and safe_ticks_diff(now, self.last_tap_ms) < TOUCH_TAP_DEDUP_MS:
            self._debug("[TOUCH] duplicate tap ignored", action, payload)
            return True
        self.last_tap_key = key
        self.last_tap_ms = now
        return False

    def _commit_tap(self):
        # TAP / CONFIRM 都只在一次完整接触结束时提交一次。
        if not self.active or not self._movement_ok():
            return
        if self.button is not None:
            mode = self.button[7]
            if mode in (PRESS_TAP, PRESS_CONFIRM) and self._inside_snapshot(self.button, self.last_x, self.last_y):
                action = self.button[5]
                payload = self.button[6]
                now = time.ticks_ms()
                if self._tap_is_duplicate(action, payload, now):
                    return
                self._debug("[TOUCH] TAP commit", self.button[4], action, payload, mode)

                # 关键：切模锁必须在 on_action 之前设置。
                # UI 可能马上把“下个模型”的绝对 payload 从2改成3，
                # 但锁已经生效，因此同一物理触摸的尾部不可能再次命中新 payload。
                if action == "yolo_model_set":
                    self._lock_yolo_navigation()

                if mode == PRESS_CONFIRM:
                    self.app.on_confirm_action(action, payload)
                else:
                    self.app.on_action(action, payload)
        elif self.canvas_page is not None:
            self._debug("[TOUCH] canvas TAP commit", self.canvas_page, self.last_x, self.last_y)
            self.app.on_canvas_touch(self.canvas_page, self.last_x, self.last_y)

    def _end(self, now, commit_tap=True):
        # HOLD 的关键兜底：部分触摸固件在手指静止时不会持续产生 MOVE 事件，
        # 因而不能只依赖 _update() 里的计时。抬手时再按总按压时长判定一次。
        if self.active and self.button is not None:
            try:
                mode = self.button[7]
                hold_ms = self.button[8]
                elapsed = safe_ticks_diff(now, self.down_ms)
                if (mode == PRESS_HOLD and not self.hold_fired and
                        elapsed >= hold_ms and self._movement_ok() and
                        self._inside_snapshot(self.button, self.last_x, self.last_y)):
                    self.hold_fired = True
                    self.app.on_action(self.button[5], self.button[6])
                    self.app.notice("长按操作已执行", COLOR_ORANGE, 1200)
            except Exception as e:
                print("[TOUCH HOLD END]", e)

        if self.active and commit_tap:
            self._commit_tap()
        if self.active:
            self._debug("[TOUCH] end")
        self.active = False
        self.button = None
        self.canvas_page = None
        self.repeat_started = False
        self.hold_fired = False
        self.empty_since = 0
        # 注意：_commit_tap() 里的 action 可能很慢。必须在 action 返回之后重新取时钟，
        # 否则 rebuild 花掉的时间会让防抖在函数返回前就已经过期。
        self.ignore_until = time.ticks_ms() + TOUCH_REARM_MS

    def poll(self):
        now = time.ticks_ms()
        try:
            points = self.touch.read(1)
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 5 or self.error_count % 50 == 0:
                print("[TOUCH ERROR]", e)
            return

        # 两点标定和识别区域编辑会完全接管触摸。庐山派屏幕的 event/track_id
        # 在部分固件上不稳定，因此两个编辑器都只依赖“是否有触点 + x/y坐标”。
        if self.app.calibration_manager is not None and self.app.calibration_manager.active:
            if points and len(points) > 0:
                p = points[0]
                self.app.calibration_manager.handle_event(int(p.x), int(p.y), int(p.event))
            else:
                self.app.calibration_manager.handle_empty()
            return

        if self.app.region_manager is not None and self.app.region_manager.active:
            if points and len(points) > 0:
                p = points[0]
                self.app.region_manager.handle_event(int(p.x), int(p.y), int(p.event))
            else:
                self.app.region_manager.handle_empty()
            return

        # YOLO模型切换提交后，先处理专用物理触摸锁。
        if self._service_yolo_navigation_lock(points, now):
            return

        if points and len(points) > 0:
            p = points[0]
            x, y = int(p.x), int(p.y)
            ev = int(p.event)
            self._debug("[TOUCH RAW] x=%d y=%d event=%d" % (x, y, ev))

            if self._ignored(now):
                return

            # 如果当前手势已经锁定在“上个/下个模型”按钮上：
            # 只要还能读到触点，就认为手指仍在屏幕上。
            # 完全忽略 EVENT_UP / EVENT_DOWN / EVENT_MOVE，
            # 直到后续 read() 连续返回空元组才提交一次。
            if self.active and self._is_yolo_model_button():
                self.last_x, self.last_y = x, y
                self._update(x, y, ev, now)
                return

            try:
                is_up = (ev == TOUCH.EVENT_UP)
            except Exception:
                is_up = False

            if is_up:
                if self.active:
                    self.last_x, self.last_y = x, y
                    self._end(now, commit_tap=True)
                return

            if not self.active:
                self._begin(x, y, ev, now)
            else:
                self._update(x, y, ev, now)
            return

        # 空元组才是庐山派这块屏幕最可靠的“当前没有触点”信号。
        if self.active:
            if self.empty_since == 0:
                self.empty_since = now
            else:
                elapsed_empty = safe_ticks_diff(now, self.empty_since)
                if self._is_yolo_model_button():
                    # YOLO模型按钮需要一段稳定无触点时间后，只提交一次。
                    if elapsed_empty >= TOUCH_YOLO_RELEASE_STABLE_MS:
                        self._end(now, commit_tap=True)
                elif elapsed_empty >= TOUCH_RELEASE_GAP_MS:
                    self._end(now, commit_tap=True)


# ============================================================
# 5. Shape Engine - OpenCV 多重几何约束 + 连续帧稳定确认
# ============================================================

class ShapeEngine:
    def __init__(self, app):
        self.app = app
        self.prev_tracks = []
        self.kernel = None
        if CV2_OK:
            try:
                self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            except Exception:
                self.kernel = None

    def sx(self, x):
        return int(x * DISPLAY_W / CV_W)

    def sy(self, y):
        return int(y * DISPLAY_H / CV_H)

    def _points(self, approx):
        pts = []
        try:
            n = int(approx.shape[0])
            flat = approx.reshape((n, 2))
            for i in range(n):
                pts.append((int(flat[i][0]), int(flat[i][1])))
        except Exception:
            try:
                for item in approx:
                    p = item[0] if len(item) == 1 else item
                    pts.append((int(p[0]), int(p[1])))
            except Exception:
                return []
        return pts

    def _angle(self, a, b, c):
        v1x, v1y = a[0] - b[0], a[1] - b[1]
        v2x, v2y = c[0] - b[0], c[1] - b[1]
        n1 = math.sqrt(v1x * v1x + v1y * v1y)
        n2 = math.sqrt(v2x * v2x + v2y * v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cs = (v1x * v2x + v1y * v2y) / (n1 * n2)
        cs = clamp(cs, -1.0, 1.0)
        return math.acos(cs) * 180.0 / math.pi

    def _polygon_angles(self, pts):
        out = []
        n = len(pts)
        for i in range(n):
            out.append(self._angle(pts[(i - 1) % n], pts[i], pts[(i + 1) % n]))
        return out

    def _border_like(self, x, y, w, h):
        # 排除几乎占满整帧的边界/画框轮廓。
        if w >= int(CV_W * 0.92) or h >= int(CV_H * 0.92):
            return True
        touches = 0
        if x <= 2:
            touches += 1
        if y <= 2:
            touches += 1
        if x + w >= CV_W - 2:
            touches += 1
        if y + h >= CV_H - 2:
            touches += 1
        return touches >= 3

    def _candidate_from_contour(self, cnt, cfg):
        try:
            area = float(cv2.contourArea(cnt))
            if area < cfg["min_area"]:
                return None
            if area > CV_W * CV_H * cfg["max_area_ratio"]:
                return None

            peri = float(cv2.arcLength(cnt, True))
            if peri < 30:
                return None
            approx = cv2.approxPolyDP(cnt, cfg["epsilon_ratio"] * peri, True)
            pts = self._points(approx)
            n = len(pts)
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10 or self._border_like(x, y, w, h):
                return None

            cx = x + w * 0.5
            cy = y + h * 0.5
            mode = cfg["mode"]

            # ---------- 三角形 ----------
            if n == 3 and mode in ("auto", "triangle"):
                try:
                    if not cv2.isContourConvex(approx):
                        return None
                except Exception:
                    pass
                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                solidity = area / hull_area if hull_area > 1 else 0.0
                if solidity < cfg["min_solidity"]:
                    return None
                angles = self._polygon_angles(pts)
                if len(angles) != 3:
                    return None
                if min(angles) < 18 or max(angles) > 145:
                    return None
                # 最短边过小通常是锯齿/噪声三角。
                min_side = 99999.0
                for i in range(3):
                    dx = pts[i][0] - pts[(i + 1) % 3][0]
                    dy = pts[i][1] - pts[(i + 1) % 3][1]
                    min_side = min(min_side, math.sqrt(dx * dx + dy * dy))
                if min_side < 14:
                    return None
                return {
                    "type": "triangle", "cx": cx, "cy": cy, "w": w, "h": h,
                    "pts": pts, "score": min(1.0, solidity), "area": area,
                }

            # ---------- 矩形 ----------
            if n == 4 and mode in ("auto", "rect"):
                try:
                    if not cv2.isContourConvex(approx):
                        return None
                except Exception:
                    pass
                angles = self._polygon_angles(pts)
                if len(angles) != 4:
                    return None
                # 给透视留余量，但排除明显平行四边/尖角杂形。
                if min(angles) < 50 or max(angles) > 130:
                    return None
                rr = cv2.minAreaRect(cnt)
                rw, rh = float(rr[1][0]), float(rr[1][1])
                rarea = rw * rh
                if rarea < 1:
                    return None
                fill = area / rarea
                if fill < cfg["rect_min_fill"] or fill > 1.12:
                    return None
                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                solidity = area / hull_area if hull_area > 1 else 0.0
                if solidity < cfg["min_solidity"]:
                    return None
                angle_quality = 1.0 - min(1.0, max([abs(a - 90.0) for a in angles]) / 45.0)
                return {
                    "type": "rect", "cx": cx, "cy": cy, "w": w, "h": h,
                    "pts": pts, "score": min(fill, solidity) * (0.75 + 0.25 * angle_quality),
                    "area": area,
                }

            # ---------- 圆形 ----------
            if n >= 6 and mode in ("auto", "circle"):
                circularity = 4.0 * math.pi * area / (peri * peri) if peri > 1 else 0.0
                if circularity < cfg["circle_min_circularity"]:
                    return None
                (cc, radius) = cv2.minEnclosingCircle(cnt)
                radius = float(radius)
                if radius < 8:
                    return None
                circle_area = math.pi * radius * radius
                fill = area / circle_area if circle_area > 1 else 0.0
                if fill < cfg["circle_min_fill"] or fill > 1.10:
                    return None
                aspect = w / float(h) if h > 0 else 99
                if aspect < 0.72 or aspect > 1.38:
                    return None
                return {
                    "type": "circle", "cx": float(cc[0]), "cy": float(cc[1]),
                    "w": radius * 2, "h": radius * 2, "r": radius,
                    "score": min(1.0, circularity) * min(1.0, fill), "area": area,
                }
        except Exception as e:
            # 某个轮廓异常不拖垮整帧。
            return None
        return None

    def _pixel_chroma(self, img, x, y):
        """取一个像素的"彩度"（0=纯灰，越大越艳）；读不到返回 -1。

        优先按 RGB 读，老固件不支持 rgbtuple 时退回默认格式（RGB565 图通常
        返回 LAB）。两种都折算到接近的 0~100 量级，这样只有一个门限要调。
        """
        try:
            p = img.get_pixel(x, y, rgbtuple=True)
            if isinstance(p, (tuple, list)) and len(p) >= 3:
                mx = max(int(p[0]), int(p[1]), int(p[2]))
                mn = min(int(p[0]), int(p[1]), int(p[2]))
                return float(mx - mn) * 200.0 / 255.0
        except Exception:
            pass
        try:
            p = img.get_pixel(x, y)
            if isinstance(p, (tuple, list)) and len(p) >= 3:
                a = float(p[1])
                b = float(p[2])
                v = math.sqrt(a * a + b * b)
                return v if v <= 100.0 else 100.0
            return -1.0
        except Exception:
            return -1.0

    def _ring_chroma(self, img, cx, cy, half):
        """在光点外面套两圈采样，看"光晕有没有颜色"。

        激光点的核心是过曝的白（没颜色），但它外面一圈一定是激光的颜色；
        而物体上的白色高光，外面一圈还是白的（没颜色）。
        返回最彩那一圈的彩度；读不到像素时返回 -1（= 不判断）。
        """
        best = -1.0
        for extra in (3.0, 7.0):
            r = max(2.0, half + extra)
            total = 0.0
            n = 0
            for k in range(8):
                ang = k * 0.7853981634          # 2*pi/8
                x = int(cx + r * math.cos(ang))
                y = int(cy + r * math.sin(ang))
                if x < 0 or y < 0 or x >= CV_W or y >= CV_H:
                    continue
                v = self._pixel_chroma(img, x, y)
                if v < 0.0:
                    continue
                total += v
                n += 1
            if n > 0:
                mean = total / float(n)
                if mean > best:
                    best = mean
        return best

    def _laser_candidates(self, img):
        """激光点识别：找整帧里"最亮的那个小圆点"。

        用法和颜色识别一样是 find_blobs，但门限只卡【亮度 L】，
        所以红的、绿的、紫的激光笔都能认，不用重新调颜色。
        参数里填的是 0~255 的亮度，这里换算成 LAB 的 L(0~100) 再用。
        ★白纸、灯光、反光虽然也亮，但面积大 / 密度低，会被
          laser_max_pixels 和 laser_min_density 一起滤掉。
        """
        cfg = self.app.cfg["shape"]
        out = []
        lo255 = int(clamp(cfg.get("laser_threshold", 245), 60, 255))
        l_lo = int(clamp(lo255 * 100 // 255, 1, 100))
        min_px = int(clamp(cfg.get("laser_min_pixels", 4), 1, 4000))
        max_px = int(clamp(cfg.get("laser_max_pixels", 1200), min_px, 200000))
        min_density = float(clamp(cfg.get("laser_min_density", 0.35), 0.02, 1.0))
        min_chroma = float(clamp(cfg.get("laser_min_chroma", 40), 0.0, 100.0))
        bright_margin = float(clamp(cfg.get("laser_bright_margin", 8), 0.0, 255.0))

        # L 卡下限，a/b 全放开 -> 只管亮度，不管颜色。
        thresholds = [(l_lo, 100, -128, 127, -128, 127)]
        try:
            blobs = img.find_blobs(
                thresholds, x_stride=2, y_stride=2,
                area_threshold=min_px, pixels_threshold=min_px,
                merge=True, margin=4
            )
        except Exception as e:
            print("[SHAPE] laser find_blobs error:", e)
            return out

        for b in blobs:
            try:
                px = int(b.pixels())
                if px < min_px or px > max_px:
                    continue
                w = int(b.w())
                h = int(b.h())
                # 400x240 的画面里，激光点不会超过 80 像素宽高。
                if w < 1 or h < 1 or w > 80 or h > 80:
                    continue
                density = float(b.density())
                if density < min_density:
                    continue
                aspect = w / float(h)
                if aspect < 0.33 or aspect > 3.0:
                    continue
                # ★关键：光晕彩度。激光点外面一圈是彩色的，
                #   物体上的白色高光外面还是白的 -> 直接滤掉。
                #   采样失败时 ring = -1，表示"这只固件读不到像素，不判断"，
                #   绝不能让采样异常把真正的激光点也一起丢掉。
                if min_chroma > 0.0:
                    try:
                        half = max(w, h) * 0.5
                        ring = self._ring_chroma(img, float(b.cx()), float(b.cy()), half)
                    except Exception:
                        ring = -1.0
                    if ring >= 0.0 and ring < min_chroma:
                        continue
                # 顺手把实际亮度读出来，显示在屏幕上方便调门限。
                bright = lo255
                level = float(lo255)
                try:
                    m = b.mean()
                    lv = float(m[0]) if isinstance(m, (tuple, list)) else float(m)
                    if lv <= 100.5:
                        level = lv                                  # LAB 的 L：0~100
                        bright = int(clamp(lv * 255.0 / 100.0, 0, 255))
                    else:
                        level = lv                                  # 直接给的 0~255
                        bright = int(clamp(lv, 0, 255))
                except Exception:
                    level = float(lo255)
                    bright = lo255
                out.append({
                    "type": "laser",
                    "cx": float(b.cx()), "cy": float(b.cy()),
                    "w": float(w), "h": float(h),
                    "score": min(1.0, density),
                    "area": float(px),
                    "bright": bright,
                    "level": level,
                })
            except Exception:
                continue

        # ★第二道：只认"最亮的那一批"。激光点通常就是整个画面里最亮的，
        #   其它高光哪怕勉强过了门限，只要比它暗一点就被丢掉。
        if out and bright_margin < 250.0:
            top = out[0]["level"]
            for v in out:
                if v["level"] > top:
                    top = v["level"]
            out = [v for v in out if v["level"] >= top - bright_margin]

        # 越像激光点的排前面（密度高、面积够）。
        out.sort(key=lambda v: -(v["score"] * v["area"]))
        return out[:4]

    def _stabilize(self, candidates):
        cfg = self.app.cfg["shape"]
        new_tracks = []
        for c in candidates:
            best = None
            best_d2 = 1e18
            for p in self.prev_tracks:
                if p["type"] != c["type"]:
                    continue
                dx, dy = c["cx"] - p["cx"], c["cy"] - p["cy"]
                d2 = dx * dx + dy * dy
                if d2 > cfg["stable_dist"] * cfg["stable_dist"]:
                    continue
                # 尺寸突变过大也不视作同一个候选。
                pw = max(1.0, float(p["w"]))
                ph = max(1.0, float(p["h"]))
                wr = float(c["w"]) / pw
                hr = float(c["h"]) / ph
                if wr < 0.55 or wr > 1.8 or hr < 0.55 or hr > 1.8:
                    continue
                if d2 < best_d2:
                    best_d2 = d2
                    best = p
            nc = c
            nc["hits"] = (best.get("hits", 1) + 1) if best is not None else 1
            new_tracks.append(nc)
        self.prev_tracks = new_tracks
        return [c for c in new_tracks if c["hits"] >= cfg["stable_frames"]]

    def _draw_polygon(self, osd, pts, color):
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            osd.draw_line(self.sx(a[0]), self.sy(a[1]), self.sx(b[0]), self.sy(b[1]),
                          color=color, thickness=3)

    def process(self, img, osd):
        osd.clear()
        if not CV2_OK:
            self.app.results.clear("shape")
            draw_short_status(osd, "未检测到cv2：请确认固件支持OpenCV", COLOR_RED)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            return

        cfg = self.app.cfg["shape"]
        mode = cfg["mode"]
        candidates = []

        # ---------- 激光点：亮度门限 + find_blobs，和形状算法完全独立 ----------
        laser_candidates = []
        if (mode == "laser") or (int(cfg.get("laser_overlay", 1)) != 0):
            try:
                laser_candidates = self._laser_candidates(img)
            except Exception as e:
                print("[SHAPE] laser error:", e)
                laser_candidates = []

        if mode != "laser":
            try:
                rgb = img.to_rgb888()
                arr = rgb.to_numpy_ref()
                # 形状算法保持原有整帧 OpenCV 流程，识别区域在候选稳定后做中心过滤。
                # 这样不改变现有 cv2/ulab 内存路径，优先保证兼容性和稳定性。
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, cfg["canny_low"], cfg["canny_high"])
                if self.kernel is not None:
                    try:
                        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel)
                    except Exception:
                        pass
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    c = self._candidate_from_contour(cnt, cfg)
                    if c is not None:
                        candidates.append(c)

                candidates.sort(key=lambda v: -(v["score"] * math.sqrt(max(1.0, v["area"]))))
                if len(candidates) > cfg["max_results"] * 2:
                    candidates = candidates[:cfg["max_results"] * 2]
            except Exception as e:
                print("[SHAPE] OpenCV error:", e)
                self.app.results.clear("shape")
                draw_short_status(osd, "形状处理异常: " + str(e), COLOR_RED)
                self.prev_tracks = []
                self.app.draw_runtime_decor(osd)
                Display.show_image(osd, layer=Display.LAYER_OSD0)
                return

        stable = self._stabilize(candidates)
        # 激光点不做"连续帧稳定"判定：激光笔经常在动，等两帧就抓不到了。
        if laser_candidates:
            stable = stable + laser_candidates
        if mode == "laser":
            stable = [c for c in stable if c["type"] == "laser"]
        if self.app.region_manager is not None and self.app.region_manager.enabled("shape"):
            stable = [c for c in stable if self.app.region_manager.contains_display(
                "shape", self.sx(c["cx"]), self.sy(c["cy"]))]
        if len(stable) > cfg["max_results"]:
            stable = stable[:cfg["max_results"]]

        targets = []
        type_ids = {"triangle": 0, "rect": 1, "circle": 2, "laser": 3}
        type_names = {"triangle": "三角形", "rect": "矩形", "circle": "圆形", "laser": "激光点"}
        for i, c in enumerate(stable):
            color = PALETTE[i % len(PALETTE)]
            is_laser = (c["type"] == "laser")
            label = type_names.get(c["type"], c["type"])
            if is_laser:
                # 激光点：画圈 + 十字准星；标签显示实际亮度，方便对着调门限
                lx = self.sx(c["cx"])
                ly = self.sy(c["cy"])
                lr = max(9, self.sx(c["w"]) // 2 + 10)
                osd.draw_circle(lx, ly, lr, color=color, thickness=3)
                draw_cross_lines(osd, lx, ly, color=color, size=lr + 6, thickness=2)
                osd.draw_string_advanced(lx - 42, safe_label_y(ly - lr, lr * 2), 18,
                                         "%s %d" % (label, int(c.get("bright", 0))),
                                         color=color)
            else:
                if c["type"] in ("triangle", "rect"):
                    self._draw_polygon(osd, c["pts"], color)
                else:
                    osd.draw_circle(self.sx(c["cx"]), self.sy(c["cy"]),
                                    max(2, self.sx(c["r"])), color=color, thickness=3)
                dx = self.sx(c["cx"] - c["w"] * 0.5)
                dy = self.sy(c["cy"] - c["h"] * 0.5)
                dh = self.sy(c["h"])
                osd.draw_string_advanced(dx, safe_label_y(dy, dh), 18,
                                         "%s %.2f" % (label, c["score"]), color=color)
            targets.append({
                "id": i,
                "class_id": type_ids.get(c["type"], 0),
                "label": label,
                "score": float(c.get("score", 0.0)),
                "cx": self.sx(c["cx"]),
                "cy": self.sy(c["cy"]),
                "w": max(1, self.sx(c["w"])),
                "h": max(1, self.sy(c["h"])),
                "area": float(c.get("area", 0.0)) * 4.0,
                "angle": 0.0,
            })

        self.app.results.publish("shape", targets, {"mode": cfg["mode"]})
        if not stable:
            if mode == "laser":
                draw_short_status(osd, "没找到激光点：把“激光亮度门限”调低一点", COLOR_YELLOW)
            else:
                draw_short_status(osd, "等待稳定形状", COLOR_YELLOW)
        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)

# ============================================================
# 6. Color Engine - 独立目标 + 多帧训练 + 多色并行
# ============================================================

class ColorEngine:
    def __init__(self, app):
        self.app = app
        self.trainer = LABTrainer(6, 0.15, 0.85, 4, 7)
        self.sample_armed = False
        self.frame_id = 0
        self.last_ui_adapt_ms = 0

    def sx(self, x):
        return int(x * DISPLAY_W / CV_W)

    def sy(self, y):
        return int(y * DISPLAY_H / CV_H)

    def cancel_training(self):
        self.trainer.cancel()
        self.sample_armed = False

    def arm_training(self):
        self.trainer.cancel()
        self.sample_armed = True
        self.app.notice("请在左侧画面单击目标颜色", COLOR_YELLOW, 2200)

    def request_training_at(self, x, y):
        if not self.sample_armed:
            return False
        idx = self.app.cfg["color"]["current"]
        ok = self.trainer.start_from_display(x, y, token=idx, roi_size=34)
        if ok:
            self.sample_armed = False
            self.app.notice("正在采样目标%d：保持画面稳定" % (idx + 1), COLOR_YELLOW, 1400)
            return True
        return False

    def reset_current(self):
        c = self.app.cfg["color"]
        idx = c["current"]
        c["profiles"][idx] = default_profile(idx)
        self.cancel_training()

    def _sample_lab_roi(self, img, roi):
        try:
            hist = img.get_histogram(roi=roi)
            p10 = hist.get_percentile(0.10)
            p90 = hist.get_percentile(0.90)
            return [
                clamp(int(p10.l_value()) - 4, 0, 100),
                clamp(int(p90.l_value()) + 4, 0, 100),
                clamp(int(p10.a_value()) - 7, -128, 127),
                clamp(int(p90.a_value()) + 7, -128, 127),
                clamp(int(p10.b_value()) - 7, -128, 127),
                clamp(int(p90.b_value()) + 7, -128, 127),
            ]
        except Exception:
            return None

    def _bounded_adapt(self, old, new, step):
        out = []
        for i in range(6):
            d = new[i] - old[i]
            d = clamp(d, -step, step)
            out.append(int(old[i] + d))
        # 保证 low <= high。
        if out[0] > out[1]:
            out[0], out[1] = out[1], out[0]
        if out[2] > out[3]:
            out[2], out[3] = out[3], out[2]
        if out[4] > out[5]:
            out[4], out[5] = out[5], out[4]
        return out

    def process(self, img, osd):
        cfg = self.app.cfg["color"]
        osd.clear()
        self.frame_id += 1

        train_result = self.trainer.feed(img)
        self.trainer.draw_roi(osd)
        if train_result is not None:
            ok, token, lab, extra = train_result
            if ok:
                idx = int(token)
                if 0 <= idx < len(cfg["profiles"]):
                    p = cfg["profiles"][idx]
                    p["lab"] = copy_lab(lab)
                    p["trained"] = True
                    p["enabled"] = True
                    p["auto"] = False
                    self.app.notice("目标%d训练完成" % (idx + 1), COLOR_GREEN, 1800)
                    self.app.ui.invalidate()
                    self.app.store.save(self.app.cfg)
            else:
                self.app.notice("取色失败: " + str(extra), COLOR_RED, 2200)

        active_profiles = []
        thresholds = []
        for idx, p in enumerate(cfg["profiles"]):
            if p.get("trained", False) and p.get("enabled", False):
                active_profiles.append(idx)
                thresholds.append(tuple(p["lab"]))

        if not thresholds:
            self.app.results.clear("color")
            if self.sample_armed:
                draw_short_status(osd, "取色已就绪：单击目标", COLOR_YELLOW)
            elif self.trainer.active:
                draw_short_status(osd, "LAB采样 %d/%d" % (self.trainer.count, self.trainer.frame_count), COLOR_YELLOW)
            else:
                draw_short_status(osd, "没有启用且已训练的颜色目标", COLOR_YELLOW)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            return

        roi = self.app.region_manager.cv_roi("color") if self.app.region_manager is not None else None
        try:
            if roi is not None:
                blobs = img.find_blobs(
                    thresholds, roi=roi,
                    x_stride=cfg["x_stride"], y_stride=cfg["y_stride"],
                    area_threshold=cfg["area_threshold"],
                    pixels_threshold=cfg["pixels_threshold"], merge=False
                )
            else:
                blobs = img.find_blobs(
                    thresholds,
                    x_stride=cfg["x_stride"], y_stride=cfg["y_stride"],
                    area_threshold=cfg["area_threshold"],
                    pixels_threshold=cfg["pixels_threshold"], merge=False
                )
        except Exception as e:
            self.app.results.clear("color")
            draw_short_status(osd, "颜色识别异常: " + str(e), COLOR_RED)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            return

        largest_by_profile = {}
        targets = []
        shown = 0
        for b in blobs:
            if shown >= cfg["max_results"]:
                break
            try:
                density = float(b.density())
                if density < cfg["min_density"]:
                    continue
                code = int(b.code())
                local_idx = -1
                for bit in range(len(active_profiles)):
                    if code & (1 << bit):
                        local_idx = bit
                        break
                if local_idx < 0:
                    continue
                profile_idx = active_profiles[local_idx]
                p = cfg["profiles"][profile_idx]
                color = PALETTE[profile_idx % len(PALETTE)]
                x, y, w, h = b.rect()
                dx, dy = self.sx(x), self.sy(y)
                dw, dh = max(1, self.sx(w)), max(1, self.sy(h))
                cx, cy = self.sx(b.cx()), self.sy(b.cy())
                osd.draw_rectangle(dx, dy, dw, dh, color=color, thickness=3)
                draw_cross_lines(osd, cx, cy, color=color, size=7)
                osd.draw_string_advanced(dx, safe_label_y(dy, dh), 18,
                                         "%s %.2f" % (p["name"], density), color=color)
                old = largest_by_profile.get(profile_idx)
                if old is None or b.pixels() > old.pixels():
                    largest_by_profile[profile_idx] = b
                targets.append({
                    "id": shown,
                    "class_id": profile_idx,
                    "label": str(p["name"]),
                    "score": density,
                    "cx": cx,
                    "cy": cy,
                    "w": dw,
                    "h": dh,
                    "area": float(b.pixels()) * 4.0,
                    "angle": 0.0,
                })
                shown += 1
            except Exception:
                pass

        if self.frame_id % max(1, cfg["adapt_interval"]) == 0:
            changed_current = False
            for profile_idx in largest_by_profile:
                p = cfg["profiles"][profile_idx]
                if not p.get("auto", False):
                    continue
                b = largest_by_profile[profile_idx]
                new_lab = self._sample_lab_roi(img, b.rect())
                if new_lab is not None:
                    p["lab"] = self._bounded_adapt(p["lab"], new_lab, cfg["adapt_step"])
                    if profile_idx == cfg["current"]:
                        changed_current = True
            if changed_current:
                now = time.ticks_ms()
                if safe_ticks_diff(now, self.last_ui_adapt_ms) >= 500:
                    self.last_ui_adapt_ms = now
                    self.app.ui.invalidate()

        self.app.results.publish("color", targets, {"active_profiles": active_profiles})
        if self.sample_armed:
            draw_short_status(osd, "取色已就绪：单击目标", COLOR_YELLOW)
        elif self.trainer.active:
            draw_short_status(osd, "LAB采样 %d/%d" % (self.trainer.count, self.trainer.frame_count), COLOR_YELLOW)

        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)

# ============================================================
# 7. Line Following Engine - 三算法 + 独立颜色训练
# ============================================================

class LineFollowerEngine:
    """
    三算法巡线：
    1) ROI：底部起种 + 动态走廊 + 连续性评分 + 多窗口中心拟合，适合弯道；
    2) REG：底部起种后限制回归走廊，再调用官方 get_regression(robust=True)，
       避免全画面同色干扰把回归线拉偏；
    3) 8N：二值化后做“八邻域连通 + 色带中心化 + 断线恢复”，不再逐像素蛇形抖动。
    三种算法共用独立巡线 LAB，并使用统一的平滑/丢线策略。
    """
    def __init__(self, app):
        self.app = app
        self.trainer = LABTrainer(7, 0.18, 0.82, 3, 6)
        self.sample_armed = False
        self.smooth_error = None
        self.smooth_angle = None
        self.last_quality = 0.0
        self.track_x = None
        self.lost_frames = 0

    def sx(self, x):
        return int(x * DISPLAY_W / CV_W)

    def sy(self, y):
        return int(y * DISPLAY_H / CV_H)

    def reset_track(self):
        self.smooth_error = None
        self.smooth_angle = None
        self.track_x = None
        self.lost_frames = 0

    def cancel_training(self):
        self.sample_armed = False
        self.trainer.cancel()

    def arm_training(self):
        self.trainer.cancel()
        self.sample_armed = True
        self.app.notice("请点左侧赛道颜色中心；尽量别点到背景", COLOR_YELLOW, 2600)

    def request_training_at(self, x, y):
        if not self.sample_armed:
            return False
        # 巡线通常较细，大取样区域很容易把背景一起训练进去。
        # 取样区域缩到 18x18，并使用 7 帧截尾采样。
        ok = self.trainer.start_from_display(x, y, token="line", roi_size=18)
        if ok:
            self.sample_armed = False
            self.app.notice("正在采样巡线颜色 7 帧，请保持画面稳定", COLOR_YELLOW, 1700)
            return True
        return False

    def clear_training(self):
        cfg = self.app.cfg["line"]
        cfg["trained"] = False
        cfg["lab"] = copy_lab(FULL_LAB)
        self.cancel_training()
        self.reset_track()

    def _smooth(self, err, angle):
        alpha = float(self.app.cfg["line"]["smooth_alpha"])
        if self.smooth_error is None:
            self.smooth_error = float(err)
            self.smooth_angle = float(angle)
        else:
            self.smooth_error = alpha * float(err) + (1.0 - alpha) * self.smooth_error
            self.smooth_angle = alpha * float(angle) + (1.0 - alpha) * self.smooth_angle
        return self.smooth_error, self.smooth_angle

    def _draw_result_text(self, osd, err, angle, quality, mode):
        txt = "%s  E:%+.1f  A:%+.1f  Q:%.2f" % (mode, err, angle, quality)
        draw_short_status(osd, txt, COLOR_CYAN)

    def _blob_score(self, b, predicted_x, continuity, min_density):
        try:
            den = float(b.density())
            if den < min_density:
                return None
            px = float(b.cx())
            dist = 0.0 if predicted_x is None else abs(px - predicted_x)
            if predicted_x is not None and dist > continuity:
                return None
            # 像素量是主项，距离预测中心是强约束，密度用于压掉稀疏噪声。
            return float(b.pixels()) * (0.65 + 0.35 * min(1.0, den)) - dist * 7.0
        except Exception:
            return None

    def _choose_blob(self, blobs, predicted_x, continuity, min_density):
        best = None
        best_score = None
        for b in blobs:
            s = self._blob_score(b, predicted_x, continuity, min_density)
            if s is None:
                continue
            if best is None or s > best_score:
                best = b
                best_score = s
        return best

    def _fit_x_by_y(self, points):
        # x = a*y + b，先最小二乘，再按残差做一次轻量裁剪。
        if len(points) < 2:
            return None

        def fit(pts):
            n = float(len(pts))
            sy = sx = syy = syx = 0.0
            for x, y in pts:
                sx += x
                sy += y
                syy += y * y
                syx += y * x
            den = n * syy - sy * sy
            if abs(den) < 1e-6:
                return None
            a = (n * syx - sy * sx) / den
            b = (sx - a * sy) / n
            return a, b

        first = fit(points)
        if first is None:
            return None
        a, b = first
        gate = max(14.0, float(self.app.cfg["line"]["continuity_px"]) * 0.55)
        trimmed = []
        for x, y in points:
            if abs(x - (a * y + b)) <= gate:
                trimmed.append((x, y))
        if len(trimmed) >= max(3, len(points) - 2):
            second = fit(trimmed)
            if second is not None:
                a, b = second
                points = trimmed
        return a, b, points

    def _result_from_points(self, points, expected, osd, color=COLOR_CYAN):
        if len(points) < 2:
            return None
        fit = self._fit_x_by_y(points)
        if fit is None:
            return None
        a, b, used = fit
        bottom_y = CV_H - 1
        look_y = int(CV_H * 0.48)
        x_bottom = clamp(a * bottom_y + b, 0, CV_W - 1)
        x_look = clamp(a * look_y + b, 0, CV_W - 1)
        err = x_bottom - CV_W * 0.5
        angle = math.atan2(x_look - x_bottom, max(1.0, bottom_y - look_y)) * 180.0 / math.pi
        quality = clamp(len(used) / float(max(1, expected)), 0.0, 1.0)

        used_sorted = list(used)
        used_sorted.sort(key=lambda p: -p[1])
        for i in range(len(used_sorted) - 1):
            a0, a1 = used_sorted[i], used_sorted[i + 1]
            osd.draw_line(self.sx(a0[0]), self.sy(a0[1]), self.sx(a1[0]), self.sy(a1[1]),
                          color=color, thickness=3)
        draw_cross_lines(osd, self.sx(x_bottom), self.sy(bottom_y - 2),
                         color=COLOR_YELLOW, size=7)
        self.track_x = float(x_bottom)
        return err, angle, quality

    def _find_bottom_seed(self, img, cfg, h=62):
        y = CV_H - h
        try:
            blobs = img.find_blobs(
                [tuple(cfg["lab"])], invert=cfg["invert"], roi=(0, y, CV_W, h),
                x_stride=2, y_stride=1,
                area_threshold=cfg["area_threshold"],
                pixels_threshold=cfg["pixels_threshold"], merge=True, margin=2
            )
        except Exception:
            blobs = []
        predicted = self.track_x
        # 第一次允许全宽；已有轨迹后优先延续旧轨迹，但容忍较大偏移。
        continuity = max(float(cfg["continuity_px"]) * 2.0, 80.0)
        b = self._choose_blob(blobs, predicted, continuity, cfg["min_density"])
        if b is None and predicted is not None:
            b = self._choose_blob(blobs, None, continuity, cfg["min_density"])
        return b

    def _algo_roi(self, img, osd, cfg):
        n = int(clamp(cfg["roi_count"], 4, 9))
        top_y = int(CV_H * 0.24)
        usable = CV_H - top_y
        band_h = max(16, usable // n)
        threshold = [tuple(cfg["lab"])]
        centers = []
        predicted = self.track_x
        continuity = float(cfg["continuity_px"])
        corridor = int(cfg["corridor_half_width"])

        # 从底部向上，下一窗口只在前一窗口附近搜索。
        for i in range(n):
            y2 = CV_H - i * band_h
            y1 = max(top_y, y2 - band_h)
            h = max(6, y2 - y1)
            if predicted is None or i == 0:
                rx, rw = 0, CV_W
            else:
                rx = int(clamp(predicted - corridor, 0, CV_W - 8))
                rw = int(min(CV_W - rx, corridor * 2))
                if rw < 8:
                    continue
            try:
                blobs = img.find_blobs(
                    threshold, invert=cfg["invert"], roi=(rx, y1, rw, h),
                    x_stride=2, y_stride=1,
                    area_threshold=max(20, int(cfg["area_threshold"] * 0.55)),
                    pixels_threshold=max(15, int(cfg["pixels_threshold"] * 0.55)),
                    merge=True, margin=2
                )
            except Exception:
                blobs = []

            gate = continuity if predicted is not None else CV_W
            best = self._choose_blob(blobs, predicted, gate, cfg["min_density"])
            if best is None:
                continue
            cx, cy = float(best.cx()), float(best.cy())
            centers.append((cx, cy))

            # 用最近两点外推下一层，弯道不会只按固定 x 走廊。
            if len(centers) >= 2:
                p0, p1 = centers[-2], centers[-1]
                dy = p0[1] - p1[1]
                if abs(dy) > 1:
                    predicted = p1[0] + (p1[0] - p0[0]) * (band_h / abs(dy))
                else:
                    predicted = p1[0]
            else:
                predicted = cx
            predicted = clamp(predicted, 0, CV_W - 1)

            osd.draw_rectangle(self.sx(best.x()), self.sy(best.y()),
                               max(2, self.sx(best.w())), max(2, self.sy(best.h())),
                               color=COLOR_BLUE, thickness=2)

        if len(centers) < max(3, n // 2):
            return None
        return self._result_from_points(centers, n, osd, COLOR_CYAN)

    def _algo_regression(self, img, osd, cfg):
        seed = self._find_bottom_seed(img, cfg, h=68)
        if seed is None:
            return None
        seed_x = float(seed.cx())
        half = int(max(45, cfg["corridor_half_width"]))
        rx = int(clamp(seed_x - half, 0, CV_W - 12))
        rw = int(min(CV_W - rx, half * 2))
        ry = int(CV_H * 0.18)
        roi = (rx, ry, rw, CV_H - ry)
        try:
            ln = img.get_regression(
                [tuple(cfg["lab"])], invert=cfg["invert"], roi=roi,
                x_stride=2, y_stride=1,
                area_threshold=cfg["area_threshold"],
                pixels_threshold=cfg["pixels_threshold"],
                robust=cfg["robust"]
            )
        except Exception as e:
            print("[LINE] regression:", e)
            return None
        if ln is None:
            return None
        try:
            x1, y1, x2, y2 = ln.line()
            osd.draw_rectangle(self.sx(rx), self.sy(ry), self.sx(rw), self.sy(CV_H - ry),
                               color=COLOR_BLUE, thickness=1)
            osd.draw_line(self.sx(x1), self.sy(y1), self.sx(x2), self.sy(y2),
                          color=COLOR_GREEN, thickness=4)
            if y1 >= y2:
                bx, by, tx, ty = float(x1), float(y1), float(x2), float(y2)
            else:
                bx, by, tx, ty = float(x2), float(y2), float(x1), float(y1)
            dy = by - ty
            if abs(dy) < 4:
                return None
            x_bottom_reg = bx + (CV_H - 1 - by) * (bx - tx) / dy
            # 底部种子比远端回归更可靠，二者融合能明显降低回归线受噪声拖拽。
            x_bottom = clamp(0.68 * seed_x + 0.32 * x_bottom_reg, 0, CV_W - 1)
            look_y = int(CV_H * 0.48)
            x_look = bx + (look_y - by) * (bx - tx) / dy
            err = x_bottom - CV_W * 0.5
            angle = math.atan2(x_look - x_bottom, max(1.0, CV_H - 1 - look_y)) * 180.0 / math.pi
            try:
                mag = float(ln.magnitude())
                q_mag = clamp(mag / 3500.0, 0.0, 1.0)
            except Exception:
                q_mag = 0.65
            try:
                q_seed = clamp(float(seed.density()) * 1.6, 0.0, 1.0)
            except Exception:
                q_seed = 0.6
            quality = 0.55 * q_mag + 0.45 * q_seed
            self.track_x = float(x_bottom)
            draw_cross_lines(osd, self.sx(x_bottom), self.sy(CV_H - 3),
                             color=COLOR_YELLOW, size=7)
            return err, angle, quality
        except Exception:
            return None

    def _pixel_white_arr(self, arr, x, y):
        try:
            return int(arr[y][x]) > 127
        except Exception:
            return False

    def _pixel_white_img(self, gray, x, y):
        try:
            return int(gray.get_pixel(x, y)) > 127
        except Exception:
            return False

    def _row_run_near(self, is_white, predicted_x, y, half_width, max_band_width):
        left_bound = int(clamp(predicted_x - half_width, 0, CV_W - 1))
        right_bound = int(clamp(predicted_x + half_width, 0, CV_W - 1))
        runs = []
        in_run = False
        start = left_bound
        x = left_bound
        while x <= right_bound:
            white = is_white(x, y)
            if white and not in_run:
                in_run = True
                start = x
            if in_run and ((not white) or x == right_bound):
                end = x if white and x == right_bound else x - 1
                width = end - start + 1
                if 2 <= width <= max_band_width:
                    center = 0.5 * (start + end)
                    runs.append((abs(center - predicted_x), center, width))
                in_run = False
            x += 1
        if not runs:
            return None
        runs.sort(key=lambda r: r[0])
        return runs[0][1], runs[0][2]

    def _algo_eight(self, img, osd, cfg):
        seed = self._find_bottom_seed(img, cfg, h=62)
        if seed is None:
            return None
        seed_x = float(seed.cx())
        seed_y = int(min(CV_H - 2, seed.y() + seed.h() - 2))
        mask = img
        gray = None
        arr = None
        try:
            mask.binary([tuple(cfg["lab"])], invert=cfg["invert"])
            # 只闭运算补小断点，不再 open()，避免细线被腐蚀掉。
            try:
                mask.close(1)
            except Exception:
                pass
            try:
                gray = mask.to_grayscale(copy=False)
            except Exception:
                gray = mask.to_grayscale()
            try:
                arr = gray.to_numpy_ref()
                is_white = lambda x, y: self._pixel_white_arr(arr, x, y)
            except Exception:
                arr = None
                is_white = lambda x, y: self._pixel_white_img(gray, x, y)

            half = int(clamp(cfg["eight_half_width"], 12, 100))
            max_gap = int(clamp(cfg["eight_max_gap"], 0, 8))
            max_band = int(clamp(cfg.get("eight_max_band_width", 110), 12, 180))
            points = []
            x = seed_x
            y = seed_y
            dx_pred = 0.0
            gap_count = 0
            step_y = 2

            first = self._row_run_near(is_white, x, y, half, max_band)
            if first is not None:
                x = first[0]
            points.append((float(x), float(y)))

            while y > int(CV_H * 0.14):
                ny = y - step_y
                predicted = clamp(x + dx_pred * step_y, 0, CV_W - 1)
                found = self._row_run_near(is_white, predicted, ny, half, max_band)
                if found is None:
                    recovered = None
                    for g in range(1, max_gap + 1):
                        ty = ny - g * step_y
                        if ty < 0:
                            break
                        recovered = self._row_run_near(
                            is_white, predicted, ty,
                            min(half + g * 5, 110), max_band
                        )
                        if recovered is not None:
                            ny = ty
                            break
                    if recovered is None:
                        gap_count += 1
                        if gap_count > max_gap + 1:
                            break
                        y = ny
                        continue
                    found = recovered
                else:
                    gap_count = 0

                nx = float(found[0])
                raw_dx = (nx - x) / max(1.0, float(y - ny))
                dx_pred = 0.65 * dx_pred + 0.35 * raw_dx
                x, y = nx, ny
                points.append((x, float(y)))

            if len(points) < int(cfg["eight_min_points"]):
                return None
            expected = max(1, int((seed_y - int(CV_H * 0.14)) / step_y))
            result = self._result_from_points(points, expected, osd, COLOR_GREEN)
            if result is not None:
                err, angle, quality = result
                # 八邻域质量再乘连通覆盖率。
                quality = clamp(quality * (1.0 - min(0.5, gap_count * 0.05)), 0.0, 1.0)
                return err, angle, quality
            return None
        except Exception as e:
            print("[LINE] eight-neighbor error:", e)
            return None
        finally:
            arr = None
            gray = None
            mask = None

    def process(self, img, osd):
        cfg = self.app.cfg["line"]
        osd.clear()

        train_result = self.trainer.feed(img)
        self.trainer.draw_roi(osd, COLOR_ORANGE)
        if train_result is not None:
            ok, token, lab, extra = train_result
            if ok:
                cfg["lab"] = copy_lab(lab)
                cfg["trained"] = True
                self.reset_track()
                self.app.notice("巡线颜色训练完成", COLOR_GREEN, 1800)
                self.app.ui.invalidate()
                self.app.store.save(self.app.cfg)
            else:
                self.app.notice("巡线取色失败: " + str(extra), COLOR_RED, 2600)

        if not cfg["trained"]:
            self.app.results.clear("line")
            if self.sample_armed:
                draw_short_status(osd, "巡线取色：请点线条中心，不要点背景", COLOR_YELLOW)
            elif self.trainer.active:
                draw_short_status(osd, "巡线LAB采样 %d/%d" % (self.trainer.count, self.trainer.frame_count), COLOR_YELLOW)
            else:
                draw_short_status(osd, "请先训练巡线颜色", COLOR_YELLOW)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            return

        result = None
        mode = cfg["mode"]
        if mode == "roi":
            result = self._algo_roi(img, osd, cfg)
            mode_name = "多区域"
        elif mode == "regression":
            result = self._algo_regression(img, osd, cfg)
            mode_name = "回归"
        else:
            result = self._algo_eight(img, osd, cfg)
            mode_name = "八邻域"

        if result is not None:
            err, angle, quality = result
            err, angle = self._smooth(err, angle)
            self.last_quality = quality
            self.lost_frames = 0
            self._draw_result_text(osd, err, angle, quality, mode_name)
            dx_display = float(err) * DISPLAY_W / float(CV_W)
            cx_display = DISPLAY_W * 0.5 + dx_display
            self.app.results.publish("line", [{
                "id": 0, "class_id": 0, "label": "赛道",
                "score": float(quality),
                "cx": int(round(cx_display)), "cy": DISPLAY_H - 6,
                "w": 0, "h": 0, "area": 0, "angle": float(angle),
                "line_error": float(dx_display),
            }], {"mode": mode_name, "error": float(dx_display), "angle": float(angle)})
        else:
            self.app.results.clear("line")
            self.lost_frames += 1
            # 短暂丢线不立刻把平滑状态清零，避免赛道小断点造成控制量跳变。
            if self.lost_frames > 4:
                self.smooth_error = None
                self.smooth_angle = None
                self.track_x = None
            draw_short_status(osd, "%s 未找到有效赛道" % mode_name, COLOR_YELLOW)

        if self.sample_armed:
            draw_short_status(osd, "巡线取色：请点线条中心", COLOR_YELLOW)
        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)


# ============================================================
# 8. YOLO Engine - 多模型 profile + 触屏切换
# ============================================================

class YoloEngine:
    """
    多模型 YOLO 引擎。

    模型静态信息只来自源码最前面的 YOLO_MODEL_PROFILES。
    切换流程：保存当前阈值 -> deinit旧实例 -> 套用新profile -> config_preprocess -> run。

    detect 结果：优先按官方/已验证的 [boxes, class_ids, scores] 三组结构解析，
    同时保留少量结构兼容。运行时绝不调用官方 draw_result()，因为官方绘制函数会
    自己绘制框和标签，无法服从本工作台的 show_box/show_label/show_conf 开关。
    """
    def __init__(self, app):
        self.app = app
        self.yolo = None
        self.ready = False
        self.needs_rebuild = False
        self.last_error = ""
        self.loaded_family = ""
        self.loaded_model_index = -1
        self.loaded_model_name = ""
        self.last_failed_index = -1
        self.manual_unloaded = False
        self.debug_once = False

    def unload(self, manual=False):
        if self.yolo is not None:
            try:
                self.yolo.deinit()
            except Exception:
                pass
        self.yolo = None
        self.ready = False
        self.needs_rebuild = False
        self.loaded_family = ""
        self.loaded_model_index = -1
        self.loaded_model_name = ""
        self.manual_unloaded = bool(manual)
        if manual:
            self.last_failed_index = -1
        gc.collect()

    def _new_model(self, cfg):
        family = normalize_yolo_family(cfg.get("family", "yolov8"))

        if family == "yolov5":
            if YOLOv5 is None:
                raise RuntimeError("libs.YOLO.YOLOv5 不可用；请检查当前 CanMV 固件是否包含 YOLOv5 模块")
            return YOLOv5(
                task_type="detect", mode="video",
                kmodel_path=cfg["model_path"], labels=cfg["labels"],
                rgb888p_size=[AI_W, AI_H],
                model_input_size=cfg["model_input_size"],
                display_size=[DISPLAY_W, DISPLAY_H],
                conf_thresh=cfg["conf"],
                nms_thresh=cfg.get("nms", 0.45),
                max_boxes_num=cfg["max_boxes"], debug_mode=0
            )

        if family == "yolov8":
            if YOLOv8 is None:
                raise RuntimeError("libs.YOLO.YOLOv8 不可用")
            return YOLOv8(
                task_type="detect", mode="video",
                kmodel_path=cfg["model_path"], labels=cfg["labels"],
                rgb888p_size=[AI_W, AI_H],
                model_input_size=cfg["model_input_size"],
                display_size=[DISPLAY_W, DISPLAY_H],
                conf_thresh=cfg["conf"],
                nms_thresh=cfg.get("nms", 0.45),
                max_boxes_num=cfg["max_boxes"], debug_mode=0
            )

        if family == "yolo11":
            if YOLO11 is None:
                raise RuntimeError("libs.YOLO.YOLO11 不可用")
            return YOLO11(
                task_type="detect", mode="video",
                kmodel_path=cfg["model_path"], labels=cfg["labels"],
                rgb888p_size=[AI_W, AI_H],
                model_input_size=cfg["model_input_size"],
                display_size=[DISPLAY_W, DISPLAY_H],
                conf_thresh=cfg["conf"],
                nms_thresh=cfg.get("nms", 0.45),
                max_boxes_num=cfg["max_boxes"], debug_mode=0
            )

        if family == "yolo26":
            if YOLO26 is None:
                raise RuntimeError("libs.YOLO.YOLO26 不可用")
            # 官方 YOLO26 detect 示例不传 nms_thresh。
            return YOLO26(
                task_type="detect", mode="video",
                kmodel_path=cfg["model_path"], labels=cfg["labels"],
                rgb888p_size=[AI_W, AI_H],
                model_input_size=cfg["model_input_size"],
                display_size=[DISPLAY_W, DISPLAY_H],
                conf_thresh=cfg["conf"],
                max_boxes_num=cfg["max_boxes"], debug_mode=0
            )

        raise RuntimeError("不支持的YOLO family: %s；请填写 yolov5 / yolov8 / yolo11 / yolo26" % family)

    def rebuild(self):
        # rebuild 前必须释放旧 KPU / ai2d 资源。
        self.unload(manual=False)
        self.last_error = ""
        self.manual_unloaded = False
        cfg = self.app.cfg["yolo"]
        idx = int(cfg.get("model_index", 0))
        name = str(cfg.get("profile_name", "模型%d" % (idx + 1)))

        try:
            with open(cfg["model_path"], "rb"):
                pass
        except Exception:
            self.last_error = "模型不存在: " + cfg["model_path"]
            self.last_failed_index = idx
            print("[YOLO]", self.last_error)
            return False

        try:
            self.yolo = self._new_model(cfg)
            self.yolo.config_preprocess()
            self.ready = True
            self.needs_rebuild = False
            self.loaded_family = normalize_yolo_family(cfg.get("family", "yolov8"))
            self.loaded_model_index = idx
            self.loaded_model_name = name
            self.last_failed_index = -1
            self.debug_once = False
            self.app.notice("已加载: " + name, COLOR_GREEN, 1400)
            print("[YOLO] loaded", self.loaded_family,
                  "index", idx, "name", name,
                  "path", cfg["model_path"],
                  "labels", cfg["labels"],
                  "input", cfg["model_input_size"],
                  "conf", cfg["conf"], "nms", cfg.get("nms", 0.45))
            return True
        except Exception as e:
            self.last_error = str(e)
            self.last_failed_index = idx
            print("[YOLO] load failed:", e)
            try:
                if self.yolo is not None:
                    self.yolo.deinit()
            except Exception:
                pass
            self.yolo = None
            self.ready = False
            gc.collect()
            return False

    def _group_to_list(self, group):
        try:
            return [group[i] for i in range(len(group))]
        except Exception:
            return []

    def _looks_like_box_group(self, group):
        try:
            if len(group) <= 0:
                return False
            sample = group[0]
            return len(sample) >= 4
        except Exception:
            return False

    def _value_group_quality(self, group, want_class, label_count):
        vals = self._group_to_list(group)
        if not vals:
            return -1.0
        ok = 0
        used = 0
        for v in vals[:12]:
            try:
                fv = float(v)
            except Exception:
                continue
            used += 1
            if want_class:
                iv = int(round(fv))
                if abs(fv - iv) < 0.001 and iv >= 0 and iv < max(1, label_count):
                    ok += 1
            else:
                if fv >= 0.0 and fv <= 1.05:
                    ok += 1
        if used <= 0:
            return -1.0
        return ok / float(used)

    def _parse_grouped_result(self, res):
        """
        兼容 [boxes, class_ids, scores] 与 [boxes, scores, class_ids]。
        当前 ball YOLOv8 已实机验证前一种；其它 family 先用数值语义判断。
        """
        try:
            n = len(res)
        except Exception:
            return []
        if n < 3:
            return []

        box_idx = -1
        for i in range(min(n, 4)):
            try:
                if self._looks_like_box_group(res[i]):
                    box_idx = i
                    break
            except Exception:
                pass
        if box_idx < 0:
            return []

        others = []
        for i in range(min(n, 4)):
            if i != box_idx:
                try:
                    if len(res[i]) >= 0:
                        others.append(i)
                except Exception:
                    pass
        if len(others) < 2:
            return []

        cfg = self.app.cfg["yolo"]
        labels = cfg.get("labels", ["object"])
        label_count = max(1, len(labels))

        # 找最像类别ID的组。若并列，保持当前实机验证顺序：res[1] 优先类别。
        best_cls_idx = others[0]
        best_cls_q = -1.0
        for i in others:
            q = self._value_group_quality(res[i], True, label_count)
            if q > best_cls_q:
                best_cls_q = q
                best_cls_idx = i

        score_candidates = [i for i in others if i != best_cls_idx]
        if not score_candidates:
            return []
        best_score_idx = score_candidates[0]
        best_score_q = -1.0
        for i in score_candidates:
            q = self._value_group_quality(res[i], False, label_count)
            if q > best_score_q:
                best_score_q = q
                best_score_idx = i

        boxes = res[box_idx]
        class_ids = res[best_cls_idx]
        scores = res[best_score_idx]
        try:
            count = min(len(boxes), len(class_ids), len(scores))
        except Exception:
            return []

        out = []
        max_boxes = int(cfg.get("max_boxes", 20))
        for i in range(min(count, max_boxes)):
            try:
                box = boxes[i]
                if len(box) < 4:
                    continue
                x = float(box[0])
                y = float(box[1])
                w = float(box[2])
                h = float(box[3])
                cls = int(round(float(class_ids[i])))
                score = float(scores[i])
            except Exception:
                continue
            if cls < 0 or cls >= label_count:
                continue
            if score < float(cfg.get("conf", 0.20)) or score > 1.05:
                continue
            if w <= 2.0 or h <= 2.0:
                continue
            if x + w < 0 or y + h < 0 or x >= DISPLAY_W or y >= DISPLAY_H:
                continue
            out.append((x, y, w, h, score, cls))
        return out

    def _parse_row_result(self, res):
        """兜底兼容逐目标六元组：[cls,score,x1,y1,x2,y2]。"""
        cfg = self.app.cfg["yolo"]
        labels = cfg.get("labels", ["object"])
        label_count = max(1, len(labels))
        out = []
        try:
            count = min(len(res), int(cfg.get("max_boxes", 20)))
        except Exception:
            return out
        for i in range(count):
            try:
                r = res[i]
                if len(r) < 6:
                    continue
                cls = int(round(float(r[0])))
                score = float(r[1])
                x1, y1, x2, y2 = float(r[2]), float(r[3]), float(r[4]), float(r[5])
                w, h = x2 - x1, y2 - y1
            except Exception:
                continue
            if cls < 0 or cls >= label_count:
                continue
            if score < float(cfg.get("conf", 0.20)) or score > 1.05:
                continue
            if w <= 2.0 or h <= 2.0:
                continue
            out.append((x1, y1, w, h, score, cls))
        return out

    def _iter_detections(self, res):
        if res is None:
            return []
        out = self._parse_grouped_result(res)
        if out:
            return out
        return self._parse_row_result(res)

    def _result_has_objects(self, res):
        if res is None:
            return False
        try:
            if len(res) <= 0:
                return False
            # 分组结构：任意前3组有成员。
            for i in range(min(3, len(res))):
                try:
                    if len(res[i]) > 0:
                        return True
                except Exception:
                    pass
            return False
        except Exception:
            return False

    def process(self, img, osd):
        osd.clear()
        cfg = self.app.cfg["yolo"]
        selected_idx = int(cfg.get("model_index", 0))

        if self.ready and self.loaded_model_index != selected_idx:
            self.rebuild()

        if not self.ready:
            self.app.results.clear("yolo")
            if self.manual_unloaded:
                draw_short_status(osd, "YOLO已卸载，点击“加载当前模型”", COLOR_YELLOW)
                self.app.draw_runtime_decor(osd)
                Display.show_image(osd, layer=Display.LAYER_OSD0)
                return
            if self.last_failed_index == selected_idx:
                draw_short_status(osd, "加载失败: " + self.last_error, COLOR_RED)
                self.app.draw_runtime_decor(osd)
                Display.show_image(osd, layer=Display.LAYER_OSD0)
                return
            if not self.rebuild():
                draw_short_status(osd, "YOLO未启动: " + self.last_error, COLOR_RED)
                self.app.draw_runtime_decor(osd)
                Display.show_image(osd, layer=Display.LAYER_OSD0)
                return

        try:
            res = self.yolo.run(img.to_numpy_ref())
            detections = self._iter_detections(res)
            cfg = self.app.cfg["yolo"]
            labels = cfg.get("labels", ["object"])

            # YOLO 保持官方整帧预处理，识别区域只做结果过滤，避免动态裁剪引入
            # AI2D/坐标映射差异。中心点进入区域才视为有效目标。
            if self.app.region_manager is not None and self.app.region_manager.enabled("yolo"):
                filtered = []
                for d in detections:
                    x, y, w, h, score, cls = d
                    cx = x + w * 0.5
                    cy = y + h * 0.5
                    if self.app.region_manager.contains_display("yolo", cx, cy):
                        filtered.append(d)
                detections = filtered

            targets = []
            for i, item in enumerate(detections):
                x, y, w, h, score, cls = item
                dx = int(round(x))
                dy = int(round(y))
                dw = int(round(w))
                dh = int(round(h))
                cx = dx + dw // 2
                cy = dy + dh // 2
                color = PALETTE[cls % len(PALETTE)]

                if cfg.get("show_box", True):
                    osd.draw_rectangle(dx, dy, max(1, dw), max(1, dh), color=color, thickness=3)
                if cfg.get("show_center", False):
                    draw_cross_lines(osd, cx, cy, color=color, size=7)

                result_text = ""
                label = labels[cls] if 0 <= cls < len(labels) else "object"
                if cfg.get("show_label", True):
                    result_text = label
                if cfg.get("show_conf", True):
                    result_text += (" " if result_text else "") + "%.2f" % score
                if result_text:
                    osd.draw_string_advanced(dx, safe_label_y(dy, dh), 18, result_text, color=color)

                targets.append({
                    "id": i,
                    "class_id": int(cls),
                    "label": str(label),
                    "score": float(score),
                    "cx": cx,
                    "cy": cy,
                    "w": max(1, dw),
                    "h": max(1, dh),
                    "area": float(max(1, dw) * max(1, dh)),
                    "angle": 0.0,
                })

            self.app.results.publish("yolo", targets, {
                "model": cfg.get("profile_name", ""),
                "family": normalize_yolo_family(cfg.get("family", "yolov8")),
            })

            if res is not None and len(detections) == 0 and not self.debug_once:
                self.debug_once = True
                try:
                    print("[YOLO RES] no visible detection; family=", self.loaded_family,
                          "groups=", len(res),
                          "g0=", len(res[0]) if len(res) > 0 else -1,
                          "g1=", len(res[1]) if len(res) > 1 else -1,
                          "g2=", len(res[2]) if len(res) > 2 else -1,
                          "conf_ui=", cfg.get("conf", 0.0))
                except Exception as e:
                    print("[YOLO RES DEBUG]", e)

        except Exception as e:
            self.app.results.clear("yolo")
            self.last_error = str(e)
            print("[YOLO RUN]", e)
            draw_short_status(osd, "YOLO运行异常: " + self.last_error, COLOR_RED)

        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)

# ============================================================
# 9. Code Engine - QR / DataMatrix / Barcode / AprilTag
# ============================================================

class CodeEngine:
    """QR / DataMatrix / 条形码 / AprilTag。"""

    def __init__(self, app):
        self.app = app
        self.last_payload = ""
        self.last_source = ""
        self.barcode_scan_phase = 0
        self.barcode_last_scan_roi = None
        self.barcode_last_strategy = "原图全屏"
        self.barcode_otsu_threshold = -1

    def sx(self, x, src_w=CV_W):
        return int(int(x) * DISPLAY_W / max(1, int(src_w)))

    def sy(self, y, src_h=CV_H):
        return int(int(y) * DISPLAY_H / max(1, int(src_h)))

    def _safe_payload(self, obj):
        try:
            value = obj.payload()
            if isinstance(value, bytes):
                try:
                    return value.decode("utf-8")
                except Exception:
                    return str(value)
            return str(value)
        except Exception:
            return ""

    def _roi_for_size(self, key, dst_w, dst_h):
        if self.app.region_manager is None:
            return None
        try:
            r = self.app.region_manager.display_rect(key, only_if_enabled=True)
        except Exception:
            r = None
        if r is None:
            return None
        x, y, w, h = r
        dw = max(2, int(dst_w))
        dh = max(2, int(dst_h))
        rx = int(clamp(int(x * dw / DISPLAY_W), 0, dw - 2))
        ry = int(clamp(int(y * dh / DISPLAY_H), 0, dh - 2))
        rw = int(clamp(int(w * dw / DISPLAY_W), 2, dw - rx))
        rh = int(clamp(int(h * dh / DISPLAY_H), 2, dh - ry))
        return (rx, ry, rw, rh)

    def _apriltag_entries(self):
        return [
            ("16h5", "TAG16H5", getattr(image, "TAG16H5", 0)),
            ("25h7", "TAG25H7", getattr(image, "TAG25H7", 0)),
            ("25h9", "TAG25H9", getattr(image, "TAG25H9", 0)),
            ("36h10", "TAG36H10", getattr(image, "TAG36H10", 0)),
            ("36h11", "TAG36H11", getattr(image, "TAG36H11", 0)),
            ("artoolkit", "ARTOOLKIT", getattr(image, "ARTOOLKIT", 0)),
        ]

    def _apriltag_mask(self, key):
        key = str(key).lower()
        entries = self._apriltag_entries()
        if key == "all":
            mask = 0
            for _, _, value in entries:
                try:
                    mask |= int(value)
                except Exception:
                    pass
            if mask:
                return mask
        for k, _, value in entries:
            if k == key and value:
                return value
        return getattr(image, "TAG36H11", 0)

    def _apriltag_name(self, obj):
        try:
            fam = obj.family()
        except Exception:
            return "TAG"
        for _, name, value in self._apriltag_entries():
            try:
                if value and fam == value:
                    return name
            except Exception:
                pass
        return "TAG"

    def _barcode_name(self, obj):
        try:
            t = obj.type()
        except Exception:
            return "条码"
        mapping = (
            ("EAN2", "EAN2"), ("EAN5", "EAN5"), ("EAN8", "EAN8"),
            ("UPCE", "UPCE"), ("ISBN10", "ISBN10"), ("UPCA", "UPCA"),
            ("EAN13", "EAN13"), ("ISBN13", "ISBN13"), ("I25", "I25"),
            ("DATABAR", "DATABAR"), ("DATABAR_EXP", "DATABAR_EXP"),
            ("CODABAR", "CODABAR"), ("CODE39", "CODE39"),
            ("PDF417", "PDF417"), ("CODE93", "CODE93"),
            ("CODE128", "CODE128"),
        )
        for const_name, show_name in mapping:
            value = getattr(image, const_name, None)
            if value is not None:
                try:
                    if t == value:
                        return show_name
                except Exception:
                    pass
        return "条码"

    def _gray_barcode(self, img, cfg):
        """
        一维条码固定使用官方原生链路：
            CH2 = 640x480
            Sensor.GRAYSCALE
            find_barcodes()

        不再提供 RGB565 / 软件转灰度作为正常工作模式。
        只有原生链路尚未完成切换时，才临时返回回退图并显示等待状态。
        """
        if self.app.barcode_native_active:
            native = self.app.sensor.snapshot(chn=CAM_CHN_ID_2)
            return native, "640x480原生灰度"

        # 正常情况下这里只会在刚进入条码页、相机链路尚未重建完成时短暂发生。
        return grayscale_heap_copy(img), "等待原生灰度通道"

    def _barcode_auto_rois(self, w, h):
        """
        官方建议：一维码最好使用 640x40/80/160 的长条窗口；
        也支持旋转90°后的 40/80/160 x 480 窗口。
        这里每帧只扫一个条带，避免一次调用十几次 find_barcodes 阻塞。
        """
        rois = []

        # 横向条码：640 x 160，50%重叠，优先画面中心。
        band_h = min(160, h)
        ys = []
        if h > band_h:
            center_y = max(0, (h - band_h) // 2)
            ys.append(center_y)
            step = max(40, band_h // 2)
            y = 0
            while y <= h - band_h:
                if y not in ys:
                    ys.append(y)
                y += step
            last = h - band_h
            if last not in ys:
                ys.append(last)
        else:
            ys = [0]
        for y in ys:
            rois.append((0, int(y), int(w), int(band_h)))

        # 纵向条码：160 x 480，50%重叠，同样优先中心。
        band_w = min(160, w)
        xs = []
        if w > band_w:
            center_x = max(0, (w - band_w) // 2)
            xs.append(center_x)
            step = max(40, band_w // 2)
            x = 0
            while x <= w - band_w:
                if x not in xs:
                    xs.append(x)
                x += step
            last = w - band_w
            if last not in xs:
                xs.append(last)
        else:
            xs = [0]
        for x in xs:
            rois.append((int(x), 0, int(band_w), int(h)))

        return rois

    def _barcode_copy(self, work):
        """创建条码预处理副本，优先放普通 HEAP。"""
        try:
            return work.copy(alloc=image.ALLOC_HEAP)
        except Exception:
            try:
                return work.copy()
            except Exception:
                return None

    def _barcode_otsu(self, work, roi=None, invert=False):
        """
        官方 Otsu + binary 二值化增强。
        返回 (results, threshold)。
        """
        proc = None
        try:
            hist = work.get_histogram(roi=roi) if roi is not None else work.get_histogram()
            threshold = int(hist.get_threshold().value())

            proc = self._barcode_copy(work)
            if proc is None:
                return [], threshold

            # 高亮区域变白、暗条纹保持黑；invert=True 时再尝试相反极性。
            proc.binary([(threshold, 255)], invert=bool(invert))

            if roi is not None:
                results = proc.find_barcodes(roi=roi)
            else:
                results = proc.find_barcodes()

            return results, threshold

        except Exception as e:
            print("[条形码增强] Otsu失败:", e)
            return [], -1
        finally:
            proc = None
            gc.collect()

    def _barcode_fixed_bands(self, w, h):
        """
        固定推荐窗口，不移动、不在屏幕画黄色框。
        官方推荐长边640、短边40/80/160。
        这里采用160像素高的三个固定横向区域。
        """
        band_h = min(160, int(h))
        max_y = max(0, int(h) - band_h)

        center_y = max(0, max_y // 2)
        top_y = 0
        bottom_y = max_y

        rois = [
            ("中心160", (0, center_y, int(w), band_h)),
            ("上部160", (0, top_y, int(w), band_h)),
            ("下部160", (0, bottom_y, int(w), band_h)),
        ]

        # 去掉完全重复的ROI（例如图像高度本身就<=160）
        unique = []
        seen = set()
        for name, r in rois:
            if r not in seen:
                seen.add(r)
                unique.append((name, r))
        return unique

    def _find_barcodes_robust(self, work, user_roi=None):
        """
        一维条码兼容增强：

        第一优先级（每帧都执行）：
            官方成功链路
            640x480 原生 GRAYSCALE + find_barcodes()

        如果原图失败，再每帧轮流执行一个增强策略：
            - 固定中心/上部/下部 640x160 推荐窗口
            - Otsu 自动二值化
            - Otsu 反相二值化

        用户手工 ROI 开启后：
            - 原图固定ROI
            - Otsu固定ROI
            - Otsu反相固定ROI

        不使用移动ROI，不绘制自动扫描框。
        """
        self.barcode_otsu_threshold = -1

        # 1. 官方原始链路永远优先。
        try:
            if user_roi is not None:
                raw_results = work.find_barcodes(roi=user_roi)
                self.barcode_last_strategy = "原图固定区域"
            else:
                raw_results = work.find_barcodes()
                self.barcode_last_strategy = "原图全屏"

            if raw_results:
                return raw_results
        except Exception as e:
            print("[条形码] 原图扫描异常:", e)

        # 2. 原图失败后才做增强。
        phase = int(self.barcode_scan_phase)
        self.barcode_scan_phase = (phase + 1) % 8

        # 用户已经手工框ROI：不再换区域，只改变图像处理方式。
        if user_roi is not None:
            mode = phase % 2

            if mode == 0:
                results, th = self._barcode_otsu(work, user_roi, invert=False)
                self.barcode_otsu_threshold = th
                self.barcode_last_strategy = "Otsu固定区域"
                if results:
                    return results
            else:
                results, th = self._barcode_otsu(work, user_roi, invert=True)
                self.barcode_otsu_threshold = th
                self.barcode_last_strategy = "反相固定区域"
                if results:
                    return results

            return []

        # 未设置ROI：先尝试官方建议的固定长条窗口。
        bands = self._barcode_fixed_bands(int(work.width()), int(work.height()))

        if phase < len(bands):
            name, roi = bands[phase]
            try:
                self.barcode_last_strategy = name
                results = work.find_barcodes(roi=roi)
                if results:
                    return results
            except Exception as e:
                print("[条形码增强] 固定条带失败:", e)
            return []

        # 之后轮流尝试整帧/中心条带 Otsu。
        extra = phase - len(bands)

        if extra == 0:
            results, th = self._barcode_otsu(work, None, invert=False)
            self.barcode_otsu_threshold = th
            self.barcode_last_strategy = "Otsu全屏"
            return results

        if extra == 1:
            results, th = self._barcode_otsu(work, None, invert=True)
            self.barcode_otsu_threshold = th
            self.barcode_last_strategy = "反相全屏"
            return results

        if bands:
            _, center_roi = bands[0]

            if extra == 2:
                results, th = self._barcode_otsu(work, center_roi, invert=False)
                self.barcode_otsu_threshold = th
                self.barcode_last_strategy = "Otsu中心160"
                return results

            if extra == 3:
                results, th = self._barcode_otsu(work, center_roi, invert=True)
                self.barcode_otsu_threshold = th
                self.barcode_last_strategy = "反相中心160"
                return results

        self.barcode_last_strategy = "原图全屏"
        return []


    def process(self, img, osd):
        cfg = self.app.cfg["code"]
        mode = cfg["mode"]
        osd.clear()
        work = None

        try:
            if mode == "barcode":
                work, source = self._gray_barcode(img, cfg)
            else:
                work = grayscale_heap_copy(img)
                source = "400x240灰度"

            sw, sh = int(work.width()), int(work.height())
            roi = self._roi_for_size("code", sw, sh)

            if mode == "qr":
                results = work.find_qrcodes(roi=roi) if roi is not None else work.find_qrcodes()
            elif mode == "dm":
                if roi is not None:
                    results = work.find_datamatrices(roi=roi, effort=cfg["dm_effort"])
                else:
                    results = work.find_datamatrices(effort=cfg["dm_effort"])
            elif mode == "barcode":
                results = self._find_barcodes_robust(work, roi)
            else:
                families = self._apriltag_mask(cfg.get("apriltag_family", "all"))
                if roi is not None:
                    results = work.find_apriltags(roi=roi, families=families)
                else:
                    results = work.find_apriltags(families=families)

            self.last_source = source

        except Exception as e:
            self.app.results.clear("code")
            draw_short_status(osd, "码类识别异常: " + str(e), COLOR_RED)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            work = None
            gc.collect()
            return

        targets = []
        shown = 0
        for obj in results:
            if shown >= cfg["max_results"]:
                break
            try:
                x, y, w, h = obj.rect()
                dx = self.sx(x, sw)
                dy = self.sy(y, sh)
                dw = max(1, self.sx(w, sw))
                dh = max(1, self.sy(h, sh))
                cx = dx + dw // 2
                cy = dy + dh // 2
                color = PALETTE[shown % len(PALETTE)]
                osd.draw_rectangle(dx, dy, dw, dh, color=color, thickness=3)

                class_id = 0
                payload = ""
                family_name = ""
                angle = 0.0

                if mode == "apriltag":
                    try:
                        class_id = int(obj.id())
                    except Exception:
                        class_id = 0
                    family_name = self._apriltag_name(obj)
                    text = "%s ID:%d" % (family_name, class_id)
                    payload = "%s:%d" % (family_name, class_id)
                    try:
                        cx = self.sx(obj.cx(), sw)
                        cy = self.sy(obj.cy(), sh)
                        draw_cross_lines(osd, cx, cy, color=COLOR_GREEN, size=7)
                    except Exception:
                        pass
                    try:
                        angle = float(obj.rotation()) * 180.0 / math.pi
                    except Exception:
                        angle = 0.0

                elif mode == "barcode":
                    payload = self._safe_payload(obj)
                    name = self._barcode_name(obj)
                    text = name if not payload else (name + ": " + payload)
                    self.last_payload = payload
                    try:
                        angle = float(obj.rotation()) * 180.0 / math.pi
                    except Exception:
                        angle = 0.0

                else:
                    text = self._safe_payload(obj)
                    payload = text
                    self.last_payload = text

                show_text = str(text)
                if len(show_text) > 28:
                    show_text = show_text[:28] + "..."
                if show_text:
                    osd.draw_string_advanced(
                        dx, safe_label_y(dy, dh), 18,
                        show_text, color=color
                    )

                targets.append({
                    "id": shown,
                    "class_id": class_id,
                    "label": str(payload),
                    "score": 1.0,
                    "cx": cx,
                    "cy": cy,
                    "w": dw,
                    "h": dh,
                    "area": float(dw * dh),
                    "angle": angle,
                    "family": family_name,
                })
                shown += 1
            except Exception as e:
                print("[CODE RESULT]", e)

        self.app.results.publish("code", targets, {
            "mode": mode,
            "source": self.last_source,
            "apriltag_family": cfg.get("apriltag_family", "all"),
            "barcode_strategy": self.barcode_last_strategy if mode == "barcode" else "",
            "barcode_otsu": self.barcode_otsu_threshold if mode == "barcode" else -1,
        })

        if shown == 0:
            if mode == "barcode":
                draw_short_status(
                    osd,
                    "等待条码 | %s | %s" %
                    (self.last_source, self.barcode_last_strategy),
                    COLOR_YELLOW
                )
            else:
                names = {
                    "qr": "二维码",
                    "dm": "DataMatrix",
                    "barcode": "条形码",
                    "apriltag": "AprilTag"
                }
                draw_short_status(osd, "等待识别: " + names.get(mode, mode), COLOR_YELLOW)

        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)
        work = None
        gc.collect()


class GeometryEngine:
    """
    直线 / 线段 / 圆 / 矩形统一几何检测。

    设计原则：
    1) 底层仍调用 CanMV 官方 find_lines/find_line_segments/find_circles/find_rects；
    2) 不额外申请大图，继续沿用原地 mean_pool 的低内存工作图；
    3) 在官方原始候选之后增加轻量后处理：
       - 线段：最小长度 + 近重复抑制；
       - 圆：半径范围 + 同心/近同心 NMS + 圆周/四象限真实边缘验证；
       - 矩形：角度约束 + 四边真实像素对比度验证 + 重复抑制。
    这样既保留通用性，也避免把算法的所有原始候选直接画到屏幕上。
    """
    MODE_NAMES = {"line": "直线", "segment": "线段", "circle": "圆形", "rect": "矩形"}

    def __init__(self, app):
        self.app = app
        self.memory_fallback_once = False

    def sx(self, x, work_w):
        return int(round(float(x) * DISPLAY_W / float(max(1, work_w))))

    def sy(self, y, work_h):
        return int(round(float(y) * DISPLAY_H / float(max(1, work_h))))

    def _angle_diff(self, a, b):
        d = abs(float(a) - float(b)) % 180.0
        if d > 90.0:
            d = 180.0 - d
        return d

    def _line_target(self, obj, idx, label, work_w, work_h):
        x1, y1, x2, y2 = obj.line()
        dx1, dy1 = self.sx(x1, work_w), self.sy(y1, work_h)
        dx2, dy2 = self.sx(x2, work_w), self.sy(y2, work_h)
        cx, cy = (dx1 + dx2) // 2, (dy1 + dy2) // 2
        length = math.sqrt((dx2 - dx1) * (dx2 - dx1) + (dy2 - dy1) * (dy2 - dy1))
        try:
            angle = float(obj.theta())
        except Exception:
            angle = math.atan2(dy2 - dy1, dx2 - dx1) * 180.0 / math.pi
        try:
            magnitude = float(obj.magnitude())
        except Exception:
            magnitude = 0.0
        return {
            "id": idx, "class_id": 0 if label == "直线" else 1, "label": label,
            "score": 1.0, "cx": cx, "cy": cy,
            "w": abs(dx2 - dx1), "h": abs(dy2 - dy1), "area": 0.0,
            "angle": angle, "magnitude": magnitude, "length": length,
            "line": [dx1, dy1, dx2, dy2],
        }

    def _detect(self, work, cfg, mode, roi, total_div):
        """只做官方算法调用；后处理与绘制在成功之后进行。"""
        div = max(1, int(total_div))
        if mode == "line":
            kwargs = dict(
                x_stride=int(cfg["line_x_stride"]),
                y_stride=int(cfg["line_y_stride"]),
                threshold=max(80, int(cfg["line_threshold"]) // div),
                theta_margin=int(cfg["theta_margin"]),
                rho_margin=max(1, int(cfg["rho_margin"]) // div),
            )
            return work.find_lines(roi=roi, **kwargs) if roi is not None else work.find_lines(**kwargs)

        if mode == "segment":
            kwargs = dict(
                merge_distance=max(0, int(cfg["segment_merge_distance"]) // div),
                max_theta_difference=int(cfg["segment_theta_difference"]),
            )
            return work.find_line_segments(roi=roi, **kwargs) if roi is not None else work.find_line_segments(**kwargs)

        if mode == "circle":
            kwargs = dict(
                x_stride=int(cfg["circle_x_stride"]),
                y_stride=int(cfg["circle_y_stride"]),
                threshold=max(80, int(cfg["circle_threshold"]) // div),
                x_margin=max(1, int(cfg["circle_x_margin"]) // div),
                y_margin=max(1, int(cfg["circle_y_margin"]) // div),
                r_margin=max(1, int(cfg["circle_r_margin"]) // div),
            )
            return work.find_circles(roi=roi, **kwargs) if roi is not None else work.find_circles(**kwargs)

        kwargs = dict(threshold=max(80, int(cfg["rect_threshold"]) // div))
        return work.find_rects(roi=roi, **kwargs) if roi is not None else work.find_rects(**kwargs)

    def _filter_lines(self, results, cfg, mode, work_w, work_h):
        label = "直线" if mode == "line" else "线段"
        min_len = int(cfg.get("line_min_length", 0) if mode == "line" else cfg.get("segment_min_length", 0))
        raw = []
        for obj in (results or []):
            try:
                t = self._line_target(obj, 0, label, work_w, work_h)
                if cfg.get("smart_filter", True) and t["length"] < min_len:
                    continue
                raw.append(t)
            except Exception:
                pass

        # 直线优先看 Sobel/Hough 强度；线段优先保留更长的结构边。
        if mode == "line":
            raw.sort(key=lambda t: (t.get("magnitude", 0.0), t.get("length", 0.0)), reverse=True)
        else:
            raw.sort(key=lambda t: (t.get("length", 0.0), t.get("magnitude", 0.0)), reverse=True)

        if not cfg.get("smart_filter", True):
            return raw[:int(cfg["max_results"])]

        kept = []
        nms_angle = float(cfg.get("segment_nms_angle", 8))
        nms_dist = float(cfg.get("segment_nms_distance", 24))
        for t in raw:
            duplicate = False
            if mode == "segment":
                for k in kept:
                    if self._angle_diff(t["angle"], k["angle"]) <= nms_angle:
                        md = math.sqrt((t["cx"] - k["cx"]) ** 2 + (t["cy"] - k["cy"]) ** 2)
                        # 只抑制真正“叠在一起”的近重复线段，不会把矩形上下两条平行边合掉。
                        if md <= nms_dist:
                            duplicate = True
                            break
            if not duplicate:
                kept.append(t)
            if len(kept) >= int(cfg["max_results"]):
                break
        return kept

    def _circle_edge_quality(self, work, obj, cfg, roi=None):
        """
        对霍夫圆候选做“圆周真实性”二次验证。

        官方 find_circles() 使用 Sobel + Hough 投票；复杂背景里的电线、键盘边缘、
        桌面纹理可能由若干不连续弧段共同投票出一个大圆。这里不创建额外图像，
        只沿候选圆周均匀取样，并在圆周法线方向比较内外像素差：
        - 真圆：大部分角度都应存在真实边缘；
        - 假圆：通常只有少数弧段命中，而且会集中在一两个象限。

        返回：overall_support, min_quadrant_support，范围 0~1。
        """
        try:
            cx = float(obj.x()); cy = float(obj.y()); r = float(obj.r())
            w = int(work.width()); h = int(work.height())
        except Exception:
            return 0.0, 0.0
        if r < 2.5 or w < 4 or h < 4:
            return 0.0, 0.0

        # 28 个采样点 = 每象限 7 点。相比只看总命中率，四象限最低支持率
        # 能有效排除“某一段弯电线/圆弧刚好很像圆”的情况。
        sample_count = 28
        q_hits = [0, 0, 0, 0]
        q_total = [0, 0, 0, 0]
        hits = 0
        total = 0
        contrast = float(cfg.get("circle_edge_contrast", 24))

        # mean_pool 后小圆半径可能只有 6~10px，因此偏移量随半径自适应，
        # 同时限制在合理范围，避免跨过整个小目标。
        d1 = max(0.8, min(1.8, r * 0.16))
        d2 = max(1.4, min(2.8, r * 0.28))

        if roi is None:
            rx, ry, rw, rh = 0, 0, w, h
        else:
            try:
                rx, ry, rw, rh = [int(v) for v in roi]
            except Exception:
                rx, ry, rw, rh = 0, 0, w, h
        rx2 = rx + rw - 1; ry2 = ry + rh - 1

        for i in range(sample_count):
            ang = (2.0 * math.pi * float(i)) / float(sample_count)
            ca = math.cos(ang); sa = math.sin(ang)
            q = int((i * 4) // sample_count)
            if q > 3: q = 3
            q_total[q] += 1
            total += 1

            # 圆周内外两档采样，取更强的一档，兼容模糊/mean_pool 后 1~2px 的边缘厚度。
            best = 0.0
            valid = False
            for d in (d1, d2):
                xin = cx + (r - d) * ca; yin = cy + (r - d) * sa
                xout = cx + (r + d) * ca; yout = cy + (r + d) * sa
                # 超出当前 ROI/图像的圆周部分视为不支持，而不是偷偷忽略，
                # 这样只露出一小段的“残缺圆”不会获得虚高支持率。
                if (xin < rx or xin > rx2 or yin < ry or yin > ry2 or
                        xout < rx or xout > rx2 or yout < ry or yout > ry2):
                    continue
                valid = True
                diff = self._pixel_difference(work, xin, yin, xout, yout)
                if diff > best:
                    best = diff
            if valid and best >= contrast:
                hits += 1
                q_hits[q] += 1

        overall = float(hits) / float(max(1, total))
        q_support = []
        for i in range(4):
            q_support.append(float(q_hits[i]) / float(max(1, q_total[i])))
        return overall, min(q_support) if q_support else 0.0

    def _filter_circles(self, results, cfg, work, work_w, work_h, roi=None):
        raw = []
        min_r = int(cfg.get("circle_min_radius", 2))
        max_r = int(cfg.get("circle_max_radius", 600))
        verify = bool(cfg.get("circle_verify", True))
        min_support = float(cfg.get("circle_edge_support", 50)) / 100.0
        min_quad = float(cfg.get("circle_quadrant_support", 28)) / 100.0

        for obj in (results or []):
            try:
                cx = self.sx(obj.x(), work_w)
                cy = self.sy(obj.y(), work_h)
                r = max(1, self.sx(obj.r(), work_w))
                if cfg.get("smart_filter", True) and (r < min_r or r > max_r):
                    continue
                try:
                    mag = float(obj.magnitude())
                except Exception:
                    mag = 0.0

                edge_support = 1.0
                quadrant_support = 1.0
                if cfg.get("smart_filter", True) and verify:
                    edge_support, quadrant_support = self._circle_edge_quality(work, obj, cfg, roi)
                    if edge_support < min_support or quadrant_support < min_quad:
                        continue

                raw.append({
                    "obj": obj, "cx": cx, "cy": cy, "r": r, "magnitude": mag,
                    "edge_support": edge_support, "quadrant_support": quadrant_support,
                    # 质量优先于霍夫 magnitude。复杂背景下一个很大的错误圆可能 magnitude 很高，
                    # 但它的圆周真实性通常很低。
                    "quality": edge_support * 0.72 + quadrant_support * 0.28,
                })
            except Exception:
                pass

        raw.sort(key=lambda c: (c.get("quality", 0.0), c["magnitude"], -c["r"]), reverse=True)

        if not cfg.get("smart_filter", True):
            return raw[:int(cfg["max_results"])]

        kept = []
        center_gate = float(cfg.get("circle_merge_center", 36))
        radius_gate = float(cfg.get("circle_merge_radius", 36))
        for c in raw:
            duplicate = False
            for k in kept:
                d = math.sqrt((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2)
                dr = abs(c["r"] - k["r"])
                adaptive_c = max(center_gate, min(c["r"], k["r"]) * 0.22)
                adaptive_r = max(radius_gate, max(c["r"], k["r"]) * 0.28)
                if d <= adaptive_c and dr <= adaptive_r:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(c)
            if len(kept) >= int(cfg["max_results"]):
                break
        return kept

    def _pixel_difference(self, img, x1, y1, x2, y2):
        try:
            w, h = int(img.width()), int(img.height())
            x1 = int(clamp(int(round(x1)), 0, w - 1)); y1 = int(clamp(int(round(y1)), 0, h - 1))
            x2 = int(clamp(int(round(x2)), 0, w - 1)); y2 = int(clamp(int(round(y2)), 0, h - 1))
            p1 = img.get_pixel(x1, y1)
            p2 = img.get_pixel(x2, y2)
            if isinstance(p1, tuple) and isinstance(p2, tuple):
                n = min(len(p1), len(p2), 3)
                if n <= 0:
                    return 0.0
                s = 0.0
                for i in range(n):
                    s += abs(float(p1[i]) - float(p2[i]))
                return s / float(n)
            return abs(float(p1) - float(p2))
        except Exception:
            return 0.0

    def _corner_angle_error(self, corners):
        if corners is None or len(corners) != 4:
            return 999.0
        max_err = 0.0
        for i in range(4):
            a = corners[(i - 1) % 4]
            b = corners[i]
            c = corners[(i + 1) % 4]
            v1x, v1y = float(a[0] - b[0]), float(a[1] - b[1])
            v2x, v2y = float(c[0] - b[0]), float(c[1] - b[1])
            n1 = math.sqrt(v1x * v1x + v1y * v1y)
            n2 = math.sqrt(v2x * v2x + v2y * v2y)
            if n1 < 1e-6 or n2 < 1e-6:
                return 999.0
            cs = (v1x * v2x + v1y * v2y) / (n1 * n2)
            cs = clamp(cs, -1.0, 1.0)
            ang = math.acos(cs) * 180.0 / math.pi
            max_err = max(max_err, abs(90.0 - ang))
        return max_err

    def _rect_edge_support(self, work, corners, contrast_threshold):
        """
        沿候选四边形的四条边取样，在边线法向两侧比较像素。
        真矩形：整条边大部分位置都有明显跨边缘对比；
        圆形被 find_rects 外接成菱形时：通常只有切点附近有对比，其余边段没有真实边缘。
        只调用 get_pixel，不创建任何额外帧缓冲。
        """
        if corners is None or len(corners) != 4:
            return 0.0
        samples = (0.18, 0.34, 0.50, 0.66, 0.82)
        hit = 0
        total = 0
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            vx = float(p1[0] - p0[0]); vy = float(p1[1] - p0[1])
            ln = math.sqrt(vx * vx + vy * vy)
            if ln < 4.0:
                continue
            nx = -vy / ln; ny = vx / ln
            for t in samples:
                x = float(p0[0]) + vx * t
                y = float(p0[1]) + vy * t
                # mean_pool 后的边缘较厚，1.2/2.2 像素两档取最大值更稳。
                d1 = self._pixel_difference(work, x + nx * 1.2, y + ny * 1.2,
                                             x - nx * 1.2, y - ny * 1.2)
                d2 = self._pixel_difference(work, x + nx * 2.2, y + ny * 2.2,
                                             x - nx * 2.2, y - ny * 2.2)
                if max(d1, d2) >= float(contrast_threshold):
                    hit += 1
                total += 1
        return (float(hit) / float(total)) if total > 0 else 0.0

    def _rect_corner_support(self, work, corners, contrast_threshold):
        """
        检查候选四个“角点”是否真的落在图像边缘上。

        find_rects 返回的是四边形候选。圆形有时会被拟合成外接/近外接菱形，
        此时四条虚拟边可能因为低分辨率/模糊而获得一定 edge_support，
        但菱形的四个顶点往往不是真实角点。

        对每个角点，以“中心 -> 角点”的径向方向，在角点内外多组距离取样；
        真矩形角点会出现明显的内外像素跃迁，而圆形外接菱形的虚假角通常不会。
        只用 get_pixel，不创建额外图像缓冲。
        """
        if corners is None or len(corners) != 4:
            return 0.0
        try:
            cx = sum(float(p[0]) for p in corners) / 4.0
            cy = sum(float(p[1]) for p in corners) / 4.0
        except Exception:
            return 0.0

        hit = 0
        total = 0
        # mean_pool 后角点可能有 1~2 px 定位误差，使用三档距离取最大对比。
        steps = (1.4, 2.6, 4.2)
        for p in corners:
            vx = float(p[0]) - cx
            vy = float(p[1]) - cy
            ln = math.sqrt(vx * vx + vy * vy)
            if ln < 3.0:
                continue
            ux, uy = vx / ln, vy / ln
            best = 0.0
            for d in steps:
                # inside: 朝中心；outside: 离开中心。
                diff = self._pixel_difference(
                    work,
                    float(p[0]) - ux * d, float(p[1]) - uy * d,
                    float(p[0]) + ux * d, float(p[1]) + uy * d,
                )
                if diff > best:
                    best = diff
            if best >= float(contrast_threshold):
                hit += 1
            total += 1
        return float(hit) / float(total) if total > 0 else 0.0

    def _rect_circle_veto_filter(self, work, rects, cfg, roi, total_div, work_w, work_h):
        """
        矩形模式下的圆形互斥验证。

        只对“接近正方形”的矩形候选执行一次官方 find_circles()：
        - 普通长方形不会触发额外霍夫圆，保持帧率；
        - 如果同位置存在强圆，且圆直径与候选短边接近，则判定为圆形误拟合并删除。

        任何霍夫圆异常/内存不足都只会跳过互斥，不影响矩形主流程。
        """
        if (not rects or not cfg.get("smart_filter", True)
                or not cfg.get("rect_circle_veto", True)):
            return rects

        aspect_limit = max(1.0, float(cfg.get("rect_circle_aspect_max", 145)) / 100.0)
        suspicious = []
        for r in rects:
            short_side = float(max(1, min(r.get("dw", 1), r.get("dh", 1))))
            long_side = float(max(r.get("dw", 1), r.get("dh", 1)))
            if long_side / short_side <= aspect_limit:
                suspicious.append(r)
        if not suspicious:
            return rects

        try:
            # 完全沿用官方 find_circles 的现有平台参数，不使用文档之外的参数。
            circles_raw = self._detect(work, cfg, "circle", roi, total_div)
            circles = self._filter_circles(circles_raw, cfg, work, work_w, work_h, roi)
        except Exception as e:
            # 排圆是二次校验，失败时宁可保留主矩形检测，也不让整个页面报错。
            if not is_fast_fb_error(e):
                print("[矩形排圆] 圆形二次校验跳过:", e)
            return rects

        if not circles:
            return rects

        center_ratio = float(cfg.get("rect_circle_center_ratio", 28)) / 100.0
        diam_min = float(cfg.get("rect_circle_diam_min", 58)) / 100.0
        diam_max = float(cfg.get("rect_circle_diam_max", 118)) / 100.0

        kept = []
        for r in rects:
            short_side = float(max(1, min(r.get("dw", 1), r.get("dh", 1))))
            long_side = float(max(r.get("dw", 1), r.get("dh", 1)))
            # 长方形直接保留，不额外套圆形规则。
            if long_side / short_side > aspect_limit:
                kept.append(r)
                continue

            rejected = False
            for c in circles:
                d = math.sqrt((float(r["cx"]) - float(c["cx"])) ** 2 +
                              (float(r["cy"]) - float(c["cy"])) ** 2)
                diam_ratio = (2.0 * float(c["r"])) / short_side
                if d <= short_side * center_ratio and diam_min <= diam_ratio <= diam_max:
                    rejected = True
                    break
            if not rejected:
                kept.append(r)
        return kept

    def _filter_rects(self, results, cfg, work, work_w, work_h):
        raw = []
        for obj in (results or []):
            try:
                x, y, w, h = obj.rect()
                corners_w = list(obj.corners())
                if len(corners_w) != 4:
                    continue
                corners_d = [(self.sx(p[0], work_w), self.sy(p[1], work_h)) for p in corners_w]
                side_lengths = []
                for i in range(4):
                    a, b = corners_d[i], corners_d[(i + 1) % 4]
                    side_lengths.append(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
                min_side = min(side_lengths) if side_lengths else 0.0
                angle_err = self._corner_angle_error(corners_w)
                support = self._rect_edge_support(work, corners_w, cfg.get("rect_edge_contrast", 28))
                corner_support = self._rect_corner_support(work, corners_w, cfg.get("rect_edge_contrast", 28))
                try:
                    mag = float(obj.magnitude())
                except Exception:
                    mag = 0.0
                dx = self.sx(x, work_w); dy = self.sy(y, work_h)
                dw = max(1, self.sx(w, work_w)); dh = max(1, self.sy(h, work_h))
                cx, cy = dx + dw // 2, dy + dh // 2
                raw.append({"obj": obj, "corners": corners_d, "corners_work": corners_w,
                            "dx": dx, "dy": dy, "dw": dw, "dh": dh, "cx": cx, "cy": cy,
                            "min_side": min_side, "angle_err": angle_err,
                            "edge_support": support, "corner_support": corner_support,
                            "magnitude": mag})
            except Exception:
                pass

        raw.sort(key=lambda r: (r.get("corner_support", 0.0), r["edge_support"], r["magnitude"], r["min_side"]), reverse=True)
        kept = []
        for r in raw:
            if cfg.get("smart_filter", True):
                if r["min_side"] < float(cfg.get("rect_min_side", 55)):
                    continue
                if r["angle_err"] > float(cfg.get("rect_max_angle_error", 28)):
                    continue
                if r["edge_support"] * 100.0 < float(cfg.get("rect_edge_support", 55)):
                    continue
                if r.get("corner_support", 0.0) * 100.0 < float(cfg.get("rect_corner_support", 75)):
                    continue

            duplicate = False
            for k in kept:
                cd = math.sqrt((r["cx"] - k["cx"]) ** 2 + (r["cy"] - k["cy"]) ** 2)
                size_d = abs(r["dw"] - k["dw"]) + abs(r["dh"] - k["dh"])
                if cd <= 18 and size_d <= 40:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(r)
            if len(kept) >= int(cfg["max_results"]):
                break
        return kept

    def process(self, img, osd):
        cfg = self.app.cfg["geometry"]
        mode = cfg.get("mode", "line")
        osd.clear()
        targets = []
        results = None
        work = img
        total_div = max(1, int(GEOMETRY_WORK_DIV))

        try:
            # 原地降采样，不额外申请大图，继续解决 fast frame buffer 问题。
            pool_image_in_place(work, total_div)
            work_w, work_h = int(work.width()), int(work.height())
            base_roi = self.app.region_manager.cv_roi("geometry") if self.app.region_manager is not None else None
            roi = scale_roi_between_images(base_roi, CV_W, CV_H, work_w, work_h)

            try:
                results = self._detect(work, cfg, mode, roi, total_div)
            except Exception as first_error:
                if not is_fast_fb_error(first_error):
                    raise
                pool_image_in_place(work, 2)
                total_div *= 2
                work_w, work_h = int(work.width()), int(work.height())
                roi = scale_roi_between_images(base_roi, CV_W, CV_H, work_w, work_h)
                results = self._detect(work, cfg, mode, roi, total_div)
                if not self.memory_fallback_once:
                    self.memory_fallback_once = True
                    print("[几何] fast frame buffer不足，已自动切换更低工作分辨率")

            if mode in ("line", "segment"):
                picked = self._filter_lines(results, cfg, mode, work_w, work_h)
                for i, t in enumerate(picked):
                    t["id"] = i
                    x1, y1, x2, y2 = t["line"]
                    color = PALETTE[i % len(PALETTE)]
                    osd.draw_line(x1, y1, x2, y2, color=color, thickness=3)
                    if cfg.get("show_center", True):
                        draw_cross_lines(osd, t["cx"], t["cy"], color=color, size=6)
                    text = t["label"]
                    if cfg.get("show_angle", True):
                        text += " %.0f°" % t["angle"]
                    osd.draw_string_advanced(t["cx"] + 5, safe_label_y(t["cy"], 0), 16, text, color=color)
                    targets.append(t)

            elif mode == "circle":
                picked = self._filter_circles(results, cfg, work, work_w, work_h, roi)
                for i, c in enumerate(picked):
                    cx, cy, r = c["cx"], c["cy"], c["r"]
                    color = PALETTE[i % len(PALETTE)]
                    osd.draw_circle(cx, cy, r, color=color, thickness=3)
                    if cfg.get("show_center", True):
                        draw_cross_lines(osd, cx, cy, color=color, size=6)
                    q = int(round(float(c.get("quality", 1.0)) * 100.0))
                    label_text = "圆 R:%d" % r
                    if cfg.get("smart_filter", True) and cfg.get("circle_verify", True):
                        label_text += " Q:%d" % q
                    osd.draw_string_advanced(cx + r + 4, safe_label_y(cy - r, r * 2), 16,
                                             label_text, color=color)
                    targets.append({"id": i, "class_id": 2, "label": "圆形", "score": float(c.get("quality", 1.0)),
                                    "cx": cx, "cy": cy, "w": r * 2, "h": r * 2,
                                    "area": math.pi * r * r, "angle": 0.0,
                                    "magnitude": c["magnitude"], "radius": r,
                                    "edge_support": c.get("edge_support", 0.0),
                                    "quadrant_support": c.get("quadrant_support", 0.0)})

            else:
                picked = self._filter_rects(results, cfg, work, work_w, work_h)
                # 二次互斥：只对近正方形候选检查同位置强圆，压掉“圆 -> 菱形矩形”。
                picked = self._rect_circle_veto_filter(work, picked, cfg, roi, total_div, work_w, work_h)
                for i, rinfo in enumerate(picked):
                    corners = rinfo["corners"]
                    dx, dy, dw, dh = rinfo["dx"], rinfo["dy"], rinfo["dw"], rinfo["dh"]
                    angle = 0.0
                    if len(corners) >= 2:
                        angle = math.atan2(corners[1][1] - corners[0][1],
                                           corners[1][0] - corners[0][0]) * 180.0 / math.pi
                    color = PALETTE[i % len(PALETTE)]
                    for k in range(4):
                        p0, p1 = corners[k], corners[(k + 1) % 4]
                        osd.draw_line(p0[0], p0[1], p1[0], p1[1], color=color, thickness=3)
                    cx, cy = rinfo["cx"], rinfo["cy"]
                    if cfg.get("show_center", True):
                        draw_cross_lines(osd, cx, cy, color=color, size=6)
                    text = "矩形"
                    if cfg.get("show_angle", True):
                        text += " %.0f°" % angle
                    osd.draw_string_advanced(dx, safe_label_y(dy, dh), 16, text, color=color)
                    targets.append({"id": i, "class_id": 3, "label": "矩形", "score": 1.0,
                                    "cx": cx, "cy": cy, "w": dw, "h": dh,
                                    "area": float(dw * dh), "angle": angle,
                                    "magnitude": rinfo["magnitude"],
                                    "edge_support": rinfo["edge_support"],
                                    "corner_support": rinfo.get("corner_support", 0.0)})

        except Exception as e:
            self.app.results.clear("geometry")
            if is_fast_fb_error(e):
                msg = "几何内存不足：请缩小识别区域"
            else:
                msg = "几何识别异常: " + str(e)
            draw_short_status(osd, msg, COLOR_RED)
            print("[几何识别异常]", e)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            return
        finally:
            results = None
            work = None

        self.app.results.publish("geometry", targets, {"mode": mode, "work_div": total_div})
        if not targets:
            draw_short_status(osd, "等待%s" % self.MODE_NAMES.get(mode, "几何目标"), COLOR_YELLOW)
        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)


# ============================================================
# 模板识别 - 灰度 NCC 模板匹配
# ============================================================

class TemplateEngine:
    def __init__(self, app):
        self.app = app
        self.template_img = None
        self.loaded_slot = -1
        self.last_error = ""

    def path(self, slot=None):
        if slot is None:
            slot = int(self.app.cfg["template"].get("selected", 0))
        return TEMPLATE_PATH_FMT % (int(slot) + 1)

    def legacy_path(self, slot=None):
        if slot is None:
            slot = int(self.app.cfg["template"].get("selected", 0))
        return TEMPLATE_LEGACY_PATH_FMT % (int(slot) + 1)

    def _existing_path(self, slot=None):
        p = self.path(slot)
        try:
            with open(p, "rb"):
                return p
        except Exception:
            pass
        # 兼容之前尝试保存过的 BMP；新保存统一使用 PGM。
        lp = self.legacy_path(slot)
        try:
            with open(lp, "rb"):
                return lp
        except Exception:
            return None

    def exists(self, slot=None):
        return self._existing_path(slot) is not None

    def unload(self):
        self.template_img = None
        self.loaded_slot = -1
        gc.collect()

    def ensure_loaded(self):
        slot = int(self.app.cfg["template"].get("selected", 0))
        if self.template_img is not None and self.loaded_slot == slot:
            return True
        self.unload()
        path = self._existing_path(slot)
        if path is None:
            self.last_error = "模板文件不存在"
            return False
        try:
            # Image(path) 官方支持指定 ALLOC_HEAP，模板常驻普通堆，不占 fast frame buffer。
            alloc_heap = getattr(image, "ALLOC_HEAP", None)
            if alloc_heap is not None:
                try:
                    loaded = image.Image(path, alloc=alloc_heap)
                except TypeError:
                    loaded = image.Image(path)
            else:
                loaded = image.Image(path)

            try:
                fmt = loaded.format()
            except Exception:
                fmt = None
            if fmt != image.GRAYSCALE:
                converted = grayscale_heap_copy(loaded)
                loaded = None
                gc.collect()
                loaded = converted

            self.template_img = loaded
            self.loaded_slot = slot
            self.last_error = ""
            return True
        except Exception as e:
            self.last_error = str(e)
            self.template_img = None
            self.loaded_slot = -1
            print("[模板加载失败]", e)
            return False

    def _prepare_gray_work(self, img):
        """固定 2 倍原地降采样，再用普通堆创建一张很小的灰度工作图。"""
        work = img
        pool_image_in_place(work, TEMPLATE_WORK_DIV)
        gray = grayscale_heap_copy(work)
        return gray, int(work.width()), int(work.height())

    def capture_from_frame(self, img):
        base_roi = self.app.region_manager.cv_roi("template_capture")
        if base_roi is None:
            self.last_error = "请先用“框选模板”选择模板区域"
            return False
        gray = None
        work = None
        try:
            try:
                os.stat(TEMPLATE_DIR)
            except OSError:
                os.mkdir(TEMPLATE_DIR)

            # 关键：先把当前 RGB565 帧原地缩成 200x120，再创建灰度副本。
            # 不再对完整 400x240 图直接 to_grayscale()。
            work = img
            pool_image_in_place(work, TEMPLATE_WORK_DIV)
            work_w, work_h = int(work.width()), int(work.height())
            roi = scale_roi_between_images(base_roi, CV_W, CV_H, work_w, work_h)
            if roi is None or roi[2] < 6 or roi[3] < 6:
                self.last_error = "模板区域太小，请重新框选"
                return False

            gray = grayscale_heap_copy(work)
            # PGM 天然保存灰度数据，之后加载通常无需二次颜色转换。
            gray.save(self.path(), roi=roi)
            gray = None
            work = None
            self.unload()
            ok = self.ensure_loaded()
            if ok:
                self.last_error = ""
            return ok
        except Exception as e:
            self.last_error = str(e)
            print("[模板保存失败]", e)
            return False
        finally:
            gray = None
            work = None
            gc.collect()

    def delete_selected(self):
        removed = False
        for p in (self.path(), self.legacy_path()):
            try:
                os.remove(p)
                removed = True
            except Exception:
                pass
        self.unload()
        return removed

    def process(self, img, osd):
        cfg = self.app.cfg["template"]
        osd.clear()
        if not self.ensure_loaded():
            self.app.results.clear("template")
            draw_short_status(osd, "当前模板为空，请先框选并保存", COLOR_YELLOW)
            self.app.draw_runtime_decor(osd)
            Display.show_image(osd, layer=Display.LAYER_OSD0)
            return

        gray = None
        work = None
        try:
            work = img
            pool_image_in_place(work, TEMPLATE_WORK_DIV)
            work_w, work_h = int(work.width()), int(work.height())
            gray = grayscale_heap_copy(work)

            base_roi = self.app.region_manager.cv_roi("template_search")
            roi = scale_roi_between_images(base_roi, CV_W, CV_H, work_w, work_h)
            search_mode = image.SEARCH_DS if cfg.get("search") == "fast" else image.SEARCH_EX

            if roi is None:
                rect = gray.find_template(self.template_img, float(cfg["threshold"]),
                                          step=int(cfg["step"]), search=search_mode)
            else:
                rect = gray.find_template(self.template_img, float(cfg["threshold"]), roi=roi,
                                          step=int(cfg["step"]), search=search_mode)

            targets = []
            if rect is not None:
                x, y, w, h = rect
                dx = int(float(x) * DISPLAY_W / float(max(1, work_w)))
                dy = int(float(y) * DISPLAY_H / float(max(1, work_h)))
                dw = max(1, int(float(w) * DISPLAY_W / float(max(1, work_w))))
                dh = max(1, int(float(h) * DISPLAY_H / float(max(1, work_h))))
                cx, cy = dx + dw // 2, dy + dh // 2
                if cfg.get("show_box", True):
                    osd.draw_rectangle(dx, dy, dw, dh, color=COLOR_GREEN, thickness=3)
                if cfg.get("show_center", True):
                    draw_cross_lines(osd, cx, cy, color=COLOR_GREEN, size=7)
                slot = int(cfg.get("selected", 0))
                osd.draw_string_advanced(dx, safe_label_y(dy, dh), 17,
                                         "模板%d 匹配" % (slot + 1), color=COLOR_GREEN)
                targets.append({"id": 0, "class_id": slot, "label": "模板%d" % (slot + 1),
                                "score": 1.0, "cx": cx, "cy": cy, "w": dw, "h": dh,
                                "area": float(dw * dh), "angle": 0.0})

            self.app.results.publish(
                "template", targets,
                {"slot": int(cfg.get("selected", 0)),
                 "threshold": float(cfg["threshold"]),
                 "work_div": int(TEMPLATE_WORK_DIV)}
            )
            if not targets:
                draw_short_status(osd, "未找到当前模板", COLOR_YELLOW)

        except Exception as e:
            self.last_error = str(e)
            self.app.results.clear("template")
            if is_fast_fb_error(e):
                draw_short_status(osd, "模板内存不足：请缩小搜索区域", COLOR_RED)
            else:
                draw_short_status(osd, "模板识别异常: " + str(e), COLOR_RED)
            print("[模板识别异常]", e)
        finally:
            gray = None
            work = None
            # 模板灰度工作图每帧都属于临时对象；及时让普通堆回收，避免长期碎片化。
            gc.collect()

        self.app.draw_runtime_decor(osd)
        Display.show_image(osd, layer=Display.LAYER_OSD0)


# ============================================================
# 公共结果中心 / 识别区域 / 串口通信
# ============================================================

class VisionResultCenter:
    """所有算法统一把当前帧结果发布到这里，显示/串口不直接耦合算法内部实现。"""
    SOURCE_IDS = {
        "none": 0,
        "yolo": 1,
        "color": 2,
        "shape": 3,
        "line": 4,
        "code": 5,
        "geometry": 6,
        "template": 7,
    }

    SOURCE_NAMES = {
        "yolo": "YOLO识别",
        "color": "颜色识别",
        "shape": "形状识别",
        "line": "巡线导航",
        "code": "码类识别",
        "geometry": "几何视觉",
        "template": "模板识别",
    }

    def __init__(self):
        self.latest = {}

    def publish(self, source, targets=None, meta=None):
        if targets is None:
            targets = []
        if meta is None:
            meta = {}
        self.latest[str(source)] = {
            "source": str(source),
            "timestamp": time.ticks_ms(),
            "targets": list(targets),
            "meta": meta,
        }

    def clear(self, source):
        self.publish(source, [], {})

    def get(self, source, fresh_ms=SERIAL_RESULT_FRESH_MS):
        item = self.latest.get(str(source))
        if item is None:
            return None
        if fresh_ms is not None:
            try:
                if safe_ticks_diff(time.ticks_ms(), item["timestamp"]) > int(fresh_ms):
                    return None
            except Exception:
                return None
        return item

    def source_id(self, source):
        return int(self.SOURCE_IDS.get(str(source), 0))

    def source_name(self, source):
        return self.SOURCE_NAMES.get(str(source), str(source))


class RegionManager:
    """
    识别区域统一使用显示坐标保存。

    触摸策略特别针对庐山派 3.1 寸屏：
    - 只使用“当前是否有触点 + x/y 坐标”判断拖动，不依赖 event。
    - 单指第一次出现触点：记录起点。
    - 手指保持按住并移动：持续更新终点并实时画框。
    - 连续空读达到 REGION_RELEASE_GAP_MS：判定抬手并结束本次拖动。

    这样即使固件里的 event 始终为 EVENT_DOWN / EVENT_NONE，ROI 也能正常拖动，
    不会因为每一帧都重新记录起点而只显示 1~2 像素的小框。

    算法策略：
    - color/code/geometry/template: 转换为 400x240 后直接传给官方 roi 参数。
    - shape: 处理阶段使用裁剪区域并把坐标偏移回原图。
    - yolo: 保持整帧推理，只接受中心点位于识别区域的检测结果，稳定优先。
    """
    BTN_CANCEL = (18, 420, 112, 44)
    BTN_FULL = (148, 420, 112, 44)
    BTN_DISABLE = (278, 420, 132, 44)
    BTN_OK = (668, 420, 112, 44)

    # 触摸驱动偶尔可能瞬间返回空元组，因此不把第一次空读立即当成抬手。
    # 90ms 连续空读才真正结束拖动，兼顾稳定性和操作跟手性。
    REGION_RELEASE_GAP_MS = 90
    REGION_MIN_SIZE = 20
    REGION_START_MARK_SIZE = 8

    def __init__(self, app):
        self.app = app
        self.active = False
        self.key = None
        self.return_page = None

        self.touching = False
        self.dragging = False
        self.down_x = 0
        self.down_y = 0
        self.cur_x = 0
        self.cur_y = 0
        self.last_x = 0
        self.last_y = 0

        self.temp_rect = None
        self.button_down = None
        self.empty_since = 0

    def _entry(self, key):
        return self.app.cfg["regions"].get(key, default_region())

    def enabled(self, key):
        try:
            return bool(self._entry(key).get("enabled", False))
        except Exception:
            return False

    def display_rect(self, key, only_if_enabled=True):
        e = self._entry(key)
        if only_if_enabled and not e.get("enabled", False):
            return None
        try:
            r = e.get("rect", [0, 0, DISPLAY_W, DISPLAY_H])
            return (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        except Exception:
            return (0, 0, DISPLAY_W, DISPLAY_H)

    def cv_roi(self, key):
        r = self.display_rect(key, only_if_enabled=True)
        if r is None:
            return None
        x, y, w, h = r
        rx = int(clamp(int(x * CV_W / DISPLAY_W), 0, CV_W - 2))
        ry = int(clamp(int(y * CV_H / DISPLAY_H), 0, CV_H - 2))
        rw = int(clamp(int(w * CV_W / DISPLAY_W), 2, CV_W - rx))
        rh = int(clamp(int(h * CV_H / DISPLAY_H), 2, CV_H - ry))
        return (rx, ry, rw, rh)

    def contains_display(self, key, x, y):
        r = self.display_rect(key, only_if_enabled=True)
        if r is None:
            return True
        rx, ry, rw, rh = r
        return x >= rx and x < rx + rw and y >= ry and y < ry + rh

    def center(self, key):
        r = self.display_rect(key, only_if_enabled=True)
        if r is None:
            return DISPLAY_W // 2, DISPLAY_H // 2
        x, y, w, h = r
        return int(x + w // 2), int(y + h // 2)

    def draw_active(self, osd, key):
        if not key or not self.enabled(key):
            return
        r = self.display_rect(key, only_if_enabled=True)
        if r is None:
            return
        x, y, w, h = r
        osd.draw_rectangle(x, y, w, h, color=COLOR_CYAN, thickness=2)
        osd.draw_string_advanced(x + 4, y + 4, 15, "识别区域", color=COLOR_CYAN)

    def _inside(self, r, x, y):
        return x >= r[0] and x < r[0] + r[2] and y >= r[1] and y < r[1] + r[3]

    def _button_at(self, x, y):
        if self._inside(self.BTN_CANCEL, x, y):
            return "cancel"
        if self._inside(self.BTN_FULL, x, y):
            return "full"
        if self._inside(self.BTN_DISABLE, x, y):
            return "disable"
        if self._inside(self.BTN_OK, x, y):
            return "ok"
        return None

    def _draw_button(self, osd, rect, text, border):
        x, y, w, h = rect
        osd.draw_rectangle(x, y, w, h, color=(28, 34, 44), fill=True)
        osd.draw_rectangle(x, y, w, h, color=border, thickness=2)
        osd.draw_string_advanced(x + 12, y + 11, 18, text, color=COLOR_WHITE)

    def _normalized_drag_rect(self):
        x1 = min(self.down_x, self.cur_x)
        y1 = min(self.down_y, self.cur_y)
        x2 = max(self.down_x, self.cur_x)
        y2 = max(self.down_y, self.cur_y)

        x1 = int(clamp(x1, 0, DISPLAY_W - 2))
        y1 = int(clamp(y1, 0, DISPLAY_H - 2))
        x2 = int(clamp(x2, 0, DISPLAY_W - 1))
        y2 = int(clamp(y2, 0, DISPLAY_H - 1))

        # 不强行把未移动的点扩成 1~2 像素 ROI；真实宽高就是两点之间的距离。
        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def _drag_distance2(self):
        dx = int(self.cur_x) - int(self.down_x)
        dy = int(self.cur_y) - int(self.down_y)
        return dx * dx + dy * dy

    def draw_editor(self):
        if not self.active or self.app.overlay_img is None:
            return

        osd = self.app.overlay_img
        osd.clear()
        osd.draw_rectangle(10, 8, 780, 55, color=(20, 26, 34), fill=True)
        osd.draw_rectangle(10, 8, 780, 55, color=COLOR_BLUE, thickness=2)
        osd.draw_string_advanced(24, 17, 22, "单指按住并拖动框选，松手后点击“确认”", color=COLOR_WHITE)
        osd.draw_string_advanced(24, 42, 14, "不用双指；区域只影响当前功能，可随时关闭或恢复全屏", color=COLOR_LIGHT)

        rect = self.temp_rect
        if self.dragging:
            # 刚按下还没有移动时只画起点十字，不画“1像素ROI框”。
            if self._drag_distance2() < 9:
                draw_cross_lines(osd, self.down_x, self.down_y,
                                 color=COLOR_YELLOW,
                                 size=self.REGION_START_MARK_SIZE,
                                 thickness=2)
                osd.draw_string_advanced(
                    int(clamp(self.down_x + 10, 0, DISPLAY_W - 120)),
                    int(clamp(self.down_y + 8, 66, DISPLAY_H - 55)),
                    16, "起点", color=COLOR_YELLOW)
                rect = None
            else:
                rect = self._normalized_drag_rect()

        if rect is not None:
            x, y, w, h = rect
            if w > 0 and h > 0:
                osd.draw_rectangle(x, y, w, h, color=COLOR_CYAN, thickness=4)
                tx = int(clamp(x + 5, 0, DISPLAY_W - 190))
                ty = int(clamp(max(66, y + 5), 66, DISPLAY_H - 55))
                osd.draw_string_advanced(tx, ty, 16,
                                         "%d,%d  %dx%d" % (x, y, w, h),
                                         color=COLOR_YELLOW)

        self._draw_button(osd, self.BTN_CANCEL, "取消", COLOR_GRAY)
        self._draw_button(osd, self.BTN_FULL, "全屏", COLOR_BLUE)
        self._draw_button(osd, self.BTN_DISABLE, "关闭区域", COLOR_ORANGE)
        self._draw_button(osd, self.BTN_OK, "确认", COLOR_GREEN)
        Display.show_image(osd, layer=Display.LAYER_OSD0)

    def start(self, key, return_page):
        if key not in ("shape", "color", "yolo", "code", "geometry", "template_search", "template_capture"):
            return False

        self.active = True
        self.key = key
        self.return_page = return_page
        self.touching = False
        self.dragging = False
        self.button_down = None
        self.empty_since = 0
        self.temp_rect = self.display_rect(key, only_if_enabled=False)

        if self.app.panel_img is not None:
            self.app.panel_img.clear()
            Display.show_image(self.app.panel_img, x=PANEL_X, y=0, layer=Display.LAYER_OSD1)
        if self.app.ui is not None:
            self.app.ui.pool.reset()

        self.draw_editor()
        return True

    def _finish(self):
        self.active = False
        self.touching = False
        self.dragging = False
        self.button_down = None
        self.empty_since = 0
        self.app.clear_result()
        if self.app.ui is not None:
            self.app.ui.invalidate()
            self.app.ui.render()
        if self.app.touch_ctrl is not None:
            self.app.touch_ctrl.force_rearm(260)

    def _execute_button(self, action):
        if action == "cancel":
            self._finish()
            return True

        if action == "full":
            self.temp_rect = (0, 0, DISPLAY_W, DISPLAY_H)
            self.draw_editor()
            return False

        if action == "disable":
            e = self._entry(self.key)
            e["enabled"] = False
            e["rect"] = [0, 0, DISPLAY_W, DISPLAY_H]
            self.app.notice("已关闭当前功能的识别区域", COLOR_ORANGE, 1400)
            self._finish()
            return True

        if action == "ok":
            if (self.temp_rect is None or
                    self.temp_rect[2] < self.REGION_MIN_SIZE or
                    self.temp_rect[3] < self.REGION_MIN_SIZE):
                self.app.notice("区域太小，请重新框选", COLOR_RED, 1500)
                self.draw_editor()
                return False

            e = self._entry(self.key)
            e["enabled"] = True
            e["rect"] = [int(v) for v in self.temp_rect]
            self.app.notice("识别区域已保存", COLOR_GREEN, 1300)
            self._finish()
            return True

        return False

    def _release_touch(self):
        """连续空读确认松手后，提交当前按钮或结束一次 ROI 拖动。"""
        if not self.touching:
            return

        self.touching = False
        self.empty_since = 0

        if self.button_down is not None:
            action = self.button_down if self._button_at(self.last_x, self.last_y) == self.button_down else None
            self.button_down = None
            if action is not None:
                closed = self._execute_button(action)
                if closed:
                    return
            if self.active:
                self.draw_editor()
            return

        if self.dragging:
            self.cur_x = int(self.last_x)
            self.cur_y = int(self.last_y)
            r = self._normalized_drag_rect()
            self.dragging = False

            if r[2] >= self.REGION_MIN_SIZE and r[3] >= self.REGION_MIN_SIZE:
                self.temp_rect = r
                self.app.notice("区域已框选，点击“确认”保存", COLOR_GREEN, 1200)
            else:
                self.app.notice("区域太小，请按住后拖动至少20像素", COLOR_YELLOW, 1400)

            self.draw_editor()
            return

        self.draw_editor()

    def handle_event(self, x, y, event=None):
        """
        ROI 编辑期间故意不读取 event 的语义。

        庐山派 3.1 寸屏当前资料明确提示 event/track_id 并不可靠；坐标可正常获取。
        因此这里把每一个“非空触点”都当成同一根手指的当前位置：
        第一个点 = 按下，后续点 = 拖动。
        """
        if not self.active:
            return

        x = int(clamp(int(x), 0, DISPLAY_W - 1))
        y = int(clamp(int(y), 0, DISPLAY_H - 1))
        self.empty_since = 0
        self.last_x = x
        self.last_y = y

        # 第一次读到触点：决定这是按钮点击还是 ROI 拖动起点。
        if not self.touching:
            self.touching = True
            btn = self._button_at(x, y)
            if btn is not None:
                self.button_down = btn
                self.dragging = False
                return

            self.button_down = None
            self.down_x = self.cur_x = x
            self.down_y = self.cur_y = y
            self.dragging = True
            self.draw_editor()
            return

        # 同一次物理接触期间，不论 event 是 DOWN / MOVE / NONE，都只更新坐标。
        if self.button_down is not None:
            return

        if self.dragging:
            self.cur_x = x
            self.cur_y = y
            self.draw_editor()

    def handle_empty(self):
        """
        当前没有触点。
        连续空读一小段时间后才认定真正抬手，避免一次瞬时漏点把拖动截断。
        """
        if not self.active:
            return

        if not self.touching:
            self.empty_since = 0
            return

        now = time.ticks_ms()
        if self.empty_since == 0:
            self.empty_since = now
            return

        if safe_ticks_diff(now, self.empty_since) >= self.REGION_RELEASE_GAP_MS:
            self._release_touch()


class MeasurementManager:
    """统一测量层：不修改各视觉算法，只消费 VisionResultCenter 的标准结果。"""
    SOURCE_NAMES = {
        "current": "当前功能", "yolo": "YOLO识别", "color": "颜色识别",
        "shape": "形状识别", "line": "巡线导航", "code": "码类识别",
        "geometry": "几何视觉", "template": "模板识别",
    }
    RULE_NAMES = {"center": "中心最近", "score": "置信最高", "area": "面积最大"}
    REF_NAMES = {"screen_center": "画面中心", "region_center": "区域中心", "custom": "自定义点"}
    UNIT_NAMES = {"pixel": "像素", "mm": "毫米", "cm": "厘米"}

    def __init__(self, app):
        self.app = app

    def configured_source(self):
        """返回当前测量绑定的真实识别来源。current 在设置页时使用最近一次识别功能。"""
        cfg = self.app.cfg["measurement"]
        wanted = str(cfg.get("source", "current"))
        if wanted == "current":
            if self.app.page in VISION_PAGE_OPTIONS:
                return self.app.page
            last_page = getattr(self.app, "last_vision_page", "color")
            return last_page if last_page in VISION_PAGE_OPTIONS else "color"
        return wanted if wanted in VISION_PAGE_OPTIONS else "color"

    def _source_for_page(self):
        """
        测量叠加只绘制在真正运行该算法的识别页上。
        测量设置页本身不运行颜色/YOLO等算法，因此不能把“来源选择”误解成在设置页加载算法。
        用户可通过“进入来源识别”按钮直接跳到绑定页并开始实时测量。
        """
        source = self.configured_source()
        return source if self.app.page == source else "none"

    def _reference(self, source):
        cfg = self.app.cfg["measurement"]
        mode = cfg.get("reference", "screen_center")
        if mode == "region_center":
            region_key = {"template": "template_search"}.get(source, source)
            if region_key in ("yolo", "color", "shape", "code", "geometry", "template_search"):
                return self.app.region_manager.center(region_key)
        if mode == "custom":
            return int(cfg.get("custom_x", DISPLAY_W // 2)), int(cfg.get("custom_y", DISPLAY_H // 2))
        return DISPLAY_W // 2, DISPLAY_H // 2

    def _select(self, targets, source):
        if not targets:
            return None
        cfg = self.app.cfg["measurement"]
        rule = cfg.get("target_rule", "center")
        best = targets[0]
        if rule == "score":
            for t in targets[1:]:
                if float(t.get("score", 0.0)) > float(best.get("score", 0.0)):
                    best = t
            return best
        if rule == "area":
            for t in targets[1:]:
                if float(t.get("area", 0.0)) > float(best.get("area", 0.0)):
                    best = t
            return best
        rx, ry = self._reference(source)
        bd = (float(best.get("cx", 0)) - rx) ** 2 + (float(best.get("cy", 0)) - ry) ** 2
        for t in targets[1:]:
            d = (float(t.get("cx", 0)) - rx) ** 2 + (float(t.get("cy", 0)) - ry) ** 2
            if d < bd:
                best, bd = t, d
        return best

    def _scale(self):
        cfg = self.app.cfg["measurement"]
        unit = cfg.get("unit", "pixel")
        if unit == "pixel":
            return 1.0, "px", True
        mm_per_px = float(cfg.get("mm_per_pixel", 0.0))
        if mm_per_px <= 0:
            return 0.0, "mm" if unit == "mm" else "cm", False
        if unit == "cm":
            return mm_per_px / 10.0, "cm", True
        return mm_per_px, "mm", True

    def reset_calibration(self):
        cfg = self.app.cfg["measurement"]
        cfg["mm_per_pixel"] = 0.0
        cfg["calibration_p1"] = [0, 0]
        cfg["calibration_p2"] = [0, 0]

    def draw_overlay(self, osd):
        cfg = self.app.cfg.get("measurement", {})
        if not cfg.get("enabled", False):
            return
        source = self._source_for_page()
        if source == "none":
            return
        result = self.app.results.get(source, 1000)
        if result is None:
            return
        t = self._select(result.get("targets", []), source)
        if t is None:
            return

        cx = int(round(t.get("cx", 0)))
        cy = int(round(t.get("cy", 0)))
        rx, ry = self._reference(source)
        dx, dy = cx - rx, cy - ry
        dist_px = math.sqrt(dx * dx + dy * dy)
        scale, unit_name, calibrated = self._scale()

        if cfg.get("show_center", True):
            draw_cross_lines(osd, cx, cy, color=COLOR_PURPLE, size=9, thickness=2)
            draw_cross_lines(osd, rx, ry, color=COLOR_CYAN, size=7, thickness=2)
            osd.draw_line(rx, ry, cx, cy, color=COLOR_YELLOW, thickness=2)

        # 文字放在左上，不占右侧控制面板。
        label = str(t.get("label", "目标"))
        if len(label) > 10:
            label = label[:10]
        osd.draw_string_advanced(8, 8, 17, "测量 %s  X:%d Y:%d" % (label, cx, cy), color=COLOR_PURPLE)

        y = 30
        if cfg.get("show_offset", True):
            if calibrated:
                osd.draw_string_advanced(8, y, 16,
                                         "偏移:%+.1f,%+.1f%s  距离:%.1f%s" %
                                         (dx * scale, dy * scale, unit_name, dist_px * scale, unit_name),
                                         color=COLOR_YELLOW)
            else:
                osd.draw_string_advanced(8, y, 16,
                                         "偏移:%+d,%+dpx  距离:%.1fpx（未标定）" % (dx, dy, dist_px),
                                         color=COLOR_YELLOW)
            y += 21
        if cfg.get("show_size", True):
            w, h = float(t.get("w", 0)), float(t.get("h", 0))
            if calibrated:
                text = "宽高:%.1f×%.1f%s" % (w * scale, h * scale, unit_name)
            else:
                text = "宽高:%d×%dpx" % (int(round(w)), int(round(h)))
            osd.draw_string_advanced(8, y, 15, text, color=COLOR_LIGHT)
            y += 19
        if cfg.get("show_area", False):
            area = float(t.get("area", 0.0))
            if calibrated:
                area_text = "面积:%.1f%s²" % (area * scale * scale, unit_name)
            else:
                area_text = "面积:%dpx²" % int(round(area))
            osd.draw_string_advanced(8, y, 15, area_text, color=COLOR_LIGHT)


class CalibrationManager:
    """
    全屏两点精确标定：
    1) 先用单指在画面上大致点击点1、点2；
    2) 再选择“点1/点2”，使用方向键按像素微调；
    3) 步长可在 1 / 5 / 10 px 之间切换，方向键支持长按连续移动；
    4) 点击“保存标定”后同时保存两个精确坐标与 mm/像素比例。

    触摸仍只依赖“是否有触点 + x/y”，不依赖 event/track_id，
    保持与庐山派触摸屏当前稳定方案一致。
    """

    # 底部两行控制区。点位本身仍可放在整个画面中，只要没有点击到按钮。
    BTN_CANCEL = (10, 378, 82, 38)
    BTN_RESET = (98, 378, 82, 38)
    BTN_P1 = (186, 378, 82, 38)
    BTN_P2 = (274, 378, 82, 38)
    BTN_STEP = (362, 378, 104, 38)

    BTN_LEFT = (474, 378, 62, 38)
    BTN_UP = (542, 378, 62, 38)
    BTN_DOWN = (610, 378, 62, 38)
    BTN_RIGHT = (678, 378, 62, 38)

    BTN_OK = (646, 426, 144, 42)
    BTN_LOAD = (492, 426, 144, 42)
    BTN_CLEAR = (338, 426, 144, 42)

    RELEASE_GAP_MS = 90
    REPEAT_DELAY_MS = 460
    REPEAT_INTERVAL_MS = 120
    STEP_OPTIONS = (1, 5, 10)

    MOVE_ACTIONS = ("left", "right", "up", "down")

    def __init__(self, app):
        self.app = app
        self.active = False
        self.return_page = "measure"
        self.points = []
        self.selected = 0
        self.step_index = 0

        self.touching = False
        self.button_down = None
        self.last_x = 0
        self.last_y = 0
        self.empty_since = 0

        # 方向键长按连调状态。
        self.button_down_ms = 0
        self.last_repeat_ms = 0
        self.repeat_started = False

    def _inside(self, r, x, y):
        return x >= r[0] and x < r[0] + r[2] and y >= r[1] and y < r[1] + r[3]

    def _button_at(self, x, y):
        buttons = (
            (self.BTN_CANCEL, "cancel"),
            (self.BTN_RESET, "reset"),
            (self.BTN_P1, "p1"),
            (self.BTN_P2, "p2"),
            (self.BTN_STEP, "step"),
            (self.BTN_LEFT, "left"),
            (self.BTN_UP, "up"),
            (self.BTN_DOWN, "down"),
            (self.BTN_RIGHT, "right"),
            (self.BTN_CLEAR, "clear"),
            (self.BTN_LOAD, "load"),
            (self.BTN_OK, "ok"),
        )
        for rect, action in buttons:
            if self._inside(rect, x, y):
                return action
        return None

    def _draw_button(self, osd, rect, text, border, active=False, fs=17):
        x, y, w, h = rect
        bg = (43, 70, 88) if active else (28, 34, 44)
        osd.draw_rectangle(x, y, w, h, color=bg, fill=True)
        osd.draw_rectangle(x, y, w, h, color=border, thickness=2)
        tx = x + max(4, (w - len(text) * fs) // 2)
        ty = y + max(4, (h - fs) // 2)
        osd.draw_string_advanced(tx, ty, fs, text, color=COLOR_WHITE)

    def _cfg_point(self, key):
        try:
            p = self.app.cfg["measurement"].get(key, [0, 0])
            x, y = int(p[0]), int(p[1])
            # [0,0] 是默认“未标定”，不自动当作有效点载入。
            if x == 0 and y == 0:
                return None
            return (int(clamp(x, 0, DISPLAY_W - 1)),
                    int(clamp(y, 0, DISPLAY_H - 1)))
        except Exception:
            return None

    def _load_saved_points(self, notify=False):
        p1 = self._cfg_point("calibration_p1")
        p2 = self._cfg_point("calibration_p2")
        if p1 is not None and p2 is not None:
            self.points = [p1, p2]
            self.selected = int(clamp(self.selected, 0, 1))
            if notify:
                self.app.notice("已载入上次保存的两个标定点", COLOR_GREEN, 1000)
            return True
        if notify:
            self.app.notice("当前没有已保存的标定点", COLOR_YELLOW, 1000)
        return False

    def start(self, return_page="measure"):
        self.active = True
        self.return_page = return_page
        self.points = []
        self.selected = 0
        self.step_index = 0
        self.touching = False
        self.button_down = None
        self.empty_since = 0
        self.repeat_started = False

        # 已经做过标定时，重新进入编辑器直接加载上次精确坐标，方便继续微调。
        self._load_saved_points(notify=False)

        if self.app.panel_img is not None:
            self.app.panel_img.clear()
            Display.show_image(self.app.panel_img, x=PANEL_X, y=0, layer=Display.LAYER_OSD1)
        if self.app.ui is not None:
            self.app.ui.pool.reset()
        self.draw_editor()
        return True

    def _finish(self):
        self.active = False
        self.touching = False
        self.button_down = None
        self.empty_since = 0
        self.repeat_started = False
        self.app.clear_result()
        if self.app.ui is not None:
            self.app.ui.invalidate()
            self.app.ui.render()
        if self.app.touch_ctrl is not None:
            self.app.touch_ctrl.force_rearm(260)

    def _current_step(self):
        return int(self.STEP_OPTIONS[int(clamp(self.step_index, 0, len(self.STEP_OPTIONS) - 1))])

    def _pixel_distance(self):
        if len(self.points) != 2:
            return 0.0
        p1, p2 = self.points[0], self.points[1]
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _move_selected(self, dx, dy, repeat=False):
        if len(self.points) < 1:
            self.app.notice("请先在画面上点击标定点", COLOR_YELLOW, 700)
            return False
        idx = int(clamp(self.selected, 0, len(self.points) - 1))
        step = self._current_step()
        x, y = self.points[idx]
        nx = int(clamp(int(x) + int(dx) * step, 0, DISPLAY_W - 1))
        ny = int(clamp(int(y) + int(dy) * step, 0, DISPLAY_H - 1))
        self.points[idx] = (nx, ny)
        self.draw_editor()
        return True

    def _select_point(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self.points):
            self.app.notice("点%d尚未设置" % (idx + 1), COLOR_YELLOW, 700)
            return False
        self.selected = idx
        self.draw_editor()
        return True

    def draw_editor(self):
        if not self.active or self.app.overlay_img is None:
            return
        osd = self.app.overlay_img
        osd.clear()

        # 顶部信息栏。
        osd.draw_rectangle(10, 8, 780, 92, color=(20, 26, 34), fill=True)
        osd.draw_rectangle(10, 8, 780, 92, color=COLOR_BLUE, thickness=2)
        known = float(self.app.cfg["measurement"].get("known_mm", 100.0))
        osd.draw_string_advanced(20, 14, 20,
                                 "先点两个位置，再用方向键逐像素精调",
                                 color=COLOR_WHITE)
        osd.draw_string_advanced(20, 38, 14,
                                 "实际距离 %.1f mm；当前步长 %d px；方向键可长按" %
                                 (known, self._current_step()),
                                 color=COLOR_LIGHT)

        # 精确坐标显示。
        if len(self.points) >= 1:
            p1 = self.points[0]
            c1 = COLOR_YELLOW
            osd.draw_string_advanced(20, 62, 16,
                                     "点1  X:%d  Y:%d%s" %
                                     (p1[0], p1[1], "  ←当前" if self.selected == 0 else ""),
                                     color=c1)
        else:
            osd.draw_string_advanced(20, 62, 16, "点1  未设置", color=COLOR_GRAY)

        if len(self.points) >= 2:
            p2 = self.points[1]
            c2 = COLOR_GREEN
            osd.draw_string_advanced(285, 62, 16,
                                     "点2  X:%d  Y:%d%s" %
                                     (p2[0], p2[1], "  ←当前" if self.selected == 1 else ""),
                                     color=c2)
            dx = p2[0] - self.points[0][0]
            dy = p2[1] - self.points[0][1]
            d = self._pixel_distance()
            osd.draw_string_advanced(548, 62, 15,
                                     "ΔX:%+d ΔY:%+d  D:%.2fpx" % (dx, dy, d),
                                     color=COLOR_CYAN)
        else:
            osd.draw_string_advanced(285, 62, 16, "点2  未设置", color=COLOR_GRAY)

        # 标定点与连接线。当前选中点增加一层白色十字，明确正在调谁。
        for i, p in enumerate(self.points):
            color = COLOR_YELLOW if i == 0 else COLOR_GREEN
            if i == self.selected:
                draw_cross_lines(osd, p[0], p[1], color=COLOR_WHITE, size=16, thickness=2)
            draw_cross_lines(osd, p[0], p[1], color=color, size=10, thickness=3)
            label_x = int(clamp(p[0] + 12, 0, DISPLAY_W - 110))
            label_y = int(clamp(p[1] + 8, 102, DISPLAY_H - 55))
            osd.draw_string_advanced(label_x, label_y, 16,
                                     "点%d(%d,%d)" % (i + 1, p[0], p[1]),
                                     color=color)

        if len(self.points) == 2:
            p1, p2 = self.points[0], self.points[1]
            osd.draw_line(p1[0], p1[1], p2[0], p2[1], color=COLOR_CYAN, thickness=2)

        # 两行精调控制。
        self._draw_button(osd, self.BTN_CANCEL, "取消", COLOR_GRAY)
        self._draw_button(osd, self.BTN_RESET, "重选", COLOR_ORANGE)
        self._draw_button(osd, self.BTN_P1, "点1", COLOR_YELLOW,
                          active=(len(self.points) >= 1 and self.selected == 0))
        self._draw_button(osd, self.BTN_P2, "点2", COLOR_GREEN,
                          active=(len(self.points) >= 2 and self.selected == 1))
        self._draw_button(osd, self.BTN_STEP, "步长:%d" % self._current_step(), COLOR_CYAN)

        self._draw_button(osd, self.BTN_LEFT, "左", COLOR_BLUE)
        self._draw_button(osd, self.BTN_UP, "上", COLOR_BLUE)
        self._draw_button(osd, self.BTN_DOWN, "下", COLOR_BLUE)
        self._draw_button(osd, self.BTN_RIGHT, "右", COLOR_BLUE)

        self._draw_button(osd, self.BTN_CLEAR, "清除保存", COLOR_RED, fs=15)
        self._draw_button(osd, self.BTN_LOAD, "载入保存", COLOR_ORANGE, fs=15)
        self._draw_button(osd, self.BTN_OK, "保存标定", COLOR_GREEN, fs=17)

        Display.show_image(osd, layer=Display.LAYER_OSD0)

    def _save_calibration(self):
        if len(self.points) != 2:
            self.app.notice("请先设置两个标定点", COLOR_YELLOW, 1200)
            self.draw_editor()
            return False

        p1, p2 = self.points[0], self.points[1]
        px = self._pixel_distance()
        if px < 5.0:
            self.app.notice("两个标定点距离太近", COLOR_RED, 1200)
            self.draw_editor()
            return False

        known = float(self.app.cfg["measurement"].get("known_mm", 100.0))
        cfg = self.app.cfg["measurement"]
        cfg["mm_per_pixel"] = known / px
        cfg["calibration_p1"] = [int(p1[0]), int(p1[1])]
        cfg["calibration_p2"] = [int(p2[0]), int(p2[1])]

        # “保存标定”就是持久化保存，不再要求用户回测量页后额外再点一次保存参数。
        saved = False
        try:
            saved = bool(self.app.store.save(self.app.cfg))
        except Exception as e:
            print("[标定保存]", e)

        if saved:
            self.app.notice("标定点与比例已保存", COLOR_GREEN, 1500)
        else:
            self.app.notice("标定已生效，但配置文件保存失败", COLOR_ORANGE, 1800)
        self._finish()
        return True

    def _clear_saved(self):
        cfg = self.app.cfg["measurement"]
        cfg["mm_per_pixel"] = 0.0
        cfg["calibration_p1"] = [0, 0]
        cfg["calibration_p2"] = [0, 0]
        self.points = []
        self.selected = 0
        try:
            self.app.store.save(self.app.cfg)
        except Exception:
            pass
        self.app.notice("已清除保存的标定点", COLOR_ORANGE, 1000)
        self.draw_editor()

    def _execute_button(self, action):
        if action == "cancel":
            self._finish()
            return True
        if action == "reset":
            self.points = []
            self.selected = 0
            self.draw_editor()
            return False
        if action == "p1":
            return self._select_point(0)
        if action == "p2":
            return self._select_point(1)
        if action == "step":
            self.step_index = (self.step_index + 1) % len(self.STEP_OPTIONS)
            self.draw_editor()
            return False
        if action == "left":
            return self._move_selected(-1, 0)
        if action == "right":
            return self._move_selected(1, 0)
        if action == "up":
            return self._move_selected(0, -1)
        if action == "down":
            return self._move_selected(0, 1)
        if action == "load":
            self._load_saved_points(notify=True)
            self.draw_editor()
            return False
        if action == "clear":
            self._clear_saved()
            return False
        if action == "ok":
            return self._save_calibration()
        return False

    def handle_event(self, x, y, event=None):
        if not self.active:
            return
        now = time.ticks_ms()
        x = int(clamp(int(x), 0, DISPLAY_W - 1))
        y = int(clamp(int(y), 0, DISPLAY_H - 1))
        self.last_x, self.last_y = x, y
        self.empty_since = 0

        if not self.touching:
            self.touching = True
            self.button_down = self._button_at(x, y)
            self.button_down_ms = now
            self.last_repeat_ms = now
            self.repeat_started = False

            # 方向键短按应立即移动一次，且长按可以继续连续移动。
            if self.button_down in self.MOVE_ACTIONS:
                self._execute_button(self.button_down)
            return

        # 同一次持续按压：只有仍停在原方向键内才连调，避免滑出按钮仍然移动。
        if self.button_down in self.MOVE_ACTIONS:
            if self._button_at(x, y) != self.button_down:
                return
            elapsed = safe_ticks_diff(now, self.button_down_ms)
            if elapsed >= self.REPEAT_DELAY_MS:
                if (not self.repeat_started or
                        safe_ticks_diff(now, self.last_repeat_ms) >= self.REPEAT_INTERVAL_MS):
                    self.repeat_started = True
                    self.last_repeat_ms = now
                    self._execute_button(self.button_down)

    def _release_touch(self):
        if not self.touching:
            return
        self.touching = False
        self.empty_since = 0
        action = self.button_down
        self.button_down = None
        self.repeat_started = False

        if action is not None:
            # 方向键已经在按下阶段执行，抬手不重复执行。
            if action in self.MOVE_ACTIONS:
                return
            if self._button_at(self.last_x, self.last_y) == action:
                self._execute_button(action)
            return

        # 画面点击：允许在整个画面点击，只要没有落在按钮上。
        # 第一次写点1，第二次写点2；已有两点后点击哪个点附近就重定位哪个点，
        # 否则默认重定位当前选中的点，避免“第三次点击直接把两点全部清空”。
        if self._button_at(self.last_x, self.last_y) is not None:
            return

        px, py = int(self.last_x), int(self.last_y)
        if len(self.points) == 0:
            self.points.append((px, py))
            self.selected = 0
            self.app.notice("已粗定位点1，可用方向键精调", COLOR_YELLOW, 900)
        elif len(self.points) == 1:
            self.points.append((px, py))
            self.selected = 1
            self.app.notice("已粗定位点2，可用方向键精调", COLOR_GREEN, 900)
        else:
            # 若点击距离某个已有点较近，优先修改该点；否则修改当前选中点。
            d0 = (px - self.points[0][0]) ** 2 + (py - self.points[0][1]) ** 2
            d1 = (px - self.points[1][0]) ** 2 + (py - self.points[1][1]) ** 2
            near_gate2 = 70 * 70
            if min(d0, d1) <= near_gate2:
                idx = 0 if d0 <= d1 else 1
            else:
                idx = int(clamp(self.selected, 0, 1))
            self.points[idx] = (px, py)
            self.selected = idx
            self.app.notice("已重新粗定位点%d" % (idx + 1), COLOR_CYAN, 700)
        self.draw_editor()

    def handle_empty(self):
        if not self.active or not self.touching:
            self.empty_since = 0
            return
        now = time.ticks_ms()
        if self.empty_since == 0:
            self.empty_since = now
            return
        if safe_ticks_diff(now, self.empty_since) >= self.RELEASE_GAP_MS:
            self._release_touch()

class CameraControl:
    """
    相机方向控制。

    重要：
    K230 的显示视频层来自 Sensor -> Display.bind_layer() 的持续绑定链路。
    不能只在运行中做 sensor.stop() -> setter -> sensor.run()：
    某些固件上 Sensor 虽重新 run，但原 VIDEO 层绑定不会可靠恢复，表现为
    “OSD/触摸仍正常，但摄像头视频层永久黑屏”。

    因此方向变化时按官方生命周期完整重建视频链路：
        sensor.stop()
        -> Display.deinit()
        -> MediaManager.deinit()
        -> 重新创建/配置 Sensor（此时设置 mirror/flip）
        -> Display.bind_layer()
        -> Display.init()
        -> MediaManager.init()
        -> sensor.run()
        -> 重建 OSD 图像并恢复 UI

    方向切换会短暂黑屏，这是链路重建过程；完成后应自动恢复。
    """
    def __init__(self, app):
        self.app = app
        self.last_error = ""
        self.last_apply_ms = 0
        self.applied_state = None

    def mark_current_as_applied(self):
        cfg = self.app.cfg.get("camera", {})
        self.applied_state = (
            bool(cfg.get("hmirror", False)),
            bool(cfg.get("vflip", False))
        )

    def direction_name(self):
        cfg = self.app.cfg.get("camera", {})
        hm = bool(cfg.get("hmirror", False))
        vf = bool(cfg.get("vflip", False))
        if hm and vf:
            return "旋转180度"
        if hm:
            return "左右镜像"
        if vf:
            return "上下翻转"
        return "正常方向"

    def apply(self, restart=True, force=False):
        if self.app.sensor is None:
            self.last_error = "Sensor尚未初始化"
            return False

        cfg = self.app.cfg.get("camera", {})
        desired = (
            bool(cfg.get("hmirror", False)),
            bool(cfg.get("vflip", False))
        )

        if restart and (not force) and self.applied_state == desired:
            self.last_error = ""
            return True

        try:
            if restart:
                ok = self.app.restart_camera_pipeline()
            else:
                # 只允许在 Sensor 尚未开始出流的初始化阶段使用直接 setter。
                self.app.sensor.set_hmirror(desired[0])
                self.app.sensor.set_vflip(desired[1])
                ok = True

            if not ok:
                if not self.last_error:
                    self.last_error = "视频链路重建失败"
                return False

            self.applied_state = desired
            self.last_apply_ms = time.ticks_ms()
            self.last_error = ""
            print("[相机方向] 已应用:", self.direction_name())
            return True

        except Exception as e:
            self.last_error = str(e)
            print("[相机方向] 应用失败:", e)
            return False


class SceneManager:
    """
    四个场景槽。

    每个场景保存两部分：
    1) 当前整个平台配置：颜色LAB、YOLO模型与阈值、ROI、串口、测量、几何、模板等；
    2) entry_page：加载场景后应该自动进入哪个识别功能。

    因此场景才真正等于“比赛题目”，而不是只有一份看不出用途的参数快照。
    """
    def __init__(self, app):
        self.app = app

    def path(self, slot):
        return SCENE_PATH_FMT % (int(slot) + 1)

    def exists(self, slot):
        try:
            with open(self.path(slot), "r"):
                return True
        except Exception:
            return False

    def _valid_page(self, page):
        p = str(page)
        return p if p in SCENE_PAGE_OPTIONS else "color"

    def saved_entry_page(self, slot):
        """只读取场景绑定功能，用于场景页状态显示；读取失败返回空字符串。"""
        try:
            with open(self.path(slot), "r") as f:
                data = ujson.loads(f.read())
            scene = data.get("scene", {}) if isinstance(data, dict) else {}
            if not isinstance(scene, dict):
                return ""
            return self._valid_page(scene.get("entry_page", "color"))
        except Exception:
            return ""

    def save(self, slot, entry_page=None):
        slot = int(clamp(int(slot), 0, SCENE_SLOT_COUNT - 1))
        try:
            remember_yolo_runtime(self.app.cfg["yolo"])

            if entry_page is None:
                entry_page = self.app.cfg.get("scene", {}).get("entry_page", "color")
            entry_page = self._valid_page(entry_page)

            # 同步回当前配置，这样用户在场景页看到的绑定功能与实际保存一致。
            self.app.cfg.setdefault("scene", {})["selected"] = slot
            self.app.cfg["scene"]["entry_page"] = entry_page

            data = ujson.loads(ujson.dumps(self.app.cfg))
            data["scene"] = {
                "selected": slot,
                "entry_page": entry_page,
            }
            path = self.path(slot)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(ujson.dumps(data))
            try:
                os.remove(path)
            except Exception:
                pass
            os.rename(tmp, path)
            try:
                os.sync()
            except Exception:
                pass
            print("[场景] 已保存 场景%d -> %s" % (slot + 1, SCENE_PAGE_NAMES.get(entry_page, entry_page)))
            return True
        except Exception as e:
            print("[场景保存失败]", e)
            return False

    def load_config(self, slot):
        slot = int(clamp(int(slot), 0, SCENE_SLOT_COUNT - 1))
        try:
            with open(self.path(slot), "r") as f:
                loaded = ujson.loads(f.read())

            loaded_scene = loaded.get("scene", {}) if isinstance(loaded, dict) else {}
            if not isinstance(loaded_scene, dict):
                loaded_scene = {}
            entry_page = self._valid_page(loaded_scene.get("entry_page", "color"))

            cfg = default_config()
            deep_update(cfg, loaded)
            cfg["scene"] = {
                "selected": slot,
                "entry_page": entry_page,
            }
            return self.app.store._normalize(cfg)
        except Exception as e:
            print("[场景读取失败]", e)
            return None

    def delete(self, slot):
        try:
            os.remove(self.path(slot))
            return True
        except Exception:
            return False


class SerialManager:
    """三路 UART 的统一初始化、协议编码、目标选择和周期发送。"""
    PROTOCOL_NAMES = {
        "binary": "二进制",
        "ascii_hex": "ASCII十六进制",
        "string": "字符串",
    }
    SOURCE_NAMES = {
        "current": "当前功能",
        "yolo": "YOLO识别",
        "color": "颜色识别",
        "shape": "形状识别",
        "line": "巡线导航",
        "code": "码类识别",
        "geometry": "几何视觉",
        "template": "模板识别",
    }
    TARGET_MODE_NAMES = {"single": "单目标", "multi": "多目标"}
    SINGLE_RULE_NAMES = {"center": "中心最近", "score": "置信最高", "area": "面积最大"}
    COORD_MODE_NAMES = {
        "absolute": "原始坐标",
        "screen_center": "相对画面中心",
        "region_center": "相对区域中心",
        "custom": "相对自定义点",
    }

    def __init__(self, app):
        self.app = app
        self.fpioa = None
        self.uarts = [None for _ in SERIAL_PORT_PROFILES]
        self.last_cfg_key = [None for _ in SERIAL_PORT_PROFILES]
        self.last_tx_ms = [0 for _ in SERIAL_PORT_PROFILES]
        self.last_open_attempt_ms = [0 for _ in SERIAL_PORT_PROFILES]
        self.tx_count = [0 for _ in SERIAL_PORT_PROFILES]
        self.tx_bytes = [0 for _ in SERIAL_PORT_PROFILES]
        self.rx_count = [0 for _ in SERIAL_PORT_PROFILES]
        self.rx_bytes = [0 for _ in SERIAL_PORT_PROFILES]
        self.error_count = [0 for _ in SERIAL_PORT_PROFILES]
        self.last_error = ["" for _ in SERIAL_PORT_PROFILES]
        self.last_preview = ["" for _ in SERIAL_PORT_PROFILES]
        self.last_rx = [b"" for _ in SERIAL_PORT_PROFILES]
        self.last_command = ["" for _ in SERIAL_PORT_PROFILES]
        self.rx_buffers = [bytearray() for _ in SERIAL_PORT_PROFILES]
        self.last_rx_data_ms = [0 for _ in SERIAL_PORT_PROFILES]
        self.last_ui_refresh_ms = 0

    def _uart_const(self, no):
        if int(no) == 1:
            return UART.UART1
        if int(no) == 2:
            return UART.UART2
        if int(no) == 4:
            return UART.UART4
        raise ValueError("平台串口只使用 UART1 / UART2 / UART4")

    def _funcs(self, no):
        if int(no) == 1:
            return FPIOA.UART1_TXD, FPIOA.UART1_RXD
        if int(no) == 2:
            return FPIOA.UART2_TXD, FPIOA.UART2_RXD
        if int(no) == 4:
            return FPIOA.UART4_TXD, FPIOA.UART4_RXD
        raise ValueError("不支持的 UART 编号")

    def _parity_const(self, value):
        if value == "odd":
            return UART.PARITY_ODD
        if value == "even":
            return UART.PARITY_EVEN
        return UART.PARITY_NONE

    def _stop_const(self, value):
        return UART.STOPBITS_TWO if int(value) == 2 else UART.STOPBITS_ONE

    def _bits_const(self, value):
        v = int(value)
        if v == 5:
            return UART.FIVEBITS
        if v == 6:
            return UART.SIXBITS
        if v == 7:
            return UART.SEVENBITS
        return UART.EIGHTBITS

    def _cfg_key(self, index):
        port = self.app.cfg["serial"]["ports"][index]
        profile = SERIAL_PORT_PROFILES[index]
        return (bool(port.get("enabled", False)), int(port.get("baudrate", 115200)),
                int(port.get("bits", 8)), str(port.get("parity", "none")), int(port.get("stop", 1)),
                int(profile.get("uart_no", 0)), int(profile.get("tx_pin", -1)), int(profile.get("rx_pin", -1)))

    def close_port(self, index):
        uart = self.uarts[index]
        if uart is not None:
            try:
                uart.deinit()
            except Exception:
                pass
        self.uarts[index] = None
        self.last_cfg_key[index] = None

    def open_port(self, index, force=False):
        port = self.app.cfg["serial"]["ports"][index]
        key = self._cfg_key(index)
        if not port.get("enabled", False):
            self.close_port(index)
            return False
        if not force and self.uarts[index] is not None and self.last_cfg_key[index] == key:
            return True
        self.close_port(index)
        profile = SERIAL_PORT_PROFILES[index]
        try:
            if self.fpioa is None:
                self.fpioa = FPIOA()
            tx_func, rx_func = self._funcs(profile["uart_no"])
            self.fpioa.set_function(int(profile["tx_pin"]), tx_func)
            self.fpioa.set_function(int(profile["rx_pin"]), rx_func)
            uart = UART(
                self._uart_const(profile["uart_no"]),
                baudrate=int(port.get("baudrate", SERIAL_DEFAULT_BAUD)),
                bits=self._bits_const(port.get("bits", 8)),
                parity=self._parity_const(port.get("parity", "none")),
                stop=self._stop_const(port.get("stop", 1)),
                timeout=0,
            )
            self.uarts[index] = uart
            self.last_cfg_key[index] = key
            self.last_error[index] = ""
            print("[串口] %s 已开启 UART%d GPIO%d/GPIO%d %d" %
                  (profile["name"], profile["uart_no"], profile["tx_pin"], profile["rx_pin"], port["baudrate"]))
            return True
        except Exception as e:
            self.error_count[index] += 1
            self.last_error[index] = str(e)
            self.uarts[index] = None
            print("[串口] %s 初始化失败: %s" % (profile["name"], str(e)))
            return False

    def apply_selected(self):
        idx = int(self.app.cfg["serial"].get("selected", 0))
        return self.open_port(idx, force=True)

    def open_enabled(self):
        for i in range(len(SERIAL_PORT_PROFILES)):
            if self.app.cfg["serial"]["ports"][i].get("enabled", False):
                self.open_port(i, force=True)

    def deinit(self):
        for i in range(len(self.uarts)):
            self.close_port(i)
        self.fpioa = None

    def _source_for_port(self, port):
        source = port.get("source", "current")
        if source == "current":
            page = self.app.page
            if page in ("yolo", "color", "shape", "line", "code", "geometry", "template"):
                return page
            return "none"
        return source

    def _reference_point(self, port, source):
        mode = port.get("coord_mode", "absolute")
        if mode == "screen_center":
            return DISPLAY_W // 2, DISPLAY_H // 2
        if mode == "region_center":
            region_key = {"template": "template_search"}.get(source, source)
            if region_key in ("yolo", "color", "shape", "code", "geometry", "template_search"):
                return self.app.region_manager.center(region_key)
            return DISPLAY_W // 2, DISPLAY_H // 2
        if mode == "custom":
            return int(port.get("custom_x", DISPLAY_W // 2)), int(port.get("custom_y", DISPLAY_H // 2))
        return 0, 0

    def _signed16(self, value):
        v = int(round(value))
        if v < -32768:
            v = -32768
        if v > 32767:
            v = 32767
        return v & 0xFFFF

    def _u16(self, value):
        return int(clamp(int(round(value)), 0, 65535))

    def _target_xy(self, t, port, source):
        x = int(round(t.get("cx", 0)))
        y = int(round(t.get("cy", 0)))
        refx, refy = self._reference_point(port, source)
        dx = x - refx
        dy = y - refy
        return x, y, dx, dy

    def _select_targets(self, targets, port, source):
        if not targets:
            return []
        max_targets = int(clamp(port.get("max_targets", SERIAL_MAX_TARGETS), 1, SERIAL_MAX_TARGETS))
        items = list(targets)
        if port.get("target_mode", "single") == "multi":
            return items[:max_targets]

        rule = port.get("single_rule", "center")
        best = items[0]
        if rule == "score":
            best_value = float(best.get("score", 0.0))
            for t in items[1:]:
                v = float(t.get("score", 0.0))
                if v > best_value:
                    best, best_value = t, v
        elif rule == "area":
            best_value = float(best.get("area", 0.0))
            for t in items[1:]:
                v = float(t.get("area", 0.0))
                if v > best_value:
                    best, best_value = t, v
        else:
            refx, refy = self._reference_point(port, source)
            if port.get("coord_mode", "absolute") == "absolute":
                refx, refy = DISPLAY_W // 2, DISPLAY_H // 2
            bx = float(best.get("cx", 0)) - refx
            by = float(best.get("cy", 0)) - refy
            best_value = bx * bx + by * by
            for t in items[1:]:
                dx = float(t.get("cx", 0)) - refx
                dy = float(t.get("cy", 0)) - refy
                v = dx * dx + dy * dy
                if v < best_value:
                    best, best_value = t, v
        return [best]

    def _checksum_xor(self, data):
        c = 0
        for b in data:
            c ^= int(b) & 0xFF
        return c & 0xFF

    def _encode_binary(self, source, targets, port):
        # 协议1：55 01 源 数量 [目标16字节...] XOR 66
        data = bytearray()
        data.append(0x01)
        data.append(self.app.results.source_id(source) & 0xFF)
        data.append(len(targets) & 0xFF)
        for i, t in enumerate(targets):
            x, y, dx, dy = self._target_xy(t, port, source)
            cls = int(clamp(int(t.get("class_id", 0)), 0, 255))
            w = self._u16(t.get("w", 0))
            h = self._u16(t.get("h", 0))
            q = int(clamp(int(round(float(t.get("score", 0.0)) * 100.0)), 0, 100))
            ux, uy = self._u16(x), self._u16(y)
            udx, udy = self._signed16(dx), self._signed16(dy)
            data.extend(bytearray([
                i & 0xFF, cls & 0xFF,
                (ux >> 8) & 0xFF, ux & 0xFF,
                (uy >> 8) & 0xFF, uy & 0xFF,
                (udx >> 8) & 0xFF, udx & 0xFF,
                (udy >> 8) & 0xFF, udy & 0xFF,
                (w >> 8) & 0xFF, w & 0xFF,
                (h >> 8) & 0xFF, h & 0xFF,
                q & 0xFF, 0x01,
            ]))
        frame = bytearray([SERIAL_FRAME_HEAD])
        frame.extend(data)
        frame.append(self._checksum_xor(data))
        frame.append(SERIAL_FRAME_TAIL)
        return frame

    def _hex16(self, value, signed=False):
        v = self._signed16(value) if signed else self._u16(value)
        return "%04X" % v

    def _encode_ascii_hex(self, source, targets, port):
        # 协议2：真实帧头/帧尾 + 中间 ASCII 十六进制数字。
        parts = ["P=02", "S=%02X" % self.app.results.source_id(source), "N=%02X" % len(targets)]
        for i, t in enumerate(targets):
            x, y, dx, dy = self._target_xy(t, port, source)
            q = int(clamp(int(round(float(t.get("score", 0.0)) * 100.0)), 0, 100))
            part = ("I=%02X,C=%02X,X=%s,Y=%s,DX=%s,DY=%s,W=%s,H=%s,Q=%02X" %
                    (i & 0xFF, int(t.get("class_id", 0)) & 0xFF,
                     self._hex16(x), self._hex16(y), self._hex16(dx, True), self._hex16(dy, True),
                     self._hex16(t.get("w", 0)), self._hex16(t.get("h", 0)), q))
            parts.append(part)
        body = ";".join(parts)
        frame = bytearray([SERIAL_FRAME_HEAD])
        frame.extend(body.encode("ascii"))
        frame.append(SERIAL_FRAME_TAIL)
        return frame

    def _safe_ascii(self, text):
        try:
            raw = str(text).encode("ascii")
            return raw.decode("ascii")
        except Exception:
            out = []
            for ch in str(text):
                oc = ord(ch)
                out.append(ch if 32 <= oc <= 126 else "?")
            return "".join(out)

    def _encode_string(self, source, targets, port):
        # 协议3：真实帧头/帧尾 + 可读字符串，可发送字母/标签/数值。
        token = self._safe_ascii(port.get("string_token", "T"))
        parts = ["%s,%s,N=%d" % (token, source.upper(), len(targets))]
        for i, t in enumerate(targets):
            x, y, dx, dy = self._target_xy(t, port, source)
            label = self._safe_ascii(t.get("label", ""))
            parts.append("%d,%s,%d,%d,%d,%d,%.2f" %
                         (i, label, x, y, dx, dy, float(t.get("score", 0.0))))
        body = ";".join(parts)
        frame = bytearray([SERIAL_FRAME_HEAD])
        frame.extend(body.encode("ascii"))
        frame.append(SERIAL_FRAME_TAIL)
        return frame

    def encode(self, source, targets, port):
        protocol = port.get("protocol", "binary")
        if protocol == "ascii_hex":
            return self._encode_ascii_hex(source, targets, port)
        if protocol == "string":
            return self._encode_string(source, targets, port)
        return self._encode_binary(source, targets, port)

    def _preview(self, frame, protocol):
        if frame is None:
            return ""
        if protocol == "binary":
            vals = []
            for b in frame[:18]:
                vals.append("%02X" % int(b))
            if len(frame) > 18:
                vals.append("...")
            return " ".join(vals)
        try:
            body = bytes(frame[1:-1]).decode("ascii")
        except Exception:
            body = ""
        if len(body) > 30:
            body = body[:30] + "..."
        return "55 " + body + " 66"

    def _record_command(self, index, text):
        text = str(text).strip()
        if not text:
            return
        if len(text) > 32:
            text = text[:32]
        self.last_command[index] = text
        print("[串口命令] %s <- %s" % (SERIAL_PORT_PROFILES[index]["name"], text))
        try:
            self.app.handle_remote_ascii(index, text)
        except Exception as e:
            self.error_count[index] += 1
            self.last_error[index] = "命令执行:" + str(e)

    def _record_binary_command(self, index, cmd, param):
        self.last_command[index] = "BIN %02X %02X" % (cmd & 0xFF, param & 0xFF)
        try:
            self.app.handle_remote_binary(index, int(cmd) & 0xFF, int(param) & 0xFF)
        except Exception as e:
            self.error_count[index] += 1
            self.last_error[index] = "命令执行:" + str(e)

    def _process_frame_body(self, index, body):
        if not body:
            return
        # 二进制控制帧：55 10 CMD PARAM XOR 66
        if len(body) >= 4 and int(body[0]) == 0x10:
            payload = body[:-1]
            check = int(body[-1]) & 0xFF
            if self._checksum_xor(payload) == check:
                self._record_binary_command(index, int(body[1]), int(body[2]))
            else:
                self.last_error[index] = "控制帧校验失败"
            return
        # 否则把 55/66 中间内容按 ASCII 命令处理，例如 55 'M2' 66。
        try:
            text = bytes(body).decode("ascii").strip()
        except Exception:
            text = ""
        if text:
            self._record_command(index, text)

    def _consume_rx_buffer(self, index, flush_plain=False):
        buf = self.rx_buffers[index]
        # 优先解析 0x55 ... 0x66 帧；丢弃帧头前的换行/空白。
        while True:
            try:
                head = buf.index(SERIAL_FRAME_HEAD)
            except Exception:
                head = -1
            if head < 0:
                break
            if head > 0:
                prefix = bytes(buf[:head])
                del buf[:head]
                try:
                    text = prefix.decode("ascii").strip()
                except Exception:
                    text = ""
                if text:
                    self._record_command(index, text)
            tail = -1
            # 不依赖 bytearray.index(value, start) 的可选参数，兼容更多 MicroPython 固件。
            for pos in range(1, len(buf)):
                if int(buf[pos]) == SERIAL_FRAME_TAIL:
                    tail = pos
                    break
            if tail < 0:
                break
            body = bytes(buf[1:tail])
            del buf[:tail + 1]
            self._process_frame_body(index, body)

        # 无帧头时支持换行命令；还支持短命令 M2/Y/C 等不带换行，空闲40ms后提交。
        while True:
            pos_n = -1
            for sep in (10, 13):
                try:
                    p = buf.index(sep)
                    if pos_n < 0 or p < pos_n:
                        pos_n = p
                except Exception:
                    pass
            if pos_n < 0:
                break
            raw = bytes(buf[:pos_n])
            del buf[:pos_n + 1]
            try:
                text = raw.decode("ascii").strip()
            except Exception:
                text = ""
            if text:
                self._record_command(index, text)

        if flush_plain and buf and SERIAL_FRAME_HEAD not in buf:
            try:
                text = bytes(buf).decode("ascii").strip()
            except Exception:
                text = ""
            buf[:] = b""
            if text:
                self._record_command(index, text)

        # 防止异常数据长期堆积。
        if len(buf) > 192:
            del buf[:-96]

    def _poll_rx(self, index):
        uart = self.uarts[index]
        if uart is None:
            return
        try:
            data = uart.read()
            if data:
                self.rx_count[index] += 1
                self.rx_bytes[index] += len(data)
                self.last_rx[index] = data[-48:]
                self.last_rx_data_ms[index] = time.ticks_ms()
                if self.app.cfg["serial"]["ports"][index].get("rx_control", False):
                    self.rx_buffers[index].extend(data)
                    self._consume_rx_buffer(index, flush_plain=False)
            elif (self.app.cfg["serial"]["ports"][index].get("rx_control", False) and
                  self.rx_buffers[index] and
                  safe_ticks_diff(time.ticks_ms(), self.last_rx_data_ms[index]) >= 40):
                self._consume_rx_buffer(index, flush_plain=True)
        except Exception as e:
            self.error_count[index] += 1
            self.last_error[index] = str(e)

    def service(self):
        now = time.ticks_ms()
        for i in range(len(SERIAL_PORT_PROFILES)):
            port = self.app.cfg["serial"]["ports"][i]
            if not port.get("enabled", False):
                if self.uarts[i] is not None:
                    self.close_port(i)
                continue
            if self.uarts[i] is None or self.last_cfg_key[i] != self._cfg_key(i):
                # 初始化失败时不要每帧重试，避免串口接线/复用错误拖垮视觉帧率。
                if self.uarts[i] is None and safe_ticks_diff(now, self.last_open_attempt_ms[i]) < 1500:
                    continue
                self.last_open_attempt_ms[i] = now
                if not self.open_port(i):
                    continue
            self._poll_rx(i)
            # 接收命令可能刚刚加载了场景并重建了整套配置/关闭了串口。
            # 此时本轮不要继续拿旧 port 引用发送，下一轮主循环再按新配置工作。
            if self.uarts[i] is None or port is not self.app.cfg["serial"]["ports"][i]:
                continue

            hz = max(1, int(port.get("send_hz", 20)))
            interval = max(20, int(1000 / hz))
            if safe_ticks_diff(now, self.last_tx_ms[i]) < interval:
                continue
            source = self._source_for_port(port)
            result = None if source == "none" else self.app.results.get(source, SERIAL_RESULT_FRESH_MS)
            raw_targets = [] if result is None else result.get("targets", [])
            targets = self._select_targets(raw_targets, port, source)
            if not targets and not port.get("send_empty", True):
                self.last_tx_ms[i] = now
                continue
            try:
                frame = self.encode(source, targets, port)
                written = self.uarts[i].write(frame)
                self.last_tx_ms[i] = now
                self.tx_count[i] += 1
                if written is None:
                    written = len(frame)
                self.tx_bytes[i] += int(written)
                self.last_preview[i] = self._preview(frame, port.get("protocol", "binary"))
            except Exception as e:
                self.error_count[i] += 1
                self.last_error[i] = str(e)
                print("[串口发送异常] %s: %s" % (SERIAL_PORT_PROFILES[i]["name"], str(e)))

        if self.app.page == "serial" and self.app.ui is not None:
            if safe_ticks_diff(now, self.last_ui_refresh_ms) >= 500:
                self.last_ui_refresh_ms = now
                self.app.ui.invalidate()

# ============================================================
# 10. UI Manager - 右侧小面板 + 首页
# ============================================================

class UIManager:
    def __init__(self, app, overlay_img, panel_img):
        self.app = app
        self.overlay = overlay_img
        self.panel = panel_img
        self.pool = ButtonPool(88)
        self.dirty = True

    def invalidate(self):
        self.dirty = True

    def hit_snapshot(self, x, y):
        return self.pool.hit_snapshot(x, y)

    def _ptext(self, x, y, size, text, color=COLOR_WHITE):
        self.panel.draw_string_advanced(x, y, size, text, color=color)

    def _panel_bg(self):
        self.panel.clear()
        self.panel.draw_rectangle(0, 0, PANEL_W, PANEL_H, color=COLOR_DARK, fill=True)
        self.panel.draw_line(0, 0, 0, PANEL_H - 1, color=COLOR_BLUE, thickness=2)

    def _pbtn(self, x, y, w, h, text, action, payload=None,
              mode=PRESS_TAP, bg=COLOR_DARK2, border=COLOR_GRAY,
              fg=COLOR_WHITE, hold_ms=TOUCH_HOLD_MS):
        return self.pool.add(PANEL_X + x, y, w, h, text, action, payload,
                             press_mode=mode, bg=bg, fg=fg,
                             border=border, hold_ms=hold_ms)

    def _hbtn(self, x, y, w, h, text, action, payload=None,
              mode=PRESS_TAP, bg=COLOR_DARK2, border=COLOR_GRAY,
              fg=COLOR_WHITE, hold_ms=TOUCH_HOLD_MS):
        return self.pool.add(x, y, w, h, text, action, payload,
                             press_mode=mode, bg=bg, fg=fg,
                             border=border, hold_ms=hold_ms)

    def _header(self, title, region_key=None):
        self._pbtn(8, 7, 58, 38, "返回", "nav", "home",
                   bg=(45, 62, 82), border=COLOR_BLUE)
        show_title = str(title)
        if region_key is not None and len(show_title) > 8:
            show_title = show_title[:8]
        self._ptext(74, 15, 17 if region_key is not None else 19, show_title, COLOR_WHITE)
        if region_key is not None:
            active = self.app.region_manager.enabled(region_key) if self.app.region_manager is not None else False
            self._pbtn(180, 7, 52, 38, "区域", "region_edit", region_key,
                       bg=self._active_bg(active), border=COLOR_CYAN if active else COLOR_BLUE)
        self.panel.draw_line(6, 49, PANEL_W - 7, 49, color=COLOR_GRAY, thickness=1)

    def _active_bg(self, active):
        return (45, 92, 132) if active else COLOR_DARK2

    def _active_border(self, active):
        return COLOR_CYAN if active else COLOR_GRAY

    def _lab_rows(self, lab, trained, y0=120):
        if not trained:
            self._ptext(12, y0, 20, "L     --  ~  --", COLOR_LIGHT)
            self._ptext(12, y0 + 28, 20, "A     --  ~  --", COLOR_LIGHT)
            self._ptext(12, y0 + 56, 20, "B     --  ~  --", COLOR_LIGHT)
            return
        self._ptext(12, y0, 20, "L  %4d ~ %4d" % (lab[0], lab[1]), COLOR_LIGHT)
        self._ptext(12, y0 + 28, 20, "A  %4d ~ %4d" % (lab[2], lab[3]), COLOR_LIGHT)
        self._ptext(12, y0 + 56, 20, "B  %4d ~ %4d" % (lab[4], lab[5]), COLOR_LIGHT)

    # --------------------------------------------------------
    # Home
    # --------------------------------------------------------
    def _home(self):
        # 首页使用完整 800x480 不透明 OSD，不露出摄像头画面。
        self.panel.clear()
        self.overlay.clear()
        self.overlay.draw_rectangle(0, 0, DISPLAY_W, DISPLAY_H, color=(9, 14, 22), fill=True)

        # 顶部标题区
        self.overlay.draw_rectangle(0, 0, DISPLAY_W, 84, color=(14, 22, 34), fill=True)
        self.overlay.draw_rectangle(22, 18, 7, 48, color=(54, 174, 255), fill=True)
        self.overlay.draw_string_advanced(42, 16, 28, "K230 万能视觉平台", color=COLOR_WHITE)
        self.overlay.draw_string_advanced(43, 51, 13, "识别 · 测量 · 场景 · 三串口", color=(150, 171, 192))
        self.overlay.draw_line(20, 83, DISPLAY_W - 20, 83, color=(42, 58, 77), thickness=1)

        # 重新布局主页：3列×4行，标题和副标题手动绘制，避免和按钮自带文字重叠。
        card_w = 226
        card_h = 68
        gap_x = 24
        gap_y = 14
        x0 = 28
        y0 = 98
        cards = [
            (0, 0, "形状识别",   "shape",    (75,157,255),  "三角/矩形/圆形/激光点"),
            (1, 0, "多目标颜色", "color",    (43,207,174),  "LAB训练 / 多目标"),
            (2, 0, "YOLO识别",   "yolo",     (246,173,76),  "v5 / v8 / 11 / 26"),

            (0, 1, "码类识别",   "code",     (170,126,244), "二维码 / 条码 / 标签"),
            (1, 1, "巡线导航",   "line",     (49,190,214),  "多区域 / 回归 / 八邻域"),
            (2, 1, "几何视觉",   "geometry", (70,181,238),  "直线 / 线段 / 圆 / 矩形"),

            (0, 2, "模板识别",   "template", (210,125,245), "框选保存 / NCC匹配"),
            (1, 2, "测量标定",   "measure",  (190,110,255), "坐标 / 距离 / 尺寸"),
            (2, 2, "场景模式",   "scene",    (92,210,125),  "题目配置一键保存恢复"),

            (0, 3, "串口通信",   "serial",   (240,170,75),  "UART1 / UART2 / UART4"),
            (1, 3, "相机设置",   "camera",   (98,178,220),  "方向 / 通道状态"),
            (2, 3, "系统设置",   "settings", (143,155,173), "显示 / 参数 / 调试"),
        ]

        card_bg = (23, 33, 46)
        title_fs = 21
        subtitle_fs = 11
        for col, row, title, page, border, subtitle in cards:
            x = x0 + col * (card_w + gap_x)
            y = y0 + row * (card_h + gap_y)
            # 仅创建可点击区域，不再使用按钮默认文本绘制。
            self._hbtn(x, y, card_w, card_h, "", "nav", page, bg=card_bg, border=border)

        self.pool.draw_all(self.overlay)

        for col, row, title, page, border, subtitle in cards:
            x = x0 + col * (card_w + gap_x)
            y = y0 + row * (card_h + gap_y)
            self.overlay.draw_rectangle(x + 1, y + 1, 6, card_h - 2, color=border, fill=True)
            self.overlay.draw_string_advanced(x + 16, y + 14, title_fs, title, color=COLOR_WHITE)
            self.overlay.draw_string_advanced(x + 16, y + 43, subtitle_fs, subtitle, color=(176, 192, 208))

        # 底部状态条
        self.overlay.draw_line(20, 430, DISPLAY_W - 20, 430, color=(42, 58, 77), thickness=1)
        ycfg = self.app.cfg["yolo"]
        home_model = str(ycfg.get("profile_name", "YOLO"))
        if len(home_model) > 10:
            home_model = home_model[:10]
        serial_on = sum(1 for p in self.app.cfg["serial"]["ports"] if p.get("enabled", False))
        self.overlay.draw_string_advanced(24, 446, 15,
                                          "当前模型：%s  %dx%d" % (home_model, ycfg["model_input_size"][0], ycfg["model_input_size"][1]),
                                          color=(246, 190, 96))
        self.overlay.draw_string_advanced(612, 446, 15, "串口：%d/3" % serial_on,
                                          color=COLOR_GREEN if serial_on else COLOR_LIGHT)

        # 首页只使用已经初始化的 OSD0/OSD1。
        # OSD0 显示全屏主页；OSD1 显式刷新为空面板，防止旧功能页残留。
        Display.show_image(self.panel, x=PANEL_X, y=0, layer=Display.LAYER_OSD1)
        Display.show_image(self.overlay, layer=Display.LAYER_OSD0)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------
    def _shape(self):
        self._panel_bg()
        self._header("形状识别", "shape")
        cfg = self.app.cfg["shape"]
        modes = [
            (8, 58, 72, 34, "自动", "auto"),
            (84, 58, 72, 34, "三角", "triangle"),
            (160, 58, 72, 34, "矩形", "rect"),
            (8, 96, 72, 34, "圆形", "circle"),
            (84, 96, 72, 34, "激光", "laser"),
        ]
        for x, y, w, h, text, value in modes:
            active = cfg["mode"] == value
            self._pbtn(x, y, w, h, text, "shape_mode", value,
                       bg=self._active_bg(active), border=self._active_border(active))

        params = self.app.shape_param_specs()
        idx = int(clamp(cfg.get("param_idx", 0), 0, len(params) - 1))
        key, label, step, lo, hi = params[idx]
        val = cfg[key]
        val_text = ("%.3f" % val) if isinstance(val, float) else str(val)
        self._ptext(10, 153, 18, "调参 %d/%d" % (idx + 1, len(params)), COLOR_YELLOW)
        self._ptext(10, 177, 20, "%s: %s" % (label, val_text), COLOR_WHITE)
        self._pbtn(8, 207, 106, 38, "上一项", "shape_param", -1)
        self._pbtn(124, 207, 106, 38, "下一项", "shape_param", 1)
        self._pbtn(8, 252, 106, 42, "减少", "shape_delta", -1, mode=PRESS_REPEAT)
        self._pbtn(124, 252, 106, 42, "增加", "shape_delta", 1, mode=PRESS_REPEAT)

        self._ptext(10, 305, 17, "连续确认: %d帧" % cfg["stable_frames"], COLOR_LIGHT)
        self._ptext(10, 329, 17, "OpenCV: %s" % ("可用" if CV2_OK else "不可用"),
                    COLOR_GREEN if CV2_OK else COLOR_RED)
        if cfg["mode"] == "laser":
            self._ptext(10, 353, 16, "亮度门限 + 斑点检测", COLOR_LIGHT)
            self._ptext(10, 375, 16, "只认最亮的小圆点", COLOR_LIGHT)
        else:
            self._ptext(10, 353, 16, "轮廓+凸性+角度+填充率", COLOR_LIGHT)
            self._ptext(10, 375, 16, "+圆度+连续帧联合判定", COLOR_LIGHT)
        self._pbtn(8, 405, 106, 42, "保存", "save_cfg",
                   bg=(35, 90, 58), border=COLOR_GREEN)
        self._pbtn(124, 405, 106, 42, ("再次点击复位" if self.app.confirm_pending("reset_shape") else "复位参数"), "reset_shape",
                   mode=PRESS_CONFIRM, bg=(95, 45, 45), border=COLOR_RED)

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------
    def _color(self):
        self._panel_bg()
        self._header("多目标颜色", "color")
        cfg = self.app.cfg["color"]
        idx = int(cfg["current"])
        p = cfg["profiles"][idx]

        # 使用绝对目标按钮，而不是“上一/下一”循环按钮。
        # 即使触摸控制器收到重复事件，重复写同一个 index 也不会从目标1跳到目标4。
        for i in range(4):
            x = 8 + i * 56
            active = (idx == i)
            trained = cfg["profiles"][i]["trained"]
            bg = self._active_bg(active)
            border = self._active_border(active)
            if trained and not active:
                border = PALETTE[i]
            self._pbtn(x, 58, 50, 38, str(i + 1), "color_profile_set", i,
                       bg=bg, border=border)

        self._ptext(10, 103, 20, "%s" % p["name"], PALETTE[idx])
        state = "已训练" if p["trained"] else "未训练 / 空目标"
        self._ptext(105, 105, 17, state, COLOR_GREEN if p["trained"] else COLOR_YELLOW)

        self._lab_rows(p["lab"], p["trained"], 130)
        names = ["L低", "L高", "A低", "A高", "B低", "B高"]
        pi = int(cfg.get("param_idx", 0))
        value = "--" if not p["trained"] else str(p["lab"][pi])
        self._ptext(10, 216, 18, "当前: %s = %s" % (names[pi], value), COLOR_YELLOW)
        self._pbtn(8, 241, 106, 35, "上一项", "color_param", -1)
        self._pbtn(124, 241, 106, 35, "下一项", "color_param", 1)
        self._pbtn(8, 282, 50, 36, "-5", "color_delta", -5, mode=PRESS_REPEAT)
        self._pbtn(64, 282, 50, 36, "-1", "color_delta", -1, mode=PRESS_REPEAT)
        self._pbtn(124, 282, 50, 36, "+1", "color_delta", 1, mode=PRESS_REPEAT)
        self._pbtn(180, 282, 50, 36, "+5", "color_delta", 5, mode=PRESS_REPEAT)

        en = p["enabled"] and p["trained"]
        au = p["auto"] and p["trained"]
        self._pbtn(8, 327, 106, 40, "检测:%s" % ("开" if en else "关"),
                   "color_toggle_enable", bg=self._active_bg(en), border=self._active_border(en))
        self._pbtn(124, 327, 106, 40, "自适应:%s" % ("开" if au else "关"),
                   "color_toggle_auto", bg=self._active_bg(au), border=self._active_border(au))

        training = self.app.color_engine.sample_armed or self.app.color_engine.trainer.active
        self._pbtn(8, 375, 106, 40, "训练取色" if not training else "等待取色...",
                   "color_train", bg=(80, 67, 35), border=COLOR_YELLOW)
        self._pbtn(124, 375, 106, 40, "保存参数", "save_cfg",
                   bg=(35, 90, 58), border=COLOR_GREEN)
        pending = self.app.color_clear_pending()
        clear_text = "再次点击确认清空" if pending else "清空当前目标"
        self._pbtn(8, 423, 222, 40, clear_text, "color_clear_current",
                   mode=PRESS_TAP,
                   bg=(115, 45, 45) if pending else (72, 45, 45),
                   border=COLOR_RED)

    # --------------------------------------------------------
    # Line following
    # --------------------------------------------------------
    def _line(self):
        self._panel_bg()
        self._header("三算法巡线")
        cfg = self.app.cfg["line"]
        modes = [(8, 58, 70, "多区域", "roi"),
                 (85, 58, 70, "回归", "regression"),
                 (162, 58, 70, "八邻域", "eight")]
        for x, y, w, label, value in modes:
            active = cfg["mode"] == value
            self._pbtn(x, y, w, 38, label, "line_mode", value,
                       bg=self._active_bg(active), border=self._active_border(active))

        self._ptext(10, 104, 18, "巡线颜色: " + ("已训练" if cfg["trained"] else "未训练"),
                    COLOR_GREEN if cfg["trained"] else COLOR_YELLOW)
        self._lab_rows(cfg["lab"], cfg["trained"], 128)

        training = self.app.line_engine.sample_armed or self.app.line_engine.trainer.active
        self._pbtn(8, 214, 106, 38, "训练线色" if not training else "采样中...", "line_train",
                   bg=(80, 67, 35), border=COLOR_YELLOW)
        self._pbtn(124, 214, 106, 38, ("再次点击清色" if self.app.confirm_pending("line_clear") else "清除线色"), "line_clear",
                   mode=PRESS_CONFIRM, bg=(95, 45, 45), border=COLOR_RED)

        self._pbtn(8, 258, 106, 36, "反色:%s" % ("开" if cfg["invert"] else "关"),
                   "line_toggle", "invert", bg=self._active_bg(cfg["invert"]))
        self._pbtn(124, 258, 106, 36, "鲁棒:%s" % ("开" if cfg["robust"] else "关"),
                   "line_toggle", "robust", bg=self._active_bg(cfg["robust"]))

        specs = self.app.line_param_specs()
        idx = int(clamp(cfg.get("param_idx", 0), 0, len(specs) - 1))
        key, label, step, lo, hi = specs[idx]
        value = cfg[key]
        val_text = ("%.2f" % value) if isinstance(value, float) else str(value)
        self._ptext(10, 303, 16, "调参 %d/%d  %s" % (idx + 1, len(specs), label), COLOR_YELLOW)
        self._ptext(10, 326, 18, "当前值: " + val_text, COLOR_WHITE)
        self._pbtn(8, 350, 70, 36, "上一", "line_param", -1)
        self._pbtn(85, 350, 70, 36, "下一", "line_param", 1)
        self._pbtn(162, 350, 32, 36, "-", "line_delta", -1, mode=PRESS_REPEAT)
        self._pbtn(200, 350, 32, 36, "+", "line_delta", 1, mode=PRESS_REPEAT)
        self._ptext(10, 392, 14, "训练时点线条中心；多区域模式适合弯道", COLOR_LIGHT)
        self._ptext(10, 413, 14, "回归有动态走廊；8N支持断线恢复", COLOR_LIGHT)
        self._pbtn(8, 438, 222, 32, "保存巡线参数", "save_cfg",
                   bg=(35, 90, 58), border=COLOR_GREEN)

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------
    def _yolo(self):
        self._panel_bg()
        self._header("YOLO识别", "yolo")
        cfg = self.app.cfg["yolo"]
        idx = int(cfg.get("model_index", 0))
        total = max(1, yolo_profile_count())
        name = str(cfg.get("profile_name", "模型%d" % (idx + 1)))
        family = normalize_yolo_family(cfg.get("family", "yolov8"))
        short_name = name if len(name) <= 11 else name[:11]

        self._ptext(10, 56, 18, "模型 %d/%d  %s" % (idx + 1, total, short_name), COLOR_CYAN)
        prev_idx = (idx - 1) % total
        next_idx = (idx + 1) % total
        # 按钮携带“绝对模型索引”；TouchController 不依赖 event，
        # 以“稳定无触点”作为一次点击结束，并在重建完成后才重新允许下一次切模。
        self._pbtn(8, 80, 106, 34, "上个模型", "yolo_model_set", prev_idx,
                   mode=PRESS_TAP, bg=(45, 72, 95), border=COLOR_BLUE)
        self._pbtn(124, 80, 106, 34, "下个模型", "yolo_model_set", next_idx,
                   mode=PRESS_TAP, bg=(45, 72, 95), border=COLOR_BLUE)

        self._ptext(10, 119, 15, "%s  输入:%dx%d" % (family.upper(), cfg["model_input_size"][0], cfg["model_input_size"][1]), COLOR_LIGHT)
        self._ptext(10, 140, 16, "CONF %.2f" % cfg["conf"], COLOR_WHITE)
        self._pbtn(8, 160, 106, 32, "CONF -", "yolo_conf", -0.05, mode=PRESS_REPEAT)
        self._pbtn(124, 160, 106, 32, "CONF +", "yolo_conf", 0.05, mode=PRESS_REPEAT)

        if family == "yolo26":
            self._ptext(10, 198, 15, "NMS: YOLO26 detect不传", COLOR_LIGHT)
        else:
            self._ptext(10, 198, 16, "NMS %.2f" % cfg["nms"], COLOR_WHITE)
            self._pbtn(8, 218, 106, 32, "NMS -", "yolo_nms", -0.05, mode=PRESS_REPEAT)
            self._pbtn(124, 218, 106, 32, "NMS +", "yolo_nms", 0.05, mode=PRESS_REPEAT)

        toggles = [
            (8, 258, "检测框", "show_box"),
            (124, 258, "标签", "show_label"),
            (8, 296, "置信度", "show_conf"),
            (124, 296, "中心点", "show_center"),
        ]
        for x, y, label, key in toggles:
            active = bool(cfg.get(key, False))
            self._pbtn(x, y, 106, 32, "%s:%s" % (label, "开" if active else "关"),
                       "yolo_toggle", key, bg=self._active_bg(active), border=self._active_border(active))

        eng = self.app.yolo_engine
        if eng.ready:
            state = "已加载: " + eng.loaded_model_name
            state_color = COLOR_GREEN
        elif eng.manual_unloaded:
            state = "已卸载"
            state_color = COLOR_YELLOW
        elif eng.last_error:
            state = "失败: " + eng.last_error
            state_color = COLOR_RED
        else:
            state = "未加载"
            state_color = COLOR_YELLOW
        if eng.needs_rebuild:
            state += " / 待重载"
        if len(state) > 25:
            state = state[:25]
        self._ptext(10, 335, 15, "状态: " + state, state_color)

        labels = cfg.get("labels", ["object"])
        label_text = "/".join(labels[:3])
        if len(labels) > 3:
            label_text += "..."
        if len(label_text) > 23:
            label_text = label_text[:23]
        self._ptext(10, 357, 14, "标签: " + label_text, COLOR_LIGHT)
        path_name = str(cfg.get("model_path", "")).split("/")[-1]
        if len(path_name) > 24:
            path_name = path_name[:24]
        self._ptext(10, 377, 13, "文件: " + path_name, COLOR_LIGHT)

        self._pbtn(8, 398, 222, 34, "加载 / 重载当前模型", "yolo_apply",
                   bg=(78, 55, 45), border=COLOR_ORANGE)
        self._pbtn(8, 438, 106, 32, "保存参数", "save_cfg",
                   bg=(35, 90, 58), border=COLOR_GREEN)
        self._pbtn(124, 438, 106, 32, ("再次点击卸载" if self.app.confirm_pending("yolo_unload") else "卸载模型"), "yolo_unload",
                   mode=PRESS_CONFIRM, bg=(95, 45, 45), border=COLOR_RED)

    # --------------------------------------------------------
    # Code
    # --------------------------------------------------------
    def _code(self):
        self._panel_bg()
        self._header("码类识别", "code")
        cfg = self.app.cfg["code"]
        modes = [
            (8, 60, "QR二维码", "qr"),
            (124, 60, "DataMatrix", "dm"),
            (8, 108, "条形码", "barcode"),
            (124, 108, "AprilTag", "apriltag"),
        ]
        for x, y, label, value in modes:
            active = cfg["mode"] == value
            self._pbtn(
                x, y, 106, 40, label, "code_mode", value,
                bg=self._active_bg(active),
                border=self._active_border(active)
            )

        if cfg["mode"] == "apriltag":
            fam = str(cfg.get("apriltag_family", "all"))
            fam_name = APRILTAG_FAMILY_UI_NAMES.get(fam, "全部6类")
            self._ptext(10, 164, 17, "机器码家族: " + fam_name, COLOR_CYAN)
            self._pbtn(
                8, 194, 222, 40,
                "切换家族: " + fam_name,
                "apriltag_family_cycle",
                None,
                bg=(42, 52, 64),
                border=COLOR_BLUE
            )
            if fam == "all":
                self._ptext(10, 242, 14, "兼容6类Tag，速度会稍低", COLOR_YELLOW)
            else:
                self._ptext(10, 242, 14, "单一家族：识别速度更高", COLOR_LIGHT)
            self._ptext(10, 268, 14, "标签显示：家族 + ID", COLOR_LIGHT)

        elif cfg["mode"] == "barcode":
            # 一维条码原生灰度固定开启，不再画可点击开关。
            # 避免这块3.1寸屏一次物理点击被拆成两次 Toggle。
            native_ok = bool(self.app.barcode_native_active)

            self._ptext(
                10, 164, 17,
                "原生灰度: %s" % ("已启用" if native_ok else "正在切换"),
                COLOR_GREEN if native_ok else COLOR_YELLOW
            )

            # 这里只画状态框，不注册触摸按钮。
            self.panel.draw_rectangle(
                8, 194, 222, 40,
                color=(30, 52, 43),
                fill=True
            )
            self.panel.draw_rectangle(
                8, 194, 222, 40,
                color=COLOR_GREEN,
                thickness=2
            )
            self._ptext(20, 204, 17, "固定 640x480 GRAYSCALE", COLOR_WHITE)

            self._ptext(10, 242, 14, "优先原图；失败自动兼容增强", COLOR_YELLOW)
            self._ptext(10, 266, 14, "支持固定条带 / Otsu / 反相", COLOR_LIGHT)
            self._ptext(10, 290, 13, "小条码建议用“区域”固定框选", COLOR_LIGHT)

        elif cfg["mode"] == "dm":
            self._ptext(10, 168, 18, "识别强度: %d" % cfg["dm_effort"], COLOR_YELLOW)
            self._pbtn(8, 204, 106, 42, "-10", "dm_effort", -10, mode=PRESS_REPEAT)
            self._pbtn(124, 204, 106, 42, "+10", "dm_effort", 10, mode=PRESS_REPEAT)
            self._ptext(10, 258, 14, "DataMatrix使用灰度工作图", COLOR_LIGHT)

        else:
            self._ptext(10, 170, 15, "当前只运行一种解码算法", COLOR_LIGHT)
            self._ptext(10, 196, 15, "码类使用灰度工作图", COLOR_LIGHT)

        self._pbtn(
            8, 405, 222, 42,
            "保存参数", "save_cfg",
            bg=(35, 90, 58),
            border=COLOR_GREEN
        )

    # --------------------------------------------------------
    # Serial communication
    # --------------------------------------------------------
    def _serial(self):
        self._panel_bg()
        self._header("串口通信")
        scfg = self.app.cfg["serial"]
        idx = int(clamp(int(scfg.get("selected", 0)), 0, len(SERIAL_PORT_PROFILES) - 1))
        port = scfg["ports"][idx]
        profile = SERIAL_PORT_PROFILES[idx]
        mgr = self.app.serial_manager

        for i in range(len(SERIAL_PORT_PROFILES)):
            x = 8 + i * 76
            active = (idx == i)
            on = bool(scfg["ports"][i].get("enabled", False))
            border = COLOR_GREEN if on else self._active_border(active)
            self._pbtn(x, 58, 68, 34, str(i + 1), "serial_select", i,
                       bg=self._active_bg(active), border=border)

        self._ptext(10, 99, 15,
                    "UART%d  TX:%d RX:%d" % (profile["uart_no"], profile["tx_pin"], profile["rx_pin"]),
                    COLOR_LIGHT)
        self._pbtn(8, 121, 222, 34, "串口:%s" % ("开" if port.get("enabled", False) else "关"),
                   "serial_toggle_enable", None,
                   bg=self._active_bg(port.get("enabled", False)),
                   border=COLOR_GREEN if port.get("enabled", False) else COLOR_GRAY)

        self._pbtn(8, 161, 222, 32, "波特率: %d" % int(port.get("baudrate", 115200)),
                   "serial_cycle", "baudrate", bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(8, 199, 222, 32,
                   "协议: " + mgr.PROTOCOL_NAMES.get(port.get("protocol", "binary"), "二进制"),
                   "serial_cycle", "protocol", bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(8, 237, 222, 32,
                   "来源: " + mgr.SOURCE_NAMES.get(port.get("source", "current"), "当前功能"),
                   "serial_cycle", "source", bg=(42, 52, 64), border=COLOR_BLUE)

        mode_text = mgr.TARGET_MODE_NAMES.get(port.get("target_mode", "single"), "单目标")
        self._pbtn(8, 275, 106, 32, mode_text, "serial_cycle", "target_mode",
                   bg=self._active_bg(port.get("target_mode", "single") == "multi"), border=COLOR_CYAN)
        rule_text = mgr.SINGLE_RULE_NAMES.get(port.get("single_rule", "center"), "中心最近")
        self._pbtn(124, 275, 106, 32, rule_text, "serial_cycle", "single_rule",
                   bg=(42, 52, 64), border=COLOR_GRAY)

        coord_text = mgr.COORD_MODE_NAMES.get(port.get("coord_mode", "absolute"), "原始坐标")
        self._pbtn(8, 313, 222, 32, "坐标: " + coord_text,
                   "serial_cycle", "coord_mode", bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(8, 351, 106, 32, "频率:%dHz" % int(port.get("send_hz", 20)),
                   "serial_cycle", "send_hz", bg=(42, 52, 64), border=COLOR_BLUE)
        rx_ctl = bool(port.get("rx_control", False))
        self._pbtn(124, 351, 106, 32, "遥控:%s" % ("开" if rx_ctl else "关"),
                   "serial_toggle_rx", None, bg=self._active_bg(rx_ctl),
                   border=COLOR_GREEN if rx_ctl else COLOR_GRAY)

        state = "未开启"
        state_color = COLOR_YELLOW
        if mgr is not None and port.get("enabled", False):
            if mgr.uarts[idx] is not None:
                state = "工作中"
                state_color = COLOR_GREEN
            elif mgr.last_error[idx]:
                state = "异常"
                state_color = COLOR_RED
            else:
                state = "等待初始化"
        self._ptext(10, 389, 15, "状态:%s  发送:%d  接收:%d" %
                    (state, mgr.tx_count[idx], mgr.rx_count[idx]), state_color)
        if mgr.last_error[idx]:
            err = mgr.last_error[idx]
            if len(err) > 25:
                err = err[:25]
            self._ptext(10, 410, 13, err, COLOR_RED)
        else:
            cmd = mgr.last_command[idx]
            preview = ("收:" + cmd) if cmd else mgr.last_preview[idx]
            if len(preview) > 29:
                preview = preview[:29]
            self._ptext(10, 410, 13, preview if preview else "帧头55 / 帧尾66", COLOR_LIGHT)

        self._pbtn(8, 438, 106, 32, "应用串口", "serial_apply", None,
                   bg=(78, 55, 45), border=COLOR_ORANGE)
        self._pbtn(124, 438, 106, 32, "保存参数", "save_cfg",
                   bg=(35, 90, 58), border=COLOR_GREEN)

    # --------------------------------------------------------
    # Measurement / calibration
    # --------------------------------------------------------
    def _measure(self):
        self._panel_bg()
        self._header("测量标定")
        cfg = self.app.cfg["measurement"]
        mgr = self.app.measurement_manager

        enabled = bool(cfg.get("enabled", False))
        self._pbtn(8, 58, 222, 34, "测量叠加:%s" % ("开" if enabled else "关"),
                   "measure_toggle", "enabled", bg=self._active_bg(enabled),
                   border=COLOR_GREEN if enabled else COLOR_GRAY)

        # 来源改为绝对按钮，不再让用户不停循环猜当前选中了谁。
        source = str(cfg.get("source", "current"))
        self._ptext(10, 98, 16, "测量来源: " + mgr.SOURCE_NAMES.get(source, "当前功能"), COLOR_YELLOW)
        source_buttons = [
            (8, 120, "当前", "current"), (64, 120, "YOLO", "yolo"),
            (120, 120, "颜色", "color"), (176, 120, "形状", "shape"),
            (8, 154, "巡线", "line"), (64, 154, "码类", "code"),
            (120, 154, "几何", "geometry"), (176, 154, "模板", "template"),
        ]
        for x, y, text, value in source_buttons:
            active = source == value
            self._pbtn(x, y, 50, 28, text, "measure_source_set", value,
                       bg=self._active_bg(active), border=self._active_border(active))

        real_source = mgr.configured_source()
        open_name = mgr.SOURCE_NAMES.get(real_source, "识别功能")
        self._pbtn(8, 188, 222, 32, "进入%s并测量" % open_name,
                   "measure_open_source", None, bg=(45, 72, 95), border=COLOR_BLUE)

        self._pbtn(8, 226, 106, 32,
                   mgr.RULE_NAMES.get(cfg.get("target_rule", "center"), "中心最近"),
                   "measure_cycle", "target_rule", bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(124, 226, 106, 32,
                   mgr.REF_NAMES.get(cfg.get("reference", "screen_center"), "画面中心"),
                   "measure_cycle", "reference", bg=(42, 52, 64), border=COLOR_BLUE)

        self._pbtn(8, 264, 106, 32,
                   "单位:" + mgr.UNIT_NAMES.get(cfg.get("unit", "pixel"), "像素"),
                   "measure_cycle", "unit", bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(124, 264, 106, 32, "两点标定", "measure_calibrate", None,
                   bg=(74, 58, 92), border=COLOR_PURPLE)

        known = float(cfg.get("known_mm", 100.0))
        self._ptext(10, 302, 15, "标定实际距离: %.1f mm" % known, COLOR_YELLOW)
        self._pbtn(8, 322, 50, 28, "-10", "measure_known", -10, mode=PRESS_REPEAT)
        self._pbtn(64, 322, 50, 28, "-1", "measure_known", -1, mode=PRESS_REPEAT)
        self._pbtn(124, 322, 50, 28, "+1", "measure_known", 1, mode=PRESS_REPEAT)
        self._pbtn(180, 322, 50, 28, "+10", "measure_known", 10, mode=PRESS_REPEAT)

        scale = float(cfg.get("mm_per_pixel", 0.0))
        if scale > 0:
            self._ptext(10, 356, 15, "比例: %.5f mm/px" % scale, COLOR_GREEN)
        else:
            self._ptext(10, 356, 15, "比例: 未标定；像素测量仍可用", COLOR_YELLOW)

        self._pbtn(8, 378, 106, 28, "中心:%s" % ("开" if cfg.get("show_center", True) else "关"),
                   "measure_toggle", "show_center", bg=self._active_bg(cfg.get("show_center", True)))
        self._pbtn(124, 378, 106, 28, "偏移:%s" % ("开" if cfg.get("show_offset", True) else "关"),
                   "measure_toggle", "show_offset", bg=self._active_bg(cfg.get("show_offset", True)))
        self._pbtn(8, 412, 106, 28, "宽高:%s" % ("开" if cfg.get("show_size", True) else "关"),
                   "measure_toggle", "show_size", bg=self._active_bg(cfg.get("show_size", True)))
        self._pbtn(124, 412, 106, 28, "面积:%s" % ("开" if cfg.get("show_area", False) else "关"),
                   "measure_toggle", "show_area", bg=self._active_bg(cfg.get("show_area", False)))
        self._pbtn(8, 446, 106, 26,
                   ("再次确认" if self.app.confirm_pending("measure_clear") else "清除标定"),
                   "measure_clear", None, mode=PRESS_CONFIRM,
                   bg=(72, 45, 45), border=COLOR_RED)
        self._pbtn(124, 446, 106, 26, "保存参数", "save_cfg", None,
                   bg=(35, 90, 58), border=COLOR_GREEN)

    # --------------------------------------------------------
    # Scene presets
    # --------------------------------------------------------
    def _scene(self):
        self._panel_bg()
        self._header("场景模式")
        cfg = self.app.cfg["scene"]
        idx = int(clamp(int(cfg.get("selected", 0)), 0, SCENE_SLOT_COUNT - 1))
        mgr = self.app.scene_manager

        for i in range(SCENE_SLOT_COUNT):
            x = 8 + i * 56
            exists = mgr.exists(i)
            active = (idx == i)
            border = COLOR_GREEN if exists else self._active_border(active)
            self._pbtn(x, 60, 50, 36, str(i + 1), "scene_select", i,
                       bg=self._active_bg(active), border=border)

        exists = mgr.exists(idx)
        saved_page = mgr.saved_entry_page(idx) if exists else ""
        saved_name = SCENE_PAGE_NAMES.get(saved_page, "")
        state = "已保存" if exists else "空场景"
        if saved_name:
            state += " / " + saved_name
        self._ptext(10, 104, 18, "场景%d  %s" % (idx + 1, state),
                    COLOR_GREEN if exists else COLOR_YELLOW)

        entry_page = str(cfg.get("entry_page", "color"))
        if entry_page not in SCENE_PAGE_OPTIONS:
            entry_page = "color"
            cfg["entry_page"] = entry_page
        self._ptext(10, 132, 16, "加载后执行: " + SCENE_PAGE_NAMES.get(entry_page, entry_page), COLOR_CYAN)
        self._pbtn(8, 154, 106, 34, "上个功能", "scene_page_cycle", -1,
                   bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(124, 154, 106, 34, "下个功能", "scene_page_cycle", 1,
                   bg=(42, 52, 64), border=COLOR_BLUE)
        self._pbtn(8, 194, 222, 34, "绑定最近使用的识别功能", "scene_use_last_page", None,
                   bg=(52, 68, 62), border=COLOR_CYAN)

        self._ptext(10, 238, 14, "保存=功能 + 当前全部参数", COLOR_LIGHT)
        self._ptext(10, 258, 14, "LAB/模型/ROI/串口/测量等一起保存", COLOR_LIGHT)

        self._pbtn(8, 286, 222, 42, "加载场景并进入功能", "scene_load", None,
                   bg=(45, 72, 95), border=COLOR_BLUE)
        self._pbtn(8, 336, 222, 42, ("再次点击确认保存" if self.app.confirm_pending("scene_save") else "保存为当前场景"), "scene_save", None,
                   mode=PRESS_CONFIRM, bg=(35, 90, 58), border=COLOR_GREEN)
        self._pbtn(8, 386, 222, 42, ("再次点击确认删除" if self.app.confirm_pending("scene_delete") else "删除当前场景"), "scene_delete", None,
                   mode=PRESS_CONFIRM, bg=(95, 45, 45), border=COLOR_RED)
        self._ptext(10, 438, 13, "例：颜色调好→场景1→绑定颜色→保存", COLOR_LIGHT)
        self._ptext(10, 457, 13, "以后加载场景1会直接进入颜色识别", COLOR_LIGHT)

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------
    def _geometry(self):
        self._panel_bg()
        self._header("几何视觉", "geometry")
        cfg = self.app.cfg["geometry"]
        modes = [(8, 58, 50, "直线", "line"), (64, 58, 50, "线段", "segment"),
                 (120, 58, 50, "圆", "circle"), (176, 58, 54, "矩形", "rect")]
        for x, y, w, text, value in modes:
            active = cfg["mode"] == value
            self._pbtn(x, y, w, 36, text, "geometry_mode", value,
                       bg=self._active_bg(active), border=self._active_border(active))

        specs = self.app.geometry_param_specs()
        idx = int(clamp(cfg.get("param_idx", 0), 0, len(specs)-1))
        key, label, step, lo, hi = specs[idx]
        self._ptext(10, 106, 16, "当前: %s   调参 %d/%d" %
                    (GeometryEngine.MODE_NAMES.get(cfg["mode"], "几何"), idx+1, len(specs)), COLOR_YELLOW)
        self._ptext(10, 130, 18, "%s: %s" % (label, str(cfg[key])), COLOR_WHITE)
        self._pbtn(8, 158, 106, 36, "上一项", "geometry_param", -1)
        self._pbtn(124, 158, 106, 36, "下一项", "geometry_param", 1)
        self._pbtn(8, 200, 106, 38, "减少", "geometry_delta", -1, mode=PRESS_REPEAT)
        self._pbtn(124, 200, 106, 38, "增加", "geometry_delta", 1, mode=PRESS_REPEAT)

        sc = bool(cfg.get("show_center", True))
        sa = bool(cfg.get("show_angle", True))
        sf = bool(cfg.get("smart_filter", True))
        self._pbtn(8, 248, 70, 36, "中心:%s" % ("开" if sc else "关"), "geometry_toggle", "show_center",
                   bg=self._active_bg(sc), border=self._active_border(sc))
        mode = cfg.get("mode", "line")
        if mode == "circle":
            cv = bool(cfg.get("circle_verify", True))
            self._pbtn(85, 248, 70, 36, "圆周:%s" % ("开" if cv else "关"), "geometry_toggle", "circle_verify",
                       bg=self._active_bg(cv), border=self._active_border(cv))
        else:
            self._pbtn(85, 248, 70, 36, "角度:%s" % ("开" if sa else "关"), "geometry_toggle", "show_angle",
                       bg=self._active_bg(sa), border=self._active_border(sa))
        self._pbtn(162, 248, 68, 36, "过滤:%s" % ("开" if sf else "关"), "geometry_toggle", "smart_filter",
                   bg=self._active_bg(sf), border=self._active_border(sf))
        if mode == "line":
            tips = ["霍夫直线：优先保留长且强的直线", "阈值越高，杂线越少"]
            self._ptext(10, 302, 14, tips[0], COLOR_CYAN)
            self._ptext(10, 325, 14, tips[1], COLOR_LIGHT)
            self._ptext(10, 352, 14, "区域可单独限制几何检测范围", COLOR_LIGHT)
            self._ptext(10, 375, 14, "过滤关闭时显示官方原始候选", COLOR_LIGHT)
        elif mode == "segment":
            tips = ["LSD线段：已增加最小线长与去重复", "圆弧短切线会被优先过滤"]
            self._ptext(10, 302, 14, tips[0], COLOR_CYAN)
            self._ptext(10, 325, 14, tips[1], COLOR_LIGHT)
            self._ptext(10, 352, 14, "区域可单独限制几何检测范围", COLOR_LIGHT)
            self._ptext(10, 375, 14, "过滤关闭时显示官方原始候选", COLOR_LIGHT)
        elif mode == "circle":
            tips = ["霍夫圆只做候选，再验证整圈真实边缘", "整圈+四象限支持可压电线/纹理假圆"]
            self._ptext(10, 302, 14, tips[0], COLOR_CYAN)
            self._ptext(10, 325, 14, tips[1], COLOR_LIGHT)
            self._ptext(10, 352, 14, "标签Q越高越像完整真实圆；区域仍可限范围", COLOR_LIGHT)
            self._ptext(10, 375, 14, "圆周关闭=仅半径/NMS；过滤关闭=原始候选", COLOR_LIGHT)
        else:
            rv = bool(cfg.get("rect_circle_veto", True))
            self._pbtn(8, 292, 222, 32, "圆形互斥:%s" % ("开" if rv else "关"),
                       "geometry_toggle", "rect_circle_veto",
                       bg=self._active_bg(rv), border=self._active_border(rv))
            self._ptext(10, 332, 14, "矩形=四边+四角真实边缘验证", COLOR_CYAN)
            self._ptext(10, 354, 14, "近正方形候选再用霍夫圆互斥", COLOR_LIGHT)
            self._ptext(10, 376, 14, "圆形互斥关闭可查看原始矩形候选", COLOR_LIGHT)
        self._pbtn(8, 410, 222, 40, "保存参数", "save_cfg", bg=(35,90,58), border=COLOR_GREEN)

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------
    def _template(self):
        self._panel_bg()
        self._header("模板识别", "template_search")
        cfg = self.app.cfg["template"]
        idx = int(cfg.get("selected", 0))
        for i in range(TEMPLATE_SLOT_COUNT):
            x = 8 + i * 56
            active = i == idx
            exists = self.app.template_engine.exists(i)
            border = COLOR_GREEN if exists else self._active_border(active)
            self._pbtn(x, 58, 50, 36, str(i+1), "template_select", i,
                       bg=self._active_bg(active), border=border)
        state = "已有模板" if self.app.template_engine.exists(idx) else "空模板"
        self._ptext(10, 105, 18, "模板%d  %s" % (idx+1, state), COLOR_GREEN if state == "已有模板" else COLOR_YELLOW)
        self._ptext(10, 132, 17, "阈值 %.2f" % float(cfg.get("threshold", 0.72)), COLOR_WHITE)
        self._pbtn(8, 157, 106, 34, "阈值 -", "template_threshold", -0.02, mode=PRESS_REPEAT)
        self._pbtn(124, 157, 106, 34, "阈值 +", "template_threshold", 0.02, mode=PRESS_REPEAT)
        self._ptext(10, 200, 16, "步长 %d   搜索:%s" % (int(cfg.get("step",2)), "快速" if cfg.get("search") == "fast" else "完整"), COLOR_LIGHT)
        self._pbtn(8, 225, 70, 34, "步长-", "template_step", -1, mode=PRESS_REPEAT)
        self._pbtn(85, 225, 70, 34, "步长+", "template_step", 1, mode=PRESS_REPEAT)
        self._pbtn(162, 225, 68, 34, "切搜索", "template_search_mode")
        self._pbtn(8, 269, 106, 38, "框选模板", "region_edit", "template_capture", border=COLOR_PURPLE)
        self._pbtn(124, 269, 106, 38, "保存模板", "template_capture", None, bg=(58,82,55), border=COLOR_GREEN)
        sb = bool(cfg.get("show_box", True)); sc = bool(cfg.get("show_center", True))
        self._pbtn(8, 316, 106, 36, "检测框:%s" % ("开" if sb else "关"), "template_toggle", "show_box",
                   bg=self._active_bg(sb), border=self._active_border(sb))
        self._pbtn(124, 316, 106, 36, "中心:%s" % ("开" if sc else "关"), "template_toggle", "show_center",
                   bg=self._active_bg(sc), border=self._active_border(sc))
        self._pbtn(8, 362, 106, 38, "保存参数", "save_cfg", bg=(35,90,58), border=COLOR_GREEN)
        self._pbtn(124, 362, 106, 38, ("再次点击删除" if self.app.confirm_pending("template_delete") else "删除模板"), "template_delete", None,
                   mode=PRESS_CONFIRM, bg=(95,45,45), border=COLOR_RED)
        self._ptext(10, 410, 13, "先“框选模板”→确认→保存模板", COLOR_LIGHT)
        self._ptext(10, 431, 13, "右上“区域”设置匹配搜索范围", COLOR_LIGHT)
        self._ptext(10, 452, 13, "模板匹配使用灰度 NCC", COLOR_LIGHT)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    def _camera(self):
        self._panel_bg()
        self._header("相机设置")
        cfg = self.app.cfg["camera"]
        hm = bool(cfg.get("hmirror", False))
        vf = bool(cfg.get("vflip", False))

        # 使用绝对方向预设，不再依赖两个开关来回组合，现场更直观。
        presets = [
            (8,   62, "正常",     "normal",  (not hm and not vf)),
            (124, 62, "左右镜像", "hmirror", (hm and not vf)),
            (8,  110, "上下翻转", "vflip",   ((not hm) and vf)),
            (124,110, "旋转180",  "rotate180",(hm and vf)),
        ]
        for x, y, label, value, active in presets:
            self._pbtn(x, y, 106, 40, label, "camera_preset", value,
                       bg=self._active_bg(active), border=self._active_border(active))

        cur = self.app.camera_control.direction_name() if self.app.camera_control is not None else "未知"
        self._ptext(10, 164, 18, "当前：" + cur, COLOR_CYAN)
        self._ptext(10, 191, 15, "方向改变会重建视频链路，约1秒", COLOR_YELLOW)
        self._ptext(10, 214, 14, "显示/传统视觉/YOLO使用同一方向", COLOR_LIGHT)

        self._ptext(10, 249, 16, "显示通道 %dx%d" % (DISPLAY_W, DISPLAY_H), COLOR_LIGHT)
        self._ptext(10, 273, 16, "传统视觉 %dx%d" % (CV_W, CV_H), COLOR_LIGHT)
        self._ptext(10, 297, 16, "AI推理帧 %dx%d" % (AI_W, AI_H), COLOR_LIGHT)

        self._pbtn(8, 334, 222, 40, "重新应用当前方向", "camera_apply",
                   bg=(45,72,95), border=COLOR_BLUE)
        self._ptext(10, 385, 13, "修改方向后旧ROI对应的物理区域会变化", COLOR_LIGHT)
        self._ptext(10, 405, 13, "必要时重新框选区域；标定比例通常不变", COLOR_LIGHT)
        self._pbtn(8, 432, 222, 36, "保存参数", "save_cfg", bg=(35,90,58), border=COLOR_GREEN)

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------
    def _settings(self):
        self._panel_bg()
        self._header("系统设置")
        syscfg = self.app.cfg["system"]
        self._pbtn(8, 62, 222, 42, "FPS显示: %s" % ("开" if syscfg["show_fps"] else "关"),
                   "toggle_system", "show_fps", bg=self._active_bg(syscfg["show_fps"]))
        self._pbtn(8, 112, 222, 42, "触摸日志: %s" % ("开" if syscfg["touch_debug"] else "关"),
                   "toggle_system", "touch_debug", bg=self._active_bg(syscfg["touch_debug"]))
        if self.app.deferred_busy:
            touch_state = "处理中"
        elif self.app.deferred_action is not None:
            touch_state = "等待处理"
        else:
            touch_state = "就绪"
        self._ptext(10, 169, 17, "触摸状态: " + touch_state,
                    COLOR_YELLOW if touch_state != "就绪" else COLOR_GREEN)
        terr = self.app.touch_ctrl.error_count if self.app.touch_ctrl is not None else 0
        self._ptext(10, 193, 15, "触摸读取错误: %d" % terr, COLOR_LIGHT)
        if self.app.last_deferred_name:
            self._ptext(10, 216, 14, "最近耗时: %s %dms" %
                        (self.app.last_deferred_name[:12], self.app.last_deferred_ms), COLOR_LIGHT)
        else:
            self._ptext(10, 216, 14, "最近耗时: --", COLOR_LIGHT)
        self._pbtn(8, 260, 222, 42, "保存全部参数", "save_cfg",
                   bg=(35, 90, 58), border=COLOR_GREEN)
        self._pbtn(8, 310, 222, 42, "重新加载参数", "load_cfg",
                   bg=(45, 72, 95), border=COLOR_BLUE)
        self._pbtn(8, 360, 222, 46, ("再次点击确认恢复" if self.app.confirm_pending("reset_all") else "恢复全部默认"), "reset_all",
                   mode=PRESS_CONFIRM, bg=(95, 45, 45), border=COLOR_RED)
        self._ptext(10, 420, 14, "危险操作：两次单击确认", COLOR_YELLOW)
        self._ptext(10, 443, 14, "耗时操作在触摸结束后执行", COLOR_LIGHT)

    def render(self):
        if not self.dirty:
            return
        self.dirty = False
        self.pool.reset()
        page = self.app.page
        try:
            if page == "home":
                self._home()
                print("[UI] rendered home buttons=%d" % self.pool.used)
                return

            if page == "shape":
                self._shape()
            elif page == "color":
                self._color()
            elif page == "line":
                self._line()
            elif page == "yolo":
                self._yolo()
            elif page == "code":
                self._code()
            elif page == "serial":
                self._serial()
            elif page == "measure":
                self._measure()
            elif page == "scene":
                self._scene()
            elif page == "geometry":
                self._geometry()
            elif page == "template":
                self._template()
            elif page == "camera":
                self._camera()
            elif page == "settings":
                self._settings()
            else:
                self.app.page = "home"
                self.dirty = True
                return

            self.pool.draw_all(self.panel, offset_x=PANEL_X)
            Display.show_image(self.panel, x=PANEL_X, y=0, layer=Display.LAYER_OSD1)
            print("[UI] rendered page=%s buttons=%d" % (page, self.pool.used))
        except Exception as e:
            print("[UI ERROR]", page, e)
            self.dirty = True


# ============================================================
# 11. Main Application
# ============================================================

class VisionWorkbench:
    def __init__(self):
        self.store = ConfigStore()
        self.cfg = self.store.load()
        self.page = "home"
        # 记录最近真正运行过的视觉功能，供“当前来源”和场景快速绑定使用。
        scene_entry = str(self.cfg.get("scene", {}).get("entry_page", "color"))
        self.last_vision_page = scene_entry if scene_entry in VISION_PAGE_OPTIONS else "color"

        self.results = VisionResultCenter()
        self.region_manager = RegionManager(self)
        self.measurement_manager = MeasurementManager(self)
        self.calibration_manager = CalibrationManager(self)
        self.scene_manager = SceneManager(self)
        self.serial_manager = SerialManager(self)
        self.camera_control = CameraControl(self)

        self.sensor = None
        self.touch = None
        self.touch_ctrl = None
        self.overlay_img = None
        self.panel_img = None
        self.ui = None

        self.shape_engine = None
        self.color_engine = None
        self.line_engine = None
        self.yolo_engine = None
        self.code_engine = None
        self.geometry_engine = None
        self.template_engine = None

        self.fps_clock = time.clock()
        self.frame_counter = 0
        self.notice_text = ""
        self.notice_color = COLOR_YELLOW
        self.notice_until = 0

        # 当前颜色目标的“二次单击确认”状态。只作用于当前 profile，绝不影响其它目标。
        self.color_clear_confirm_idx = -1
        self.color_clear_confirm_until = 0

        # YOLO 模型切换采用“触摸回调只选模型，主循环再重建”的两阶段方式。
        # 这样模型 deinit/config_preprocess 的耗时不会卡在 TAP 提交过程里。
        self.pending_yolo_rebuild = False
        self.pending_yolo_ready_ms = 0
        self.pending_template_capture = False

        # 危险操作统一使用“两次单击确认”，不再依赖当前触摸屏不稳定的长按事件。
        self.confirm_action = None
        self.confirm_payload = None
        self.confirm_until = 0
        self.confirm_armed_ms = 0

        # 耗时动作从触摸回调中移出，在主循环里串行执行。
        self.deferred_action = None
        self.deferred_payload = None
        self.deferred_busy = False
        self.deferred_ready_ms = 0
        self.last_deferred_name = ""
        self.last_deferred_ms = 0

        # 一维条码专用原生灰度通道状态。
        # False: CH2 = 640x360 RGBP888（YOLO）
        # True : CH2 = 640x480 GRAYSCALE（官方条码链路）
        self.barcode_native_active = False
        self.barcode_pipeline_failed = False

    def cancel_color_clear_confirm(self):
        self.color_clear_confirm_idx = -1
        self.color_clear_confirm_until = 0

    def color_clear_pending(self):
        if self.color_clear_confirm_idx < 0:
            return False
        now = time.ticks_ms()
        if safe_ticks_diff(self.color_clear_confirm_until, now) <= 0:
            self.cancel_color_clear_confirm()
            return False
        return self.color_clear_confirm_idx == int(self.cfg["color"]["current"])

    # --------------------- parameter specs -------------------
    def shape_param_specs(self):
        # 和几何视觉一样按模式分页：激光模式下不再出现 Canny 那些用不上的项。
        if self.cfg.get("shape", {}).get("mode", "auto") == "laser":
            return [
                ("laser_threshold", "激光亮度门限", 2, 60, 255),
                ("laser_min_pixels", "激光最小像素", 1, 1, 200),
                ("laser_max_pixels", "激光最大像素", 50, 20, 20000),
                ("laser_min_density", "激光最小密度", 0.03, 0.10, 1.00),
                ("laser_min_chroma", "激光最小色度", 4, 0, 100),
                ("laser_bright_margin", "只看最亮的", 2, 0, 255),
                ("laser_overlay", "叠加其它模式", 1, 0, 1),
            ]
        return [
            ("min_area", "最小面积", 100, 100, 12000),
            ("canny_low", "Canny低", 5, 5, 220),
            ("canny_high", "Canny高", 5, 15, 255),
            ("epsilon_ratio", "多边形Eps", 0.002, 0.006, 0.060),
            ("stable_frames", "稳定帧数", 1, 1, 5),
            ("rect_min_fill", "矩形填充", 0.03, 0.45, 0.95),
            ("circle_min_circularity", "圆度阈值", 0.03, 0.45, 0.95),
            ("laser_overlay", "叠加激光点", 1, 0, 1),
        ]

    def line_param_specs(self):
        return [
            ("area_threshold", "最小区域", 10, 20, 1800),
            ("pixels_threshold", "最小像素", 5, 10, 1200),
            ("min_density", "最小密度", 0.02, 0.02, 0.70),
            ("roi_count", "ROI数量", 1, 4, 9),
            ("corridor_half_width", "走廊半宽", 5, 30, 140),
            ("continuity_px", "连续门限", 4, 12, 100),
            ("eight_max_gap", "断线容许", 1, 0, 8),
            ("eight_half_width", "8N搜索半宽", 5, 12, 100),
            ("eight_max_band_width", "最大色带宽", 5, 12, 180),
            ("smooth_alpha", "平滑系数", 0.05, 0.10, 0.95),
        ]

    def geometry_param_specs(self):
        """几何参数按当前算法分页，不再出现“圆模式还在调直线阈值”的混乱。"""
        mode = self.cfg.get("geometry", {}).get("mode", "line")
        if mode == "line":
            return [
                ("line_threshold", "直线阈值", 100, 100, 30000),
                ("line_min_length", "最小线长", 10, 10, 700),
                ("line_x_stride", "X步长", 1, 1, 8),
                ("line_y_stride", "Y步长", 1, 1, 8),
                ("theta_margin", "角度合并", 2, 0, 90),
                ("rho_margin", "距离合并", 2, 0, 120),
                ("max_results", "最大结果数", 1, 1, 20),
            ]
        if mode == "segment":
            return [
                ("segment_min_length", "最小线段长度", 10, 10, 700),
                ("segment_merge_distance", "线段合并距离", 2, 0, 160),
                ("segment_theta_difference", "合并角度差", 2, 0, 90),
                ("segment_nms_angle", "去重角度差", 1, 0, 45),
                ("segment_nms_distance", "去重中心距离", 2, 0, 160),
                ("max_results", "最大结果数", 1, 1, 20),
            ]
        if mode == "circle":
            return [
                ("circle_threshold", "圆检测阈值", 200, 100, 50000),
                ("circle_min_radius", "最小半径", 5, 2, 350),
                ("circle_max_radius", "最大半径", 5, 4, 600),
                ("circle_merge_center", "同圆中心合并", 3, 0, 200),
                ("circle_merge_radius", "同圆半径合并", 3, 0, 250),
                ("circle_edge_contrast", "圆周对比阈值", 2, 1, 180),
                ("circle_edge_support", "整圈支持率%", 5, 0, 100),
                ("circle_quadrant_support", "象限支持率%", 4, 0, 100),
                ("circle_x_stride", "X步长", 1, 1, 8),
                ("circle_y_stride", "Y步长", 1, 1, 8),
                ("circle_x_margin", "X合并范围", 2, 0, 120),
                ("circle_y_margin", "Y合并范围", 2, 0, 120),
                ("circle_r_margin", "半径底层合并", 2, 0, 160),
                ("max_results", "最大结果数", 1, 1, 20),
            ]
        return [
            ("rect_threshold", "矩形阈值", 500, 100, 80000),
            ("rect_min_side", "最小边长", 5, 10, 700),
            ("rect_max_angle_error", "直角最大误差", 2, 5, 60),
            ("rect_edge_contrast", "边缘对比阈值", 2, 1, 180),
            ("rect_edge_support", "四边支持率%", 5, 0, 100),
            ("rect_corner_support", "四角支持率%", 5, 0, 100),
            ("rect_circle_aspect_max", "排圆长宽比%", 5, 100, 250),
            ("rect_circle_center_ratio", "排圆中心距%", 2, 5, 80),
            ("max_results", "最大结果数", 1, 1, 20),
        ]

    # --------------------- notice / FPS ----------------------
    def notice(self, text, color=COLOR_YELLOW, duration_ms=1500):
        self.notice_text = str(text)
        self.notice_color = color
        self.notice_until = time.ticks_ms() + duration_ms
        print("[NOTICE]", text)

    def draw_runtime_decor(self, osd):
        if self.region_manager is not None:
            region_key = {
                "shape": "shape", "color": "color", "yolo": "yolo", "code": "code",
                "geometry": "geometry", "template": "template_search",
            }.get(self.page)
            if region_key is not None:
                self.region_manager.draw_active(osd, region_key)
        if self.measurement_manager is not None:
            self.measurement_manager.draw_overlay(osd)
        if self.cfg["system"].get("show_fps", True):
            try:
                fps = self.fps_clock.fps()
                osd.draw_string_advanced(8, DISPLAY_H - 57, 18, "FPS %.1f" % fps, color=COLOR_GREEN)
            except Exception:
                pass
        if self.notice_text:
            now = time.ticks_ms()
            if safe_ticks_diff(self.notice_until, now) > 0:
                text = self.notice_text
                if len(text) > 34:
                    text = text[:34]
                osd.draw_string_advanced(8, DISPLAY_H - 84, 18, text, color=self.notice_color)
            else:
                self.notice_text = ""

    # --------------------- hardware --------------------------
    def restart_camera_pipeline(self):
        """
        按官方 Sensor / Display / Media 生命周期完整重建摄像头视频链路。

        该方法只由延迟动作队列调用，触摸手势已经结束。
        返回 True 表示视频链路已恢复，False 表示失败。
        """
        print("[相机链路] 开始重建")
        old_overlay = self.overlay_img
        old_panel = self.panel_img

        # KPU / 模板资源可能引用当前媒体缓冲，先释放，避免 MediaManager.deinit 时残留。
        try:
            if self.yolo_engine is not None:
                self.yolo_engine.unload()
        except Exception as e:
            print("[相机链路] YOLO释放提示:", e)
        try:
            if self.template_engine is not None:
                self.template_engine.unload()
        except Exception as e:
            print("[相机链路] 模板释放提示:", e)

        # 先停止 Sensor；Display 必须在 MediaManager 之前 deinit。
        try:
            if self.sensor is not None:
                self.sensor.stop()
        except Exception as e:
            print("[相机链路] sensor.stop提示:", e)

        try:
            Display.deinit()
        except Exception as e:
            print("[相机链路] Display.deinit提示:", e)

        time.sleep_ms(80)

        try:
            MediaManager.deinit()
        except Exception as e:
            print("[相机链路] MediaManager.deinit提示:", e)

        # 解除旧 OSD 图像引用，让旧媒体/图像对象尽快释放。
        self.overlay_img = None
        self.panel_img = None
        if self.ui is not None:
            try:
                self.ui.overlay = None
                self.ui.panel = None
            except Exception:
                pass
        old_overlay = None
        old_panel = None
        self.sensor = None
        gc.collect()
        time.sleep_ms(100)

        try:
            # 重新创建 Sensor；方向必须在 run 前设置。
            sensor = Sensor()
            sensor.reset()

            camcfg = self.cfg.get("camera", {})
            sensor.set_hmirror(bool(camcfg.get("hmirror", False)))
            sensor.set_vflip(bool(camcfg.get("vflip", False)))

            # 三通道重新配置。
            sensor.set_framesize(width=DISPLAY_W, height=DISPLAY_H, chn=CAM_CHN_ID_0)
            sensor.set_pixformat(Sensor.YUV420SP, chn=CAM_CHN_ID_0)

            sensor.set_framesize(width=CV_W, height=CV_H, chn=CAM_CHN_ID_1)
            sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)

            # CH2 根据当前功能二选一：
            # - 一维条码：严格按官方例程使用 640x480 原生 GRAYSCALE；
            # - 其它功能：恢复原来的 640x360 RGBP888，供 YOLO 使用。
            if self.barcode_native_active:
                sensor.set_framesize(width=640, height=480, chn=CAM_CHN_ID_2)
                sensor.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_2)
            else:
                sensor.set_framesize(width=AI_W, height=AI_H, chn=CAM_CHN_ID_2)
                sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

            # bind_layer 必须在 Display.init 前重新建立。
            bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
            Display.bind_layer(**bind_info, layer=Display.LAYER_VIDEO1)

            try:
                Display.init(
                    Display.ST7701,
                    width=DISPLAY_W,
                    height=DISPLAY_H,
                    osd_num=2,
                    to_ide=True,
                    quality=int(self.cfg["system"].get("ide_quality", 70))
                )
            except TypeError:
                Display.init(
                    Display.ST7701,
                    width=DISPLAY_W,
                    height=DISPLAY_H,
                    osd_num=2,
                    to_ide=True
                )

            try:
                sensor._set_chn_fps(chn=CAM_CHN_ID_0, fps=Display.fps())
            except Exception:
                pass

            MediaManager.init()
            sensor.run()
            self.sensor = sensor
            time.sleep_ms(260)

            # 重建 OSD，旧 Display 通路销毁后不能继续假定旧 OSD buffer 有效。
            self.overlay_img = image.Image(DISPLAY_W, DISPLAY_H, image.ARGB8888)
            self.panel_img = image.Image(PANEL_W, PANEL_H, image.ARGB8888)
            self.overlay_img.clear()
            self.panel_img.clear()

            if self.ui is not None:
                self.ui.overlay = self.overlay_img
                self.ui.panel = self.panel_img
                self.ui.invalidate()

            Display.show_image(self.overlay_img, layer=Display.LAYER_OSD0)
            Display.show_image(
                self.panel_img,
                x=PANEL_X,
                y=0,
                layer=Display.LAYER_OSD1
            )

            # 用 snapshot 验证算法通道也已经恢复，不仅仅是 VIDEO 层。
            try:
                test_img = self.sensor.snapshot(chn=CAM_CHN_ID_1, timeout=1200)
                if test_img is None:
                    raise RuntimeError("传统视觉通道未恢复")
                test_img = None
            except Exception as e:
                raise RuntimeError("相机重启后抓帧失败: %s" % str(e))

            if self.camera_control is not None:
                self.camera_control.mark_current_as_applied()

            # 回到当前页面重新绘制 OSD/UI。
            self.clear_result()
            if self.ui is not None:
                self.ui.invalidate()

            print("[相机链路] 重建完成")
            return True

        except Exception as e:
            msg = str(e)
            print("[相机链路] 重建失败:", msg)
            if self.camera_control is not None:
                self.camera_control.last_error = msg

            # 最后一层恢复：若新链路初始化到一半失败，尽量完整清场。
            try:
                if self.sensor is not None:
                    self.sensor.stop()
            except Exception:
                pass
            try:
                Display.deinit()
            except Exception:
                pass
            try:
                MediaManager.deinit()
            except Exception:
                pass
            self.sensor = None
            gc.collect()
            return False

    def _service_barcode_native_pipeline(self):
        """
        一维条码页面自动切换 CH2 到官方原生灰度链路。
        离开条码页面后自动恢复 YOLO 的 RGBP888 通道。

        只在触摸手势结束、没有其它延迟任务时执行完整媒体链路重建。
        """
        cfg = self.cfg.get("code", {})

        # 一维条码固定使用官方原生灰度。
        # 不再由 barcode_highres 开关决定是否启用。
        desired = (
            self.page == "code" and
            cfg.get("mode") == "barcode"
        )

        if desired == self.barcode_native_active:
            return
        if self.touch_ctrl is not None and self.touch_ctrl.active:
            return
        if self.deferred_busy or self.deferred_action is not None:
            return

        # 本次进入原生灰度失败后不每帧无限重试。
        if desired and self.barcode_pipeline_failed:
            return

        old_state = self.barcode_native_active
        self.barcode_native_active = desired

        try:
            if desired:
                self.notice("正在切换条码原生灰度 640x480", COLOR_CYAN, 1200)
                print("[条形码] 切换 CH2 -> 640x480 GRAYSCALE")
            else:
                self.notice("正在恢复YOLO图像通道", COLOR_CYAN, 1000)
                print("[条形码] 恢复 CH2 -> %dx%d RGBP888" % (AI_W, AI_H))

            ok = self.restart_camera_pipeline()
            if not ok:
                raise RuntimeError("相机链路重建失败")

            self.barcode_pipeline_failed = False

        except Exception as e:
            print("[条形码通道]", e)
            self.barcode_native_active = old_state
            self.barcode_pipeline_failed = True if desired else False
            self.notice("条码原生灰度切换失败: " + str(e), COLOR_RED, 1800)

    def init_hardware(self):
        print("[INIT] K230 万能视觉平台")
        print("[CAP] cv2:", CV2_OK,
              "YOLOv5:", YOLOv5 is not None,
              "YOLOv8:", YOLOv8 is not None,
              "YOLO11:", YOLO11 is not None,
              "YOLO26:", YOLO26 is not None)
        self.sensor = Sensor()
        self.sensor.reset()
        # 镜像/翻转均为官方 Sensor API；启动前先应用保存的方向。
        try:
            self.sensor.set_hmirror(bool(self.cfg.get("camera", {}).get("hmirror", False)))
            self.sensor.set_vflip(bool(self.cfg.get("camera", {}).get("vflip", False)))
        except Exception as e:
            print("[相机方向初始化]", e)

        # CH0：直接绑定显示，算法再慢也不拖相机视频层。
        self.sensor.set_framesize(width=DISPLAY_W, height=DISPLAY_H, chn=CAM_CHN_ID_0)
        self.sensor.set_pixformat(Sensor.YUV420SP, chn=CAM_CHN_ID_0)

        # CH1：传统视觉 RGB565，支持 LAB/find_blobs/get_regression。
        self.sensor.set_framesize(width=CV_W, height=CV_H, chn=CAM_CHN_ID_1)
        self.sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)

        # CH2：YOLO RGBP888 / CHW。
        self.sensor.set_framesize(width=AI_W, height=AI_H, chn=CAM_CHN_ID_2)
        self.sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

        bind_info = self.sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
        Display.bind_layer(**bind_info, layer=Display.LAYER_VIDEO1)

        # 开启 LCD + IDE 同步。若旧固件不接受 quality 参数，自动回退到默认质量。
        try:
            Display.init(Display.ST7701, width=DISPLAY_W, height=DISPLAY_H,
                         osd_num=2, to_ide=True,
                         quality=int(self.cfg["system"].get("ide_quality", 70)))
        except TypeError:
            Display.init(Display.ST7701, width=DISPLAY_W, height=DISPLAY_H,
                         osd_num=2, to_ide=True)

        try:
            self.sensor._set_chn_fps(chn=CAM_CHN_ID_0, fps=Display.fps())
        except Exception:
            pass

        MediaManager.init()
        self.sensor.run()
        # 启动前已经按配置设置过方向，记录为当前硬件状态；
        # 后续加载同方向场景时无需无意义地再次 stop/run。
        try:
            self.camera_control.mark_current_as_applied()
        except Exception:
            pass

        # 动态结果全屏 OSD0；UI 只分配右侧 240x480 OSD1。
        self.overlay_img = image.Image(DISPLAY_W, DISPLAY_H, image.ARGB8888)
        self.panel_img = image.Image(PANEL_W, PANEL_H, image.ARGB8888)
        self.overlay_img.clear()
        self.panel_img.clear()
        Display.show_image(self.overlay_img, layer=Display.LAYER_OSD0)
        Display.show_image(self.panel_img, x=PANEL_X, y=0, layer=Display.LAYER_OSD1)

        self.touch = TOUCH(0)
        print("[TOUCH]", self.touch)

        self.shape_engine = ShapeEngine(self)
        self.color_engine = ColorEngine(self)
        self.line_engine = LineFollowerEngine(self)
        self.yolo_engine = YoloEngine(self)
        self.code_engine = CodeEngine(self)
        self.geometry_engine = GeometryEngine(self)
        self.template_engine = TemplateEngine(self)
        self.ui = UIManager(self, self.overlay_img, self.panel_img)
        self.touch_ctrl = TouchController(self, self.touch)

        # 串口默认关闭；若配置中已启用，则在硬件/显示初始化完成后再统一打开。
        self.serial_manager.open_enabled()

        self.ui.invalidate()
        self.ui.render()
        print("[INIT] 平台已就绪，LCD 与 IDE 同步显示已开启")

    def clear_result(self):
        if self.overlay_img is not None:
            self.overlay_img.clear()
            try:
                Display.show_image(self.overlay_img, layer=Display.LAYER_OSD0)
            except Exception:
                pass

    # --------------------- touch canvas ----------------------
    def on_canvas_touch(self, page, x, y):
        # 只有明确进入“训练取色”状态后，算法画面才响应点击。
        if x >= PANEL_X:
            return
        if page == "color" and self.color_engine.sample_armed:
            if self.color_engine.request_training_at(x, y):
                self.ui.invalidate()
        elif page == "line" and self.line_engine.sample_armed:
            if self.line_engine.request_training_at(x, y):
                self.ui.invalidate()

    # --------------------- action helpers --------------------
    def _shape_delta(self, direction):
        cfg = self.cfg["shape"]
        specs = self.shape_param_specs()
        idx = int(clamp(cfg.get("param_idx", 0), 0, len(specs) - 1))
        key, label, step, lo, hi = specs[idx]
        cfg[key] = clamp(cfg[key] + step * direction, lo, hi)
        if key in ("min_area", "canny_low", "canny_high", "stable_frames",
                   "laser_threshold", "laser_min_pixels", "laser_max_pixels",
                   "laser_min_chroma", "laser_bright_margin", "laser_overlay"):
            cfg[key] = int(round(cfg[key]))
        # 激光模式下自动保证 最小像素 <= 最大像素
        if cfg.get("laser_max_pixels", 0) < cfg.get("laser_min_pixels", 0):
            if key == "laser_min_pixels":
                cfg["laser_max_pixels"] = cfg["laser_min_pixels"]
            else:
                cfg["laser_min_pixels"] = cfg["laser_max_pixels"]
        if cfg["canny_high"] <= cfg["canny_low"] + 5:
            if key == "canny_low":
                cfg["canny_high"] = min(255, cfg["canny_low"] + 10)
            else:
                cfg["canny_low"] = max(5, cfg["canny_high"] - 10)
        self.shape_engine.prev_tracks = []

    def _color_delta(self, delta):
        cfg = self.cfg["color"]
        idx = cfg["current"]
        p = cfg["profiles"][idx]
        if not p["trained"]:
            self.notice("该目标尚未训练，先点击“训练取色”", COLOR_YELLOW, 1800)
            return
        pi = int(cfg.get("param_idx", 0))
        if pi in (0, 1):
            p["lab"][pi] = int(clamp(p["lab"][pi] + delta, 0, 100))
        else:
            p["lab"][pi] = int(clamp(p["lab"][pi] + delta, -128, 127))
        # 保持三组 low/high 合法。
        for lo_i, hi_i in ((0, 1), (2, 3), (4, 5)):
            if p["lab"][lo_i] > p["lab"][hi_i]:
                if pi == lo_i:
                    p["lab"][hi_i] = p["lab"][lo_i]
                else:
                    p["lab"][lo_i] = p["lab"][hi_i]

    def _line_delta(self, direction):
        cfg = self.cfg["line"]
        specs = self.line_param_specs()
        idx = int(clamp(cfg.get("param_idx", 0), 0, len(specs) - 1))
        key, label, step, lo, hi = specs[idx]
        cfg[key] = clamp(cfg[key] + step * direction, lo, hi)
        if key not in ("smooth_alpha", "min_density"):
            cfg[key] = int(round(cfg[key]))

    def _remember_current_yolo(self):
        remember_yolo_runtime(self.cfg["yolo"])

    def _select_yolo_model(self, target_index):
        # 这里只做“选择”，不在触摸回调里执行耗时 rebuild。
        # Button snapshot 携带绝对索引，所以同一次动作即使被重复调用也是幂等的。
        y = self.cfg["yolo"]
        total = yolo_profile_count()
        if total <= 0:
            self.notice("没有配置YOLO模型", COLOR_RED, 1800)
            return

        new_idx = int(clamp(int(target_index), 0, total - 1))
        old_idx = int(y.get("model_index", 0))
        if new_idx == old_idx:
            return

        self._remember_current_yolo()
        apply_yolo_profile_to_cfg(y, new_idx, use_saved=True)
        self.yolo_engine.last_failed_index = -1
        self.yolo_engine.manual_unloaded = False
        self.yolo_engine.needs_rebuild = True
        self.pending_yolo_rebuild = True
        self.pending_yolo_ready_ms = time.ticks_ms() + 80
        self.notice("切换模型: " + y["profile_name"], COLOR_CYAN, 900)

    def _service_pending_yolo_rebuild(self):
        if not self.pending_yolo_rebuild:
            return
        if self.touch_ctrl is not None and self.touch_ctrl.active:
            return
        if self.deferred_busy:
            return
        if safe_ticks_diff(self.pending_yolo_ready_ms, time.ticks_ms()) > 0:
            return
        self.pending_yolo_rebuild = False
        self.yolo_engine.rebuild()
        # rebuild 结束后重新从“现在”开始防抖，而不是沿用切换前的旧时间戳。
        if self.touch_ctrl is not None:
            self.touch_ctrl.force_rearm(TOUCH_YOLO_REARM_MS)
        if self.ui is not None:
            self.ui.invalidate()

    def _cycle_value(self, current, options):
        if not options:
            return current
        try:
            idx = options.index(current)
        except Exception:
            idx = -1
        return options[(idx + 1) % len(options)]

    def _serial_cycle(self, key):
        scfg = self.cfg["serial"]
        idx = int(clamp(int(scfg.get("selected", 0)), 0, len(SERIAL_PORT_PROFILES) - 1))
        port = scfg["ports"][idx]
        if key == "baudrate":
            port["baudrate"] = self._cycle_value(int(port.get("baudrate", SERIAL_DEFAULT_BAUD)), SERIAL_BAUD_OPTIONS)
        elif key == "protocol":
            port["protocol"] = self._cycle_value(port.get("protocol", "binary"), SERIAL_PROTOCOL_OPTIONS)
        elif key == "source":
            port["source"] = self._cycle_value(port.get("source", "current"), SERIAL_SOURCE_OPTIONS)
        elif key == "target_mode":
            port["target_mode"] = self._cycle_value(port.get("target_mode", "single"), SERIAL_TARGET_MODE_OPTIONS)
        elif key == "single_rule":
            port["single_rule"] = self._cycle_value(port.get("single_rule", "center"), SERIAL_SINGLE_RULE_OPTIONS)
        elif key == "coord_mode":
            port["coord_mode"] = self._cycle_value(port.get("coord_mode", "absolute"), SERIAL_COORD_MODE_OPTIONS)
        elif key == "send_hz":
            port["send_hz"] = self._cycle_value(int(port.get("send_hz", 20)), SERIAL_HZ_OPTIONS)

    def _measure_cycle(self, key):
        cfg = self.cfg["measurement"]
        if key == "source":
            cfg["source"] = self._cycle_value(cfg.get("source", "current"), MEASURE_SOURCE_OPTIONS)
        elif key == "target_rule":
            cfg["target_rule"] = self._cycle_value(cfg.get("target_rule", "center"), MEASURE_RULE_OPTIONS)
        elif key == "reference":
            cfg["reference"] = self._cycle_value(cfg.get("reference", "screen_center"), MEASURE_REF_OPTIONS)
        elif key == "unit":
            cfg["unit"] = self._cycle_value(cfg.get("unit", "pixel"), MEASURE_UNIT_OPTIONS)

    def _scene_cycle_page(self, direction):
        cfg = self.cfg.setdefault("scene", {})
        current = str(cfg.get("entry_page", "color"))
        try:
            idx = SCENE_PAGE_OPTIONS.index(current)
        except Exception:
            idx = 0
        idx = (idx + int(direction)) % len(SCENE_PAGE_OPTIONS)
        cfg["entry_page"] = SCENE_PAGE_OPTIONS[idx]

    def _open_measure_source(self):
        source = self.measurement_manager.configured_source()
        if source not in VISION_PAGE_OPTIONS:
            source = "color"
        # 点“进入来源识别”本身就代表用户希望看到测量结果，因此自动打开叠加。
        self.cfg["measurement"]["enabled"] = True
        self.on_action("nav", source)

    def apply_scene(self, slot):
        cfg = self.scene_manager.load_config(slot)
        if cfg is None:
            self.notice("当前场景还没有保存", COLOR_YELLOW, 1500)
            return False

        target_page = str(cfg.get("scene", {}).get("entry_page", "color"))
        if target_page not in SCENE_PAGE_OPTIONS:
            target_page = "color"

        self.pending_yolo_rebuild = False
        self.cancel_color_clear_confirm()
        try: self.color_engine.cancel_training()
        except Exception: pass
        try: self.line_engine.cancel_training()
        except Exception: pass
        try: self.yolo_engine.unload()
        except Exception: pass
        try: self.template_engine.unload()
        except Exception: pass
        try: self.serial_manager.deinit()
        except Exception: pass

        self.cfg = cfg
        self.last_vision_page = target_page
        self.shape_engine.prev_tracks = []
        self._apply_camera_direction()
        self.serial_manager.open_enabled()

        # 场景加载的核心语义：恢复参数后直接进入绑定的比赛功能。
        self.page = target_page
        self.clear_result()
        self.notice("场景%d已加载：%s" % (int(slot) + 1, SCENE_PAGE_NAMES.get(target_page, target_page)),
                    COLOR_GREEN, 1700)
        if self.ui is not None:
            self.ui.invalidate()
            try:
                self.ui.render()
            except Exception:
                pass
        if self.touch_ctrl is not None:
            self.touch_ctrl.force_rearm(300)
        return True

    def _geometry_delta(self, direction):
        cfg = self.cfg["geometry"]
        specs = self.geometry_param_specs()
        idx = int(clamp(cfg.get("param_idx", 0), 0, len(specs)-1))
        key, label, step, lo, hi = specs[idx]
        cfg[key] = int(clamp(int(cfg[key]) + int(step) * int(direction), lo, hi))
        if cfg.get("circle_max_radius", 0) < cfg.get("circle_min_radius", 0):
            if key == "circle_min_radius":
                cfg["circle_max_radius"] = cfg["circle_min_radius"]
            else:
                cfg["circle_min_radius"] = cfg["circle_max_radius"]

    def _service_pending_template_capture(self):
        if not self.pending_template_capture:
            return
        if self.touch_ctrl is not None and self.touch_ctrl.active:
            return
        if self.deferred_busy:
            return
        self.pending_template_capture = False
        try:
            img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
            if self.template_engine.capture_from_frame(img):
                self.notice("当前模板已保存", COLOR_GREEN, 1400)
            else:
                self.notice("模板保存失败: " + self.template_engine.last_error, COLOR_RED, 2200)
        except Exception as e:
            self.notice("模板保存异常: " + str(e), COLOR_RED, 2200)
        if self.ui is not None:
            self.ui.invalidate()

    def _apply_camera_direction(self, force=False):
        # 运行中的 Sensor 通过 stop -> 设置方向 -> run 重新生效。
        # 只在相机设置/场景恢复等低频操作触发，不放进普通算法帧。
        if self.camera_control.apply(restart=True, force=force):
            self.clear_result()
            if self.touch_ctrl is not None:
                self.touch_ctrl.force_rearm(420)
            self.notice("相机方向已应用：" + self.camera_control.direction_name(), COLOR_GREEN, 1100)
            return True
        self.notice("相机方向设置失败: " + self.camera_control.last_error, COLOR_RED, 2000)
        return False

    def handle_remote_ascii(self, port_index, text):
        """串口远程控制。只在对应串口“遥控=开”时由 SerialManager 调用。"""
        cmd = str(text).strip().upper().replace(" ", "")
        if not cmd:
            return
        page_map = {"H": "home", "Y": "yolo", "C": "color", "S": "shape", "L": "line", "Q": "code",
                    "G": "geometry", "GEOMETRY": "geometry", "TEMPLATE": "template", "CAMERA": "camera",
                    "MEASURE": "measure", "SCENE": "scene"}
        if cmd in page_map:
            self.on_action("nav", page_map[cmd]); return
        if cmd.startswith("M") and cmd[1:].isdigit():
            self._select_yolo_model(int(cmd[1:]) - 1); return
        if cmd == "SAVE":
            self.defer_action("save_cfg", None, "串口命令：准备保存参数")
            return
        if cmd.startswith("LOAD") and cmd[4:].isdigit():
            slot = int(cmd[4:]) - 1
            if 0 <= slot < SCENE_SLOT_COUNT:
                self.cfg["scene"]["selected"] = slot
                self.defer_action("scene_load", None, "串口命令：准备加载场景%d" % (slot + 1))
            return
        if cmd.startswith("STORE") and cmd[5:].isdigit():
            slot = int(cmd[5:]) - 1
            if 0 <= slot < SCENE_SLOT_COUNT:
                self.cfg["scene"]["selected"] = slot
                self.defer_action("scene_save", None, "串口命令：准备保存场景%d" % (slot + 1))
            return
        print("[串口命令] 未识别:", cmd)

    def handle_remote_binary(self, port_index, cmd, param):
        """二进制控制帧：55 10 CMD PARAM XOR 66。PARAM 通常从1开始编号。"""
        if cmd == 0x01:
            pages = {0: "home", 1: "yolo", 2: "color", 3: "shape", 4: "line", 5: "code",
                     6: "measure", 7: "scene", 8: "geometry", 9: "template", 10: "camera"}
            if param in pages:
                self.on_action("nav", pages[param])
        elif cmd == 0x02:
            self._select_yolo_model(int(param) - 1)
        elif cmd == 0x03:
            slot = int(param) - 1
            if 0 <= slot < SCENE_SLOT_COUNT:
                self.cfg["scene"]["selected"] = slot
                self.defer_action("scene_load", None, "串口命令：准备加载场景%d" % (slot + 1))
        elif cmd == 0x04:
            slot = int(param) - 1
            if 0 <= slot < SCENE_SLOT_COUNT:
                self.cfg["scene"]["selected"] = slot
                self.defer_action("scene_save", None, "串口命令：准备保存场景%d" % (slot + 1))
        elif cmd == 0x05:
            self.defer_action("save_cfg", None, "串口命令：准备保存参数")

    # --------------------- actions ---------------------------
    def confirm_pending(self, action, payload=None):
        if self.confirm_action is None:
            return False
        now = time.ticks_ms()
        if safe_ticks_diff(self.confirm_until, now) <= 0:
            self.clear_confirm()
            return False
        if str(self.confirm_action) != str(action):
            return False
        if payload is None:
            return True
        return repr(self.confirm_payload) == repr(payload)

    def clear_confirm(self):
        self.confirm_action = None
        self.confirm_payload = None
        self.confirm_until = 0
        self.confirm_armed_ms = 0

    def _confirm_name(self, action):
        names = {
            "reset_all": "恢复全部默认",
            "reset_shape": "复位形状参数",
            "line_clear": "清除巡线颜色",
            "yolo_unload": "卸载YOLO模型",
            "scene_save": "保存当前场景",
            "scene_delete": "删除当前场景",
            "template_delete": "删除当前模板",
            "measure_clear": "清除测量标定",
        }
        return names.get(str(action), str(action))

    def on_confirm_action(self, action, payload=None):
        now = time.ticks_ms()
        same = (self.confirm_action is not None and
                str(self.confirm_action) == str(action) and
                repr(self.confirm_payload) == repr(payload) and
                safe_ticks_diff(self.confirm_until, now) > 0)
        if same:
            # 庐山派3.1寸触摸屏的 event/track_id 当前不应作为可靠的长按依据。
            # 同一次物理按压若被空读拆成两段，也不能让“第一次确认+第二次确认”连续发生。
            if safe_ticks_diff(now, self.confirm_armed_ms) < TOUCH_CONFIRM_MIN_GAP_MS:
                self.notice("请松手后再次点击确认", COLOR_YELLOW, 1200)
                if self.ui is not None:
                    self.ui.invalidate()
                return
            self.clear_confirm()
            self.notice("已确认：" + self._confirm_name(action), COLOR_ORANGE, 900)
            self.on_action(action, payload)
            return
        self.confirm_action = str(action)
        self.confirm_payload = payload
        self.confirm_armed_ms = now
        self.confirm_until = now + TOUCH_CONFIRM_TIMEOUT_MS
        self.notice("再次点击确认：" + self._confirm_name(action), COLOR_YELLOW, 2200)
        if self.ui is not None:
            self.ui.invalidate()

    def defer_action(self, action, payload=None, notice_text=None):
        if self.deferred_busy or self.deferred_action is not None:
            self.notice("系统正在处理上一操作", COLOR_YELLOW, 1000)
            return False
        self.deferred_action = str(action)
        self.deferred_payload = payload
        # 至少先让主循环刷新一帧 UI/提示，再执行可能阻塞数百毫秒的操作。
        self.deferred_ready_ms = time.ticks_ms() + 70
        if notice_text:
            self.notice(notice_text, COLOR_CYAN, 900)
        return True

    def _service_deferred_action(self):
        if self.deferred_busy or self.deferred_action is None:
            return
        if self.touch_ctrl is not None and self.touch_ctrl.active:
            return
        if safe_ticks_diff(self.deferred_ready_ms, time.ticks_ms()) > 0:
            return
        action = self.deferred_action
        payload = self.deferred_payload
        self.deferred_action = None
        self.deferred_payload = None
        self.deferred_busy = True
        action_start_ms = time.ticks_ms()
        try:
            if action == "save_cfg":
                if self.store.save(self.cfg):
                    self.notice("参数已保存", COLOR_GREEN, 1200)
                else:
                    self.notice("参数保存失败", COLOR_RED, 1500)

            elif action == "load_cfg":
                self.pending_yolo_rebuild = False
                self.cancel_color_clear_confirm()
                self.color_engine.cancel_training()
                self.line_engine.cancel_training()
                self.yolo_engine.unload()
                self.template_engine.unload()
                self.serial_manager.deinit()
                self.cfg = self.store.load()
                self.shape_engine.prev_tracks = []
                self._apply_camera_direction()
                self.serial_manager.open_enabled()
                self.notice("参数已重新加载", COLOR_GREEN, 1400)

            elif action == "reset_all":
                self.pending_yolo_rebuild = False
                self.cancel_color_clear_confirm()
                self.color_engine.cancel_training()
                self.line_engine.cancel_training()
                self.yolo_engine.unload()
                self.template_engine.unload()
                self.serial_manager.deinit()
                self.cfg = default_config()
                self.shape_engine.prev_tracks = []
                for source_name in ("yolo", "color", "shape", "line", "code", "geometry", "template"):
                    try:
                        self.results.clear(source_name)
                    except Exception:
                        pass
                self._apply_camera_direction()
                # 真正“恢复默认”：立即覆盖旧配置，重启后也保持默认。
                self.store.save(self.cfg)
                self.notice("全部参数已恢复默认并保存", COLOR_ORANGE, 1800)

            elif action == "scene_load":
                self.apply_scene(int(self.cfg["scene"].get("selected", 0)))

            elif action == "scene_save":
                slot = int(self.cfg["scene"].get("selected", 0))
                entry_page = str(self.cfg["scene"].get("entry_page", "color"))
                if self.scene_manager.save(slot, entry_page):
                    self.notice("场景%d已保存：%s" % (slot + 1, SCENE_PAGE_NAMES.get(entry_page, entry_page)), COLOR_GREEN, 1500)
                else:
                    self.notice("场景保存失败", COLOR_RED, 1500)

            elif action == "scene_delete":
                slot = int(self.cfg["scene"].get("selected", 0))
                if self.scene_manager.delete(slot):
                    self.notice("场景%d已删除" % (slot + 1), COLOR_ORANGE, 1200)
                else:
                    self.notice("当前场景为空", COLOR_YELLOW, 1000)

            elif action == "camera_preset":
                old_hm = bool(self.cfg["camera"].get("hmirror", False))
                old_vf = bool(self.cfg["camera"].get("vflip", False))
                mode = str(payload)
                if mode == "hmirror":
                    new_hm, new_vf = True, False
                elif mode == "vflip":
                    new_hm, new_vf = False, True
                elif mode == "rotate180":
                    new_hm, new_vf = True, True
                else:
                    new_hm, new_vf = False, False
                self.cfg["camera"]["hmirror"] = new_hm
                self.cfg["camera"]["vflip"] = new_vf
                if not self._apply_camera_direction():
                    self.cfg["camera"]["hmirror"] = old_hm
                    self.cfg["camera"]["vflip"] = old_vf
                    try:
                        self.camera_control.apply(restart=True)
                    except Exception:
                        pass

            elif action == "camera_toggle":
                key = str(payload)
                old_value = bool(self.cfg["camera"].get(key, False))
                self.cfg["camera"][key] = not old_value
                if not self._apply_camera_direction():
                    self.cfg["camera"][key] = old_value
                    try:
                        self.camera_control.apply(restart=True)
                    except Exception:
                        pass

            elif action == "camera_apply":
                self._apply_camera_direction(force=True)

            elif action == "camera_reset":
                self.cfg["camera"]["hmirror"] = False
                self.cfg["camera"]["vflip"] = False
                self._apply_camera_direction()

            elif action == "serial_apply":
                idx = int(self.cfg["serial"].get("selected", 0))
                if self.cfg["serial"]["ports"][idx].get("enabled", False):
                    if self.serial_manager.apply_selected():
                        self.notice("串口参数已应用", COLOR_GREEN, 1000)
                    else:
                        self.notice("串口初始化失败，请检查引脚", COLOR_RED, 1800)
                else:
                    self.serial_manager.close_port(idx)
                    self.notice("当前串口未开启", COLOR_YELLOW, 1000)

            elif action == "yolo_unload":
                self.pending_yolo_rebuild = False
                self.yolo_engine.unload(manual=True)
                self.notice("YOLO已卸载", COLOR_ORANGE, 1200)

            elif action == "template_delete":
                if self.template_engine.delete_selected():
                    self.notice("当前模板已删除", COLOR_ORANGE, 1200)
                else:
                    self.notice("当前模板为空", COLOR_YELLOW, 1000)

            elif action == "release_yolo_heavy":
                if (RELEASE_YOLO_ON_HEAVY_PAGE and self.yolo_engine is not None and self.yolo_engine.ready):
                    self.yolo_engine.unload()
                    self.pending_yolo_rebuild = False
                    gc.collect()
                    print("[内存] 已在触摸结束后释放YOLO资源")

        except Exception as e:
            print("[DEFERRED ACTION]", action, e)
            self.notice("操作失败: " + str(e), COLOR_RED, 1800)
        finally:
            self.last_deferred_name = str(action)
            self.last_deferred_ms = max(0, safe_ticks_diff(time.ticks_ms(), action_start_ms))
            print("[DEFERRED DONE]", action, self.last_deferred_ms, "ms")
            self.deferred_busy = False
            if self.touch_ctrl is not None:
                self.touch_ctrl.force_rearm(TOUCH_REARM_MS + 180)
            if self.ui is not None:
                self.ui.invalidate()

    def on_action(self, action, payload=None):
        if self.cfg["system"].get("touch_debug", False):
            print("[ACTION]", action, payload)

        if self.confirm_action is not None and str(self.confirm_action) != str(action):
            self.clear_confirm()

        if action == "nav":
            old = self.page
            target_page = str(payload)
            if old in VISION_PAGE_OPTIONS:
                self.last_vision_page = old
            if target_page in VISION_PAGE_OPTIONS:
                self.last_vision_page = target_page
            # 从识别页/首页进入场景页时，默认把“加载后执行功能”设成最近使用的识别功能。
            # 用户仍可在场景页用“上个/下个功能”手动修改。
            if target_page == "scene" and self.last_vision_page in VISION_PAGE_OPTIONS:
                self.cfg.setdefault("scene", {})["entry_page"] = self.last_vision_page
            if old == "color" and payload != "color":
                self.color_engine.cancel_training()
            if old == "line" and payload != "line":
                self.line_engine.cancel_training()
            if old == "template" and payload != "template":
                # 模板常驻图离开页面后立即释放；再次进入会从SD卡加载。
                self.template_engine.unload()
            if old in ("shape", "color", "yolo", "line", "code", "geometry", "template") and old != payload:
                self.results.clear(old)

            # 几何/LSD/Hough/NCC 是传统视觉中内存压力最大的页面。
            # 页面切换时主动释放此刻不用的 KPU 模型资源；返回 YOLO 页会自动重建。
            if (RELEASE_YOLO_ON_HEAVY_PAGE and payload in ("geometry", "template") and
                    self.yolo_engine is not None and self.yolo_engine.ready):
                self.defer_action("release_yolo_heavy", None, "正在释放重算法页面所需资源")

            self.page = payload
            self.clear_result()

        elif action == "region_edit":
            if self.region_manager.start(str(payload), self.page):
                return

        elif action == "shape_mode":
            self.cfg["shape"]["mode"] = payload
            self.shape_engine.prev_tracks = []
        elif action == "shape_param":
            cfg = self.cfg["shape"]
            n = len(self.shape_param_specs())
            cfg["param_idx"] = (cfg.get("param_idx", 0) + payload) % n
        elif action == "shape_delta":
            self._shape_delta(payload)
        elif action == "reset_shape":
            self.cfg["shape"] = default_config()["shape"]
            self.shape_engine.prev_tracks = []
            self.notice("形状参数已恢复默认", COLOR_ORANGE, 1400)

        elif action == "color_profile_set":
            cfg = self.cfg["color"]
            self.color_engine.cancel_training()
            self.cancel_color_clear_confirm()
            cfg["current"] = int(clamp(int(payload), 0, len(cfg["profiles"]) - 1))
            p = cfg["profiles"][cfg["current"]]
            self.notice("当前%s：%s" % (p["name"], "独立阈值已加载" if p["trained"] else "空目标"),
                        COLOR_CYAN, 1000)
        elif action == "color_profile":
            # 兼容旧按钮；当前界面不再使用循环切换。
            cfg = self.cfg["color"]
            cfg["current"] = int(clamp(cfg["current"] + int(payload), 0, len(cfg["profiles"]) - 1))
        elif action == "color_param":
            cfg = self.cfg["color"]
            cfg["param_idx"] = (cfg.get("param_idx", 0) + payload) % 6
        elif action == "color_delta":
            self._color_delta(payload)
        elif action == "color_toggle_enable":
            cfg = self.cfg["color"]
            p = cfg["profiles"][cfg["current"]]
            if not p["trained"]:
                self.notice("空目标不能开启检测，请先训练", COLOR_YELLOW, 1700)
            else:
                p["enabled"] = not p["enabled"]
                if not p["enabled"]:
                    p["auto"] = False
        elif action == "color_toggle_auto":
            cfg = self.cfg["color"]
            p = cfg["profiles"][cfg["current"]]
            if not p["trained"]:
                self.notice("空目标不能自适应，请先训练", COLOR_YELLOW, 1700)
            else:
                if not p["enabled"]:
                    p["enabled"] = True
                p["auto"] = not p["auto"]
        elif action == "color_train":
            self.color_engine.arm_training()
        elif action == "color_clear_current":
            idx = int(self.cfg["color"]["current"])
            now = time.ticks_ms()
            if (self.color_clear_confirm_idx == idx and
                    safe_ticks_diff(self.color_clear_confirm_until, now) > 0):
                # 第二次单击：只重置当前目标 profile。其它 3 个 profile 完全不动。
                self.color_engine.reset_current()
                self.cancel_color_clear_confirm()
                self.notice("目标%d阈值已单独清空" % (idx + 1), COLOR_ORANGE, 1700)
            else:
                # 第一次单击只进入确认态，不修改任何阈值。
                self.color_clear_confirm_idx = idx
                self.color_clear_confirm_until = now + 2600
                self.notice("再次点击确认：只清空目标%d" % (idx + 1), COLOR_YELLOW, 2200)

        elif action == "line_mode":
            self.cfg["line"]["mode"] = payload
            self.line_engine.reset_track()
        elif action == "line_train":
            self.line_engine.arm_training()
        elif action == "line_clear":
            self.line_engine.clear_training()
            self.notice("巡线颜色已清空", COLOR_ORANGE, 1400)
        elif action == "line_toggle":
            self.cfg["line"][payload] = not self.cfg["line"][payload]
        elif action == "line_param":
            cfg = self.cfg["line"]
            n = len(self.line_param_specs())
            cfg["param_idx"] = (cfg.get("param_idx", 0) + payload) % n
        elif action == "line_delta":
            self._line_delta(payload)

        elif action == "yolo_model_set":
            self._select_yolo_model(payload)
        elif action == "yolo_toggle":
            self.cfg["yolo"][payload] = not self.cfg["yolo"].get(payload, False)
        elif action == "yolo_conf":
            self.cfg["yolo"]["conf"] = clamp(self.cfg["yolo"]["conf"] + payload, 0.05, 0.95)
            remember_yolo_runtime(self.cfg["yolo"])
            self.yolo_engine.needs_rebuild = True
        elif action == "yolo_nms":
            if normalize_yolo_family(self.cfg["yolo"].get("family", "yolov8")) != "yolo26":
                self.cfg["yolo"]["nms"] = clamp(self.cfg["yolo"]["nms"] + payload, 0.05, 0.95)
                remember_yolo_runtime(self.cfg["yolo"])
                self.yolo_engine.needs_rebuild = True
        elif action == "yolo_apply":
            self.yolo_engine.last_failed_index = -1
            self.yolo_engine.manual_unloaded = False
            self.pending_yolo_rebuild = True
            self.pending_yolo_ready_ms = time.ticks_ms() + 80
            self.notice("准备重载当前模型", COLOR_CYAN, 800)
        elif action == "yolo_unload":
            self.defer_action("yolo_unload", None, "准备卸载YOLO")

        elif action == "serial_select":
            self.cfg["serial"]["selected"] = int(clamp(int(payload), 0, len(SERIAL_PORT_PROFILES) - 1))
        elif action == "serial_toggle_enable":
            idx = int(self.cfg["serial"].get("selected", 0))
            port = self.cfg["serial"]["ports"][idx]
            port["enabled"] = not bool(port.get("enabled", False))
            if not port["enabled"]:
                self.serial_manager.close_port(idx)
                self.notice("%s已关闭" % SERIAL_PORT_PROFILES[idx]["name"], COLOR_ORANGE, 1000)
            else:
                self.notice("%s已开启" % SERIAL_PORT_PROFILES[idx]["name"], COLOR_GREEN, 1000)
        elif action == "serial_cycle":
            self._serial_cycle(str(payload))
        elif action == "serial_toggle_rx":
            idx = int(self.cfg["serial"].get("selected", 0))
            port = self.cfg["serial"]["ports"][idx]
            port["rx_control"] = not bool(port.get("rx_control", False))
            self.notice("%s遥控%s" % (SERIAL_PORT_PROFILES[idx]["name"], "已开启" if port["rx_control"] else "已关闭"),
                        COLOR_GREEN if port["rx_control"] else COLOR_ORANGE, 1000)
        elif action == "serial_apply":
            self.defer_action("serial_apply", None, "准备应用串口参数")

        elif action == "geometry_mode":
            if payload in GEOMETRY_MODE_OPTIONS:
                self.cfg["geometry"]["mode"] = payload
                self.cfg["geometry"]["param_idx"] = 0
        elif action == "geometry_param":
            cfg = self.cfg["geometry"]
            n = len(self.geometry_param_specs())
            cfg["param_idx"] = (int(cfg.get("param_idx", 0)) + int(payload)) % n
        elif action == "geometry_delta":
            self._geometry_delta(payload)
        elif action == "geometry_toggle":
            key = str(payload)
            self.cfg["geometry"][key] = not bool(self.cfg["geometry"].get(key, False))

        elif action == "template_select":
            self.cfg["template"]["selected"] = int(clamp(int(payload), 0, TEMPLATE_SLOT_COUNT-1))
            self.template_engine.unload()
        elif action == "template_threshold":
            self.cfg["template"]["threshold"] = float(clamp(float(self.cfg["template"].get("threshold",0.72)) + float(payload), 0.10, 0.99))
        elif action == "template_step":
            self.cfg["template"]["step"] = int(clamp(int(self.cfg["template"].get("step",2)) + int(payload), 1, 8))
        elif action == "template_search_mode":
            self.cfg["template"]["search"] = "fast" if self.cfg["template"].get("search") != "fast" else "exhaustive"
        elif action == "template_toggle":
            key = str(payload)
            self.cfg["template"][key] = not bool(self.cfg["template"].get(key, False))
        elif action == "template_capture":
            if not self.region_manager.enabled("template_capture"):
                self.notice("请先点击“框选模板”选择模板区域", COLOR_YELLOW, 1800)
            else:
                self.pending_template_capture = True
                self.notice("准备保存当前模板", COLOR_CYAN, 800)
        elif action == "template_delete":
            self.defer_action("template_delete", None, "准备删除模板")

        elif action == "camera_preset":
            self.defer_action("camera_preset", payload, "准备切换相机方向")
        elif action == "camera_toggle":
            self.defer_action("camera_toggle", payload, "准备切换相机方向")
        elif action == "camera_apply":
            self.defer_action("camera_apply", None, "准备重新应用相机方向")
        elif action == "camera_reset":
            self.defer_action("camera_reset", None, "准备恢复正常方向")

        elif action == "code_mode":
            self.cfg["code"]["mode"] = payload
            self.barcode_pipeline_failed = False

            # 进入一维条码即固定使用官方原生灰度链路。
            if payload == "barcode":
                self.cfg["code"]["barcode_highres"] = True

        elif action == "apriltag_family_cycle":
            cur = str(self.cfg["code"].get("apriltag_family", "all")).lower()
            try:
                idx = APRILTAG_FAMILY_OPTIONS.index(cur)
            except Exception:
                idx = 0
            self.cfg["code"]["apriltag_family"] = APRILTAG_FAMILY_OPTIONS[
                (idx + 1) % len(APRILTAG_FAMILY_OPTIONS)
            ]
            self.notice(
                "AprilTag家族: " +
                APRILTAG_FAMILY_UI_NAMES.get(
                    self.cfg["code"]["apriltag_family"], "全部6类"
                ),
                COLOR_CYAN
            )

        elif action == "dm_effort":
            self.cfg["code"]["dm_effort"] = int(clamp(self.cfg["code"]["dm_effort"] + payload, 160, 240))

        elif action == "measure_toggle":
            key = str(payload)
            self.cfg["measurement"][key] = not bool(self.cfg["measurement"].get(key, False))
        elif action == "measure_source_set":
            value = str(payload)
            if value in MEASURE_SOURCE_OPTIONS:
                self.cfg["measurement"]["source"] = value
        elif action == "measure_open_source":
            self._open_measure_source()
        elif action == "measure_cycle":
            self._measure_cycle(str(payload))
        elif action == "measure_known":
            cfg = self.cfg["measurement"]
            cfg["known_mm"] = float(clamp(float(cfg.get("known_mm", 100.0)) + float(payload), 1.0, 10000.0))
        elif action == "measure_calibrate":
            if self.calibration_manager.start(self.page):
                return
        elif action == "measure_clear":
            self.measurement_manager.reset_calibration()
            self.notice("测量标定已清除", COLOR_ORANGE, 1200)

        elif action == "scene_select":
            self.cfg["scene"]["selected"] = int(clamp(int(payload), 0, SCENE_SLOT_COUNT - 1))
        elif action == "scene_page_cycle":
            self._scene_cycle_page(int(payload))
        elif action == "scene_use_last_page":
            last_page = self.last_vision_page if self.last_vision_page in VISION_PAGE_OPTIONS else "color"
            self.cfg["scene"]["entry_page"] = last_page
            self.notice("已绑定: " + SCENE_PAGE_NAMES.get(last_page, last_page), COLOR_CYAN, 900)
        elif action == "scene_load":
            self.defer_action("scene_load", None, "准备加载当前场景")
        elif action == "scene_save":
            self.defer_action("scene_save", None, "准备保存当前场景")
        elif action == "scene_delete":
            self.defer_action("scene_delete", None, "准备删除当前场景")

        elif action == "toggle_system":
            self.cfg["system"][payload] = not self.cfg["system"].get(payload, False)
        elif action == "save_cfg":
            self.defer_action("save_cfg", None, "准备保存参数")
        elif action == "load_cfg":
            self.defer_action("load_cfg", None, "准备重新加载参数")
        elif action == "reset_all":
            self.defer_action("reset_all", None, "准备恢复全部默认")

        # 只标记 UI 失效；真正绘制交给主循环，避免触摸回调被 Display.show_image 阻塞。
        self.ui.invalidate()

    # --------------------- loop ------------------------------
    def run(self):
        self.init_hardware()
        print("[RUN] start")
        while True:
            os.exitpoint()
            self.fps_clock.tick()
            self.frame_counter += 1

            self.touch_ctrl.poll()

            if self.confirm_action is not None:
                now_confirm = time.ticks_ms()
                if safe_ticks_diff(self.confirm_until, now_confirm) <= 0:
                    self.clear_confirm()
                    self.ui.invalidate()

            # 两点标定/识别区域编辑期间暂停视觉算法，底层视频层保持实时显示。
            if self.calibration_manager is not None and self.calibration_manager.active:
                self.serial_manager.service()
                time.sleep_ms(6)
                continue

            if self.region_manager is not None and self.region_manager.active:
                self.serial_manager.service()
                time.sleep_ms(6)
                continue

            # 耗时操作都放到完整触摸手势结束以后执行，避免一次点击被阻塞。
            self._service_deferred_action()

            # 一维条码需要官方原生 640x480 GRAYSCALE；
            # 离开条码页则自动恢复 YOLO 的 RGBP888 通道。
            self._service_barcode_native_pipeline()

            self._service_pending_yolo_rebuild()
            self._service_pending_template_capture()

            # “清空当前目标”的二次确认超时后自动恢复普通按钮状态。
            if self.color_clear_confirm_idx >= 0:
                now = time.ticks_ms()
                if safe_ticks_diff(self.color_clear_confirm_until, now) <= 0:
                    self.cancel_color_clear_confirm()
                    self.ui.invalidate()

            self.ui.render()

            page = self.page
            if page == "shape":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
                self.shape_engine.process(img, self.overlay_img)
            elif page == "color":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
                self.color_engine.process(img, self.overlay_img)
            elif page == "line":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
                self.line_engine.process(img, self.overlay_img)
            elif page == "code":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
                self.code_engine.process(img, self.overlay_img)
            elif page == "yolo":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_2)
                self.yolo_engine.process(img, self.overlay_img)
            elif page == "geometry":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
                self.geometry_engine.process(img, self.overlay_img)
            elif page == "template":
                img = self.sensor.snapshot(chn=CAM_CHN_ID_1)
                self.template_engine.process(img, self.overlay_img)
            else:
                # 首页/设置/串口页不跑不必要的视觉算法。
                time.sleep_ms(6)

            # 串口与视觉算法同循环服务：识别完成后立即使用统一结果发送，
            # 不在各 Engine 内部直接 uart.write，避免算法和通信互相耦合。
            self.serial_manager.service()

            # 不每帧 GC；避免周期性大卡顿。
            if self.frame_counter % 90 == 0:
                gc.collect()

    def deinit(self):
        print("[DEINIT] start")
        try:
            if self.yolo_engine is not None:
                self.yolo_engine.unload()
        except Exception:
            pass
        try:
            if self.template_engine is not None:
                self.template_engine.unload()
        except Exception:
            pass
        try:
            if self.serial_manager is not None:
                self.serial_manager.deinit()
        except Exception:
            pass
        try:
            if self.touch is not None:
                self.touch.deinit()
        except Exception:
            pass
        try:
            if self.sensor is not None:
                self.sensor.stop()
        except Exception:
            pass
        try:
            Display.deinit()
        except Exception:
            pass
        time.sleep_ms(80)
        try:
            MediaManager.deinit()
        except Exception:
            pass
        gc.collect()
        print("[DEINIT] done")


def main():
    app = VisionWorkbench()
    try:
        app.run()
    except KeyboardInterrupt:
        print("用户停止")
    except BaseException as e:
        # 不使用部分旧固件不存在的 sys.print_exception。
        print("主程序异常:", e)
    finally:
        app.deinit()


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
