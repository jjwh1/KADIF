# save_augmented.py
import os, math, random, argparse
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageOps
import re
import cv2
import pywt

# ---------- 공용 유틸 ----------
def _split_list(s):
    return [x.strip().lower() for x in re.split(r'[+,]', s) if x.strip()]

# def parse_aug_argument(s, default_sev=1):
#     s = s.strip().lower()
#
#     def parse_token(tok):
#         m = re.match(r'^([a-z_]+)\s*(?:[:@]\s*(\d+))?$', tok.strip())
#         if not m:
#             raise ValueError(f"잘못된 aug 토큰: {tok}")
#         name = m.group(1)
#         sev  = int(m.group(2)) if m.group(2) else default_sev
#         return (name, max(1, min(5, sev)))  # 1~5로 클램프
#
#     # all 또는 all:K → 각각 따로 저장
#     m_all = re.match(r'^all(?:[:@]\s*(\d+))?$', s)
#     if m_all:
#         sev_all = int(m_all.group(1)) if m_all.group(1) else default_sev
#         sev_all = max(1, min(5, sev_all))
#         return [
#             [("haze", sev_all)],
#             [("rain", sev_all)],
#             [("raindrop", sev_all)],
#             [("lowlight", sev_all)],
#             [("overbright", sev_all)],  # ← 추가
#         ]
#
#     # 연속 적용(체인)
#     if "+" in s:
#         return [[parse_token(t) for t in s.split("+")]]
#
#     # 각각 저장
#     if "," in s:
#         return [[parse_token(t)] for t in s.split(",")]
#
#     # 단일
#     return [[parse_token(s)]]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def pil_gamma(img, gamma):
    # Pillow 버전 상관없이 동작하는 감마
    lut = [min(255, int((i/255.0) ** gamma * 255 + 0.5)) for i in range(256)]
    if img.mode == "RGB":
        return img.point(lut * 3)
    elif img.mode == "L":
        return img.point(lut)
    else:
        return img.convert("RGB").point(lut * 3)

# ---------- 1) Haze/Fog ----------
class AddHazeTV:
    def __init__(self, beta=(0.6, 1.6), A=(0.85, 0.98), blur_ratio=0.05):
        self.beta, self.A, self.blur_ratio = beta, A, blur_ratio
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        beta = np.random.uniform(*self.beta)
        A = np.random.uniform(*self.A)
        gray = img.convert("L")
        depth = ImageOps.invert(gray).filter(
            ImageFilter.GaussianBlur(radius=max(1, int(min(w, h) * self.blur_ratio))))
        depth = np.asarray(depth, np.float32) / 255.0
        t = np.exp(-beta * depth)[..., None]
        I = np.asarray(img, np.float32) / 255.0
        out = I * t + A * (1.0 - t)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))

# ---------- 2) Rain Streaks ----------
class AddRainStreaksTV:
    def __init__(self, density=(200, 600), length=(10, 22), angle=(-15, 15),
                 alpha=(0.15, 0.35), blur=1.2, width=(1, 2), color=(225, 225, 225)):
        self.density, self.length, self.angle = density, length, angle
        self.alpha, self.blur = alpha, blur
        self.width = width
        self.color = color  # RGB 값

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        den = np.random.randint(*self.density)
        L   = np.random.randint(*self.length)
        ang = np.random.uniform(*self.angle)
        a   = np.random.uniform(*self.alpha)
        thick = np.random.randint(self.width[0], self.width[1] + 1)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        dx = int(L * math.cos(math.radians(ang)))
        dy = int(L * math.sin(math.radians(ang)))

        col = (*self.color, 255)  # 불투명 라인 후 전체 알파로 블렌드
        for _ in range(den):
            x, y = random.randrange(w), random.randrange(h)
            draw.line([(x, y), (x + dx, y + dy)], fill=col, width=thick)

        overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur))
        out = Image.alpha_composite(
            img.convert("RGBA"),
            Image.blend(Image.new("RGBA", (w, h), (0,0,0,0)), overlay, a)
        )
        return out.convert("RGB")

# ---------- 3) Raindrops(근사) ----------
class AddRaindropsTV:
    def __init__(self, num=(15, 60), radius=(5, 22), alpha=(0.25, 0.55), blur=(1.5, 3.5)):
        self.num, self.radius, self.alpha, self.blur = num, radius, alpha, blur
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        n = np.random.randint(*self.num)
        a = np.random.uniform(*self.alpha)
        br = np.random.uniform(*self.blur)
        drop = Image.new("L", (w, h), 0)
        rim  = Image.new("L", (w, h), 0)
        d1, d2 = ImageDraw.Draw(drop), ImageDraw.Draw(rim)
        for _ in range(n):
            r = np.random.randint(*self.radius)
            x, y = np.random.randint(-r, w + r), np.random.randint(-r, h + r)
            box = (x - r, y - r, x + r, y + r)
            d1.ellipse(box, fill=200)                         # 내부 디밍
            d2.ellipse(box, outline=255, width=max(1, r // 6))# 림 하이라이트
        drop = drop.filter(ImageFilter.GaussianBlur(br))
        rim  = rim.filter(ImageFilter.GaussianBlur(max(0.5, br / 2)))
        base = img.convert("RGBA")
        dark = ImageEnhance.Brightness(img).enhance(0.8)
        bright = ImageEnhance.Brightness(img).enhance(1.2)
        base = Image.composite(dark, base, drop)
        base = Image.composite(bright, base, rim)
        return Image.blend(img, base.convert("RGB"), a)

# ---------- 4) Low-light ----------
class FastRetinexLowLightTV:
    """
    PIL.Image -> PIL.Image
    HSV의 V 채널만 대상으로 DWT(LL) + MSR + 역DWT.
    """
    def __init__(
        self,
        sigmas=(8, 40, 80),      # LL 기준 가우시안 시그마들 (원본 환산은 대략 ×2)
        weights=None,            # None이면 균등
        levels=1,                # DWT 레벨(1 또는 2 추천)
        alpha=0.6,               # LL과 MSR(LL)의 블렌딩 계수
        clip_percentiles=(1,99), # MSR 결과의 퍼센타일 클리핑
        wavelet='haar',
    ):
        self.sigmas = tuple(sigmas)
        self.weights = (
            np.ones(len(sigmas), dtype=np.float32) / len(sigmas)
            if weights is None else np.array(weights, dtype=np.float32)
        )
        assert len(self.weights) == len(self.sigmas)
        self.levels = int(levels)
        self.alpha = float(alpha)
        self.clip = clip_percentiles
        self.wavelet = wavelet

    # ---------- 내부 유틸 ----------
    def _dwt_levels(self, channel, levels):
        coeffs_stack = []
        LL = channel
        for _ in range(levels):
            LL, (LH, HL, HH) = pywt.dwt2(LL, self.wavelet, mode='symmetric')
            coeffs_stack.append((LH, HL, HH))
        return LL, coeffs_stack

    def _idwt_levels(self, LL, coeffs_stack):
        for (LH, HL, HH) in reversed(coeffs_stack):
            LL = pywt.idwt2((LL, (LH, HL, HH)), self.wavelet, mode='symmetric')
        return LL

    def _msr(self, img_u8, sigmas, weights):
        # 입력은 uint8 (LL 채널)
        x = img_u8.astype(np.float32) / 255.0
        logx = np.log1p(x)
        out = np.zeros_like(x, dtype=np.float32)
        for w, s in zip(weights, sigmas):
            blur = cv2.GaussianBlur(x, (0,0), float(s))
            out += w * (logx - np.log1p(blur))
        # 퍼센타일 기반 정규화 (깜빡임/들쭉날쭉 완화)
        lo, hi = np.percentile(out, self.clip)
        if hi <= lo:  # 드문 예외 대비
            lo, hi = out.min(), out.max()
        out = (out - lo) / max(1e-6, (hi - lo))
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    # ---------- 호출 ----------
    def __call__(self, img: Image.Image) -> Image.Image:
        assert isinstance(img, Image.Image), "PIL.Image 입력을 기대합니다."
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # RGB -> HSV (OpenCV는 BGR 기본이므로 주의)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # DWT 레벨 분해 (V 채널)
        LL, coeffs_stack = self._dwt_levels(v, self.levels)

        # MSR on LL
        LL_msr = self._msr(LL, self.sigmas, self.weights)

        # 블렌딩으로 과도한 대비 억제
        LL_blend = ((1.0 - self.alpha) * LL.astype(np.float32) +
                    self.alpha * LL_msr.astype(np.float32))
        LL_blend = np.clip(LL_blend, 0, 255).astype(np.uint8)

        # 역DWT로 V 복원
        v_enh = self._idwt_levels(LL_blend, coeffs_stack)
        v_enh = np.clip(v_enh, 0, 255).astype(np.uint8)

        hsv_enh = cv2.merge([h, s, v_enh])
        rgb_enh = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb_enh, mode="RGB")



# ---------- 5) Over-brighten (Curve + CLAHE, non-hazy) ----------
class OverBrightCurveTV:
    """
    Lab 공간 L 채널에 CLAHE 후, 섀도/미드톤만 올리는 톤 커브.
    y = x + s*(1-x)^2 + m*x*(1-x)  (x,y ∈ [0,1])
    - s: shadows lift, m: midtone lift
    뿌연 느낌 없이 어두운 영역 위주로 밝힘.
    """
    def __init__(self,
                 clip=2.0, tile=8,      # CLAHE 파라미터
                 s=0.25, m=0.12,        # 톤 커브 강도
                 sat=1.04,              # 채도 약간 보존
                 sharpen=0.0):          # 0이면 샤픈 끔 (권장: 0~0.4)
        self.clip = float(clip)
        self.tile = int(tile)
        self.s = float(s)
        self.m = float(m)
        self.sat = float(sat)
        self.sharpen = float(sharpen)

    def _tone_curve(self, L01):
        s, m = self.s, self.m
        y = L01 + s * (1.0 - L01) ** 2 + m * L01 * (1.0 - L01)
        return np.clip(y, 0.0, 1.0)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)

        # 1) CLAHE (로컬 대비 확보)
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile))
        Lc = clahe.apply(L)

        # 2) 톤 커브 (섀도/미드톤 위주 리프트)
        L01 = np.clip(Lc.astype(np.float32) / 255.0, 0.0, 1.0)
        Lt  = (self._tone_curve(L01) * 255.0).astype(np.uint8)

        # 3) 색 보정 (채도 약간 복원)
        a = 128 + (a.astype(np.float32) - 128) * self.sat
        b = 128 + (b.astype(np.float32) - 128) * self.sat
        lab_out = cv2.merge([Lt, np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)])
        out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

        # 4) (선택) 언샤프마스크
        if self.sharpen > 1e-6:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out  = cv2.addWeighted(out, 1 + self.sharpen, blur, -self.sharpen, 0)

        return Image.fromarray(out, mode="RGB")


# ---------- severity → 파라미터 매핑 ----------
def build_transform(aug: str, severity: int):
    s = max(1, min(5, severity))
    if aug == "haze":
        presets = [
            dict(beta=(0.4,0.8),  A=(0.85,0.92), blur_ratio=0.03),
            dict(beta=(0.6,1.2),  A=(0.86,0.94), blur_ratio=0.04),
            dict(beta=(0.4,0.8),  A=(0.85,0.92), blur_ratio=0.03),
            dict(beta=(0.6,1.2),  A=(0.86,0.94), blur_ratio=0.04),
            dict(beta=(0.4,0.8),  A=(0.85,0.92), blur_ratio=0.03),
        ][s-1]
        return AddHazeTV(**presets)
    elif aug == "rain":
        presets = [
            # s=1
            dict(density=(700, 1100), length=(18, 30), angle=(-20, 20),
                 alpha=(0.28, 0.40), blur=1.6, width=(2, 3), color=(235, 235, 235)),
            # s=2
            dict(density=(1000, 1500), length=(20, 34), angle=(-22, 22),
                 alpha=(0.30, 0.45), blur=1.7, width=(2, 3), color=(235, 235, 235)),
            # s=3
            dict(density=(1300, 1900), length=(22, 36), angle=(-24, 24),
                 alpha=(0.32, 0.48), blur=1.8, width=(2, 4), color=(238, 238, 238)),
            # s=4 (3과 동일)
            dict(density=(1300, 1900), length=(22, 36), angle=(-24, 24),
                 alpha=(0.32, 0.48), blur=1.8, width=(2, 4), color=(238, 238, 238)),
            # s=5
            dict(density=(1600, 2300), length=(24, 38), angle=(-26, 26),
                 alpha=(0.34, 0.50), blur=1.9, width=(3, 4), color=(240, 240, 240)),
        ][s - 1]
        return AddRainStreaksTV(**presets)
    elif aug == "raindrop":
        presets = [
            dict(num=(20,50), radius=(7,20),  alpha=(0.28,0.45), blur=(1.8,3.0)),
            dict(num=(30,80), radius=(8,24),  alpha=(0.32,0.50), blur=(2.0,3.2)),
            dict(num=(20,50), radius=(7,20),  alpha=(0.28,0.45), blur=(1.8,3.0)),
            dict(num=(30,80), radius=(8,24),  alpha=(0.32,0.50), blur=(2.0,3.2)),
            dict(num=(50,120),radius=(10,30), alpha=(0.35,0.55), blur=(2.2,3.5)),
        ][s-1]
        return AddRaindropsTV(**presets)
    elif aug == "low_light":
        presets = [
            dict(sigmas=(6, 24, 48), weights=None, levels=1, alpha=0.45, clip_percentiles=(2, 98)),
            dict(sigmas=(8, 32, 64), weights=None, levels=1, alpha=0.55, clip_percentiles=(2, 98)),
            dict(sigmas=(10, 40, 80), weights=None, levels=1, alpha=0.60, clip_percentiles=(1, 99)),
            dict(sigmas=(10, 40, 80), weights=None, levels=1, alpha=0.60, clip_percentiles=(1, 99)),
            dict(sigmas=(12, 48, 96), weights=None, levels=2, alpha=0.65, clip_percentiles=(1, 99)),
        ][s - 1]
        return FastRetinexLowLightTV(**presets)
    elif aug == "overbright":  # 새 방식
        presets = [
            dict(clip=1.8, tile=8, s=0.22, m=0.10, sat=1.03, sharpen=0.05),
            dict(clip=1.8, tile=8, s=0.22, m=0.10, sat=1.03, sharpen=0.05),
            dict(clip=2.0, tile=8, s=0.26, m=0.12, sat=1.04, sharpen=0.08),
            dict(clip=2.2, tile=8, s=0.32, m=0.14, sat=1.05, sharpen=0.10),
            dict(clip=2.2, tile=8, s=0.32, m=0.14, sat=1.05, sharpen=0.10),
        ][s - 1]
        return OverBrightCurveTV(**presets)
    else:
        raise ValueError(f"Unknown aug: {aug}")
