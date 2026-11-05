"""Load Video Batch From Dir  --  video version of "Load Image Batch From Dir (Inspire)".

Same inputs and behaviour as the Inspire node, for video files:
  directory        folder to read
  video_load_cap   load at most this many files (0 = no limit)          [image_load_cap]
  start_index      skip this many files first (-1 = the last file)      [start_index]
  load_always      re-run every time instead of only when inputs change [load_always]
  sort_method      None / Alphabetical / Numerical / Datetime, ASC or DESC
  frame_load_cap   load at most this many frames from each file (0 = no limit)
  force_rate       play every file at this frame rate (0 = each file's own)
  resize_to / width / height / fit_mode / resize_method   optional resizing (see README)

Outputs: IMAGE (every frame of every file, one after another), AUDIO and INT (number of files loaded),
like IMAGE / MASK / INT.

One image batch cannot hold mixed frame sizes, so later files are brought to the first file's size unless a
size is set. Audio is converted to the first file's sample rate and channel count, and each file's sound is
trimmed or padded to the length of its own frames, so picture and sound stay lined up across the join; a file
with no audio contributes silence.

Reading uses PyAV, ffmpeg or OpenCV, whichever is installed.

Adapted from ComfyUI-Inspire-Pack (image_util.py) by ltdrdata, GPL-3.0.
Drop this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import torch

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ogv")

SORT_METHODS = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)",
]


def _first_number(s):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else float("inf")


def _sort_by(items, base_path=".", method=None):
    def full(x):
        return os.path.join(base_path, x)

    def stamp(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("-inf")

    if method == "Alphabetical (ASC)":
        return sorted(items)
    if method == "Alphabetical (DESC)":
        return sorted(items, reverse=True)
    if method == "Numerical (ASC)":
        return sorted(items, key=lambda x: _first_number(os.path.splitext(x)[0]))
    if method == "Numerical (DESC)":
        return sorted(items, key=lambda x: _first_number(os.path.splitext(x)[0]), reverse=True)
    if method == "Datetime (ASC)":
        return sorted(items, key=lambda x: stamp(full(x)))
    if method == "Datetime (DESC)":
        return sorted(items, key=lambda x: stamp(full(x)), reverse=True)
    return items


# --------------------------------------------------------------------------- tools

def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _probe(path):
    """-> dict with fps, width, height, sample_rate, channels, has_audio (missing keys = unknown)."""
    info = {}
    try:
        import av
        with av.open(path) as container:
            v = next((s for s in container.streams if s.type == "video"), None)
            a = next((s for s in container.streams if s.type == "audio"), None)
            if v is not None:
                info["width"], info["height"] = int(v.codec_context.width), int(v.codec_context.height)
                rate = v.average_rate or v.guessed_rate
                if rate:
                    info["fps"] = float(rate)
            info["has_audio"] = a is not None
            if a is not None:
                info["sample_rate"] = int(a.codec_context.sample_rate)
                info["channels"] = max(int(getattr(a.codec_context, "channels", 2) or 2), 1)
        return info
    except Exception:
        pass
    probe = shutil.which("ffprobe")
    if probe:
        try:
            out = subprocess.run(
                [probe, "-v", "quiet", "-print_format", "json", "-show_streams", path],
                capture_output=True, text=True, check=True).stdout
            streams = json.loads(out).get("streams", [])
            v = next((s for s in streams if s.get("codec_type") == "video"), None)
            a = next((s for s in streams if s.get("codec_type") == "audio"), None)
            if v:
                info["width"], info["height"] = int(v["width"]), int(v["height"])
                num, _, den = str(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
                if float(den or 1) != 0 and float(num) > 0:
                    info["fps"] = float(num) / float(den or 1)
            info["has_audio"] = a is not None
            if a:
                info["sample_rate"] = int(a.get("sample_rate", 44100))
                info["channels"] = max(int(a.get("channels", 2)), 1)
            return info
        except Exception:
            pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps > 0:
                info["fps"] = fps
        cap.release()
    except Exception:
        pass
    return info


def _frames_av(path, cap):
    import av
    frames = []
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise RuntimeError("no video stream")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if cap and len(frames) >= cap:
                break
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames)


def _frames_ffmpeg(path, cap, width, height):
    exe = _ffmpeg_exe()
    if not exe or not width or not height:
        raise RuntimeError("ffmpeg or frame size unavailable")
    cmd = [exe, "-v", "error", "-i", path]
    if cap:
        cmd += ["-frames:v", str(int(cap))]
    cmd += ["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-400:])
    per = int(width) * int(height) * 3
    n = len(proc.stdout) // per
    if n == 0:
        raise RuntimeError("no frames decoded")
    return np.frombuffer(proc.stdout[: n * per], np.uint8).reshape(n, int(height), int(width), 3)


def _frames_cv2(path, cap):
    import cv2
    frames = []
    reader = cv2.VideoCapture(path)
    if not reader.isOpened():
        raise RuntimeError("OpenCV could not open the file")
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1].copy())
        if cap and len(frames) >= cap:
            break
    reader.release()
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames)


def _read_frames(path, cap, info):
    """-> uint8 [N, H, W, 3]."""
    errors = []
    for name, fn in (("PyAV", lambda: _frames_av(path, cap)),
                     ("ffmpeg", lambda: _frames_ffmpeg(path, cap, info.get("width"), info.get("height"))),
                     ("OpenCV", lambda: _frames_cv2(path, cap))):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"Could not read frames from '{os.path.basename(path)}'. " + " | ".join(errors))


def _audio_av(path):
    import av
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("no audio stream")
        sr = int(stream.codec_context.sample_rate)
        chunks = []
        for frame in container.decode(stream):
            a = frame.to_ndarray()
            if a.dtype.kind in "iu":
                a = a.astype(np.float32) / float(np.iinfo(a.dtype).max + 1)
            else:
                a = a.astype(np.float32)
            ch = max(len(frame.layout.channels), 1)
            if a.shape[0] == 1 and ch > 1:
                a = a.reshape(-1, ch).T
            chunks.append(a)
    if not chunks:
        raise RuntimeError("no audio decoded")
    return np.concatenate(chunks, axis=1), sr


def _audio_ffmpeg(path, sr, channels):
    exe = _ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg unavailable")
    cmd = [exe, "-v", "error", "-i", path, "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", str(int(channels)), "-ar", str(int(sr)), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-400:] or "no audio decoded")
    data = np.frombuffer(proc.stdout, np.float32)
    n = len(data) // int(channels)
    return data[: n * int(channels)].reshape(n, int(channels)).T.copy(), int(sr)


def _read_audio(path, info, want_sr, want_ch):
    """-> (float32 [C, T], sample_rate), or None when the file has no audio."""
    if info.get("has_audio") is False:
        return None
    try:
        return _audio_av(path)
    except Exception:
        pass
    try:
        return _audio_ffmpeg(path, info.get("sample_rate") or want_sr or 44100,
                             info.get("channels") or want_ch or 2)
    except Exception:
        return None


def _match_channels(wav, channels):
    c = wav.shape[0]
    if c == channels:
        return wav
    if channels == 1:
        return wav.mean(0, keepdim=True)
    if c == 1:
        return wav.repeat(channels, 1)
    if c > channels:
        return wav[:channels]
    return torch.cat([wav, wav[-1:].repeat(channels - c, 1)], 0)


def _resample(wav, sr_from, sr_to):
    if sr_from == sr_to:
        return wav
    try:
        import torchaudio
        return torchaudio.functional.resample(wav, sr_from, sr_to)
    except Exception:
        n = max(int(round(wav.shape[1] * sr_to / float(sr_from))), 1)
        return torch.nn.functional.interpolate(wav[None], size=n, mode="linear", align_corners=False)[0]


RESIZE_METHODS = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]
FIT_MODES = ["fit (pad)", "fill (crop)", "stretch"]
RESIZE_TO = ["keep original", "custom size"]


def _scale(nchw, height, width, method, crop="disabled"):
    """ComfyUI's own scaler when it is importable (it has lanczos), else torch."""
    try:
        from comfy.utils import common_upscale
        return common_upscale(nchw, int(width), int(height), method, crop)
    except Exception:
        pass
    mode = {"lanczos": "bicubic", "bicubic": "bicubic", "bilinear": "bilinear",
            "area": "area", "nearest-exact": "nearest-exact"}.get(method, "bilinear")
    kw = {} if mode in ("area", "nearest-exact") else {"align_corners": False}
    return torch.nn.functional.interpolate(nchw, size=(int(height), int(width)), mode=mode, **kw)


def _conform_rate(frames, src_fps, force_rate):
    """Drop or repeat frames so the clip plays at `force_rate` without changing its length."""
    if not force_rate or force_rate <= 0 or not src_fps or abs(src_fps - force_rate) < 1e-6:
        return frames, (src_fps or force_rate)
    n = max(int(round(frames.shape[0] * float(force_rate) / float(src_fps))), 1)
    idx = np.minimum((np.arange(n) * float(src_fps) / float(force_rate)).astype(np.int64), frames.shape[0] - 1)
    return frames[idx], float(force_rate)


def _resize(images, height, width, method="lanczos", fit_mode="fit (pad)"):
    """images [N, H, W, 3] -> [N, height, width, 3].

    fit (pad):  largest size that still fits inside the frame, keeping the aspect ratio, centred on black.
    fill (crop): fills the frame, keeping the aspect ratio, and the overflow is cropped off.
    stretch:     stretched to the frame, aspect ratio ignored.
    """
    height, width = int(height), int(width)
    if images.shape[1] == height and images.shape[2] == width:
        return images
    nchw = images.permute(0, 3, 1, 2)
    if fit_mode == "stretch":
        out = _scale(nchw, height, width, method)
    elif fit_mode == "fill (crop)":
        out = _scale(nchw, height, width, method, crop="center")
        if out.shape[2] != height or out.shape[3] != width:      # no comfy.utils: scale to cover, then crop
            s = max(height / nchw.shape[2], width / nchw.shape[3])
            out = _scale(nchw, max(round(nchw.shape[2] * s), height), max(round(nchw.shape[3] * s), width), method)
            top, left = (out.shape[2] - height) // 2, (out.shape[3] - width) // 2
            out = out[:, :, top:top + height, left:left + width]
    else:                                                        # fit (pad)
        s = min(height / nchw.shape[2], width / nchw.shape[3])
        h, w = max(int(round(nchw.shape[2] * s)), 1), max(int(round(nchw.shape[3] * s)), 1)
        out = _scale(nchw, h, w, method)
        if h != height or w != width:
            top, left = (height - h) // 2, (width - w) // 2
            out = torch.nn.functional.pad(out, (left, width - w - left, top, height - h - top))
    return out.permute(0, 2, 3, 1).contiguous()


# --------------------------------------------------------------------------- node

class LoadVideoBatchFromDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "video_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (SORT_METHODS,),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1,
                                           "tooltip": "Load at most this many frames from EACH file (0 = no limit). Frames are large: a 243-frame 1344x768 clip is about 3 GB of RAM."}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                                         "tooltip": "Play every file at this frame rate, dropping or repeating frames to get there (0 = each file's own rate). The clip keeps its length, so the sound still lines up."}),
                "resize_to": (RESIZE_TO, {"tooltip": "keep original: the file is loaded at its own size (with more than one file, later files are brought to the first file's size, since one image batch cannot hold mixed sizes). custom size: the width and height below."}),
                "width": ("INT", {"default": 1344, "min": 1, "max": 16384, "step": 8,
                                  "tooltip": "Only used when resize_to is 'custom size'."}),
                "height": ("INT", {"default": 768, "min": 1, "max": 16384, "step": 8,
                                   "tooltip": "Only used when resize_to is 'custom size'."}),
                "fit_mode": (FIT_MODES, {"tooltip": "Only matters when the frame is a different shape. fit (pad): the largest size that still fits, aspect ratio kept, centred on black. fill (crop): fills the frame, aspect ratio kept, overflow cropped. stretch: aspect ratio ignored."}),
                "resize_method": (RESIZE_METHODS, {"tooltip": "How the pixels are resampled."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("images", "audio", "count")
    FUNCTION = "load_video"
    CATEGORY = "image/video"

    @classmethod
    def IS_CHANGED(cls, directory="", video_load_cap=0, start_index=0, load_always=False, sort_method=None,
                   frame_load_cap=0, force_rate=0.0, resize_to="keep original", width=1344, height=768,
                   fit_mode="fit (pad)", resize_method="lanczos", **kwargs):
        if load_always:
            return float("NaN")
        stamp = []
        try:
            for f in sorted(os.listdir(directory)):
                p = os.path.join(directory, f)
                if f.lower().endswith(VIDEO_EXTENSIONS):
                    stamp.append((f, os.path.getmtime(p)))
        except OSError:
            pass
        return hash((directory, video_load_cap, start_index, sort_method, frame_load_cap, force_rate,
                     resize_to, width, height, fit_mode, resize_method, tuple(stamp)))

    def load_video(self, directory: str, video_load_cap: int = 0, start_index: int = 0, load_always=False,
                   sort_method=None, frame_load_cap: int = 0, force_rate: float = 0.0,
                   resize_to="keep original", width=1344, height=768,
                   fit_mode="fit (pad)", resize_method="lanczos"):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        files = os.listdir(directory)
        if len(files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        files = [f for f in files if f.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(directory, f))]
        files = _sort_by(files, directory, sort_method)
        files = [os.path.join(directory, f) for f in files]
        files = files[start_index:]
        if video_load_cap > 0:
            files = files[:video_load_cap]
        if len(files) == 0:
            raise FileNotFoundError(f"No video files to load in '{directory}' (check start_index and the file types).")

        batches, waves = [], []
        fps0 = h0 = w0 = sr0 = ch0 = None
        for path in files:
            info = _probe(path)
            src_fps = float(info.get("fps") or 0.0) or 24.0
            # with a forced rate, read enough source frames to still fill the cap after conversion
            read_cap = frame_load_cap
            if frame_load_cap and force_rate and force_rate > 0:
                read_cap = int(np.ceil(frame_load_cap * src_fps / float(force_rate)))
            frames = _read_frames(path, read_cap, info)
            frames, file_fps = _conform_rate(frames, src_fps, force_rate)
            if frame_load_cap and frames.shape[0] > frame_load_cap:
                frames = frames[:frame_load_cap]
            images = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
            if h0 is None:
                if resize_to == "custom size":
                    h0, w0 = int(height), int(width)
                else:
                    h0, w0 = int(images.shape[1]), int(images.shape[2])
                fps0 = float(file_fps)
            batches.append(_resize(images, h0, w0, resize_method, fit_mode))

            audio = _read_audio(path, info, sr0, ch0)
            if audio is not None:
                wav = torch.from_numpy(np.ascontiguousarray(audio[0])).float()
                sr = int(audio[1])
                if sr0 is None:
                    sr0, ch0 = sr, int(wav.shape[0])
                waves.append(_match_channels(_resample(wav, sr, sr0), ch0))
            else:
                waves.append(None)

        if sr0 is None:
            sr0, ch0 = 44100, 1
        # each file's sound is matched to the length of its own frames, so picture and sound
        # stay lined up across the join (a file with no audio becomes silence).
        for i, w in enumerate(waves):
            n = max(int(round(batches[i].shape[0] / float(fps0) * sr0)), 1)
            if w is None:
                waves[i] = torch.zeros((ch0, n), dtype=torch.float32)
            elif w.shape[1] > n:
                waves[i] = w[:, :n]
            elif w.shape[1] < n:
                waves[i] = torch.cat([w, torch.zeros((ch0, n - w.shape[1]), dtype=torch.float32)], 1)

        images = batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)
        wave = waves[0] if len(waves) == 1 else torch.cat(waves, dim=1)
        print(f"[Load Video Batch From Dir] {len(files)} file(s), {images.shape[0]} frames, "
              f"{w0}x{h0} @ {fps0:g} fps, audio {wave.shape[1] / float(sr0):.2f}s")
        return (images, {"waveform": wave.unsqueeze(0).contiguous(), "sample_rate": int(sr0)}, len(files))


NODE_CLASS_MAPPINGS = {"LoadVideoBatchFromDir": LoadVideoBatchFromDir}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadVideoBatchFromDir": "Load Video Batch From Dir"}
