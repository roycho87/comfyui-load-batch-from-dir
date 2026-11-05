# ComfyUI Load Batch From Dir (text, video, audio)

Text, video and audio versions of **Load Image Batch From Dir (Inspire)** from [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack), in one pack. Each one reads the files in a folder, in the order you choose, and loads as many as you ask for. Point one at a folder of prompts, clips or sound files and step through them one per run.

| Node | Category | Loads |
|---|---|---|
| **Load Text Batch From Dir** | `text` | txt, md, prompt, text, json, yaml, yml, csv |
| **Load Video Batch From Dir** | `image/video` | mp4, mov, mkv, webm, avi, m4v, mpg, mpeg, wmv, flv, ogv (with the sound) |
| **Load Audio Batch From Dir** | `audio` | wav, mp3, flac, ogg, opus, m4a, aac, wma, aiff, mka |

> The video and audio loaders used to be separate repos ([comfyui-load-video-batch-from-dir](https://github.com/roycho87/comfyui-load-video-batch-from-dir), [comfyui-load-audio-batch-from-dir](https://github.com/roycho87/comfyui-load-audio-batch-from-dir)). They're all here now. If you installed those, remove them so the same nodes aren't loaded twice; your workflows keep working, since the node names haven't changed.

## Install
Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/roycho87/comfyui-load-text-batch-from-dir
```

No extra packages are needed: video uses PyAV, ffmpeg or OpenCV and audio uses `torchaudio` (or `av`), whichever ComfyUI already has.

**One file per run** (any of the three): set the load cap to 1 and drive `start_index` with a counter (for example a primitive set to *increment*) to step through a folder, one file per queued run.

## Load Text Batch From Dir

### Inputs
| Input | Default | Meaning |
|---|---|---|
| `directory` | | Folder to read |
| `text_load_cap` | 0 | Load at most this many files (0 = no limit) |
| `start_index` | 0 | Skip this many files first (-1 = the last file) |
| `load_always` | off | Re-run every time instead of only when the inputs or the folder change |
| `sort_method` | None | None, Alphabetical, Numerical (first number in the file name) or Datetime, ascending or descending |
| `separator` | `\n\n` | Put between files when more than one is loaded. `\n` is a line break, `\t` a tab |
| `strip` | on | Trim blank lines and spaces from the start and end of each file |

### Outputs
* `text` - the loaded file (txt, md, prompt, text, json, yaml, yml, csv)
* `count` - how many files were loaded

An image batch stacks frames, but there is no such thing as a batch of strings, so several files are joined one after another with `separator` between them.

Files are read as UTF-8, then UTF-8 with a BOM, then latin-1, so a file saved from any editor still loads.

## Load Video Batch From Dir

### Inputs
| Input | Default | Meaning |
|---|---|---|
| `directory` | | Folder to read |
| `video_load_cap` | 0 | Load at most this many files (0 = no limit) |
| `start_index` | 0 | Skip this many files first (-1 = the last file) |
| `load_always` | off | Re-run every time instead of only when the inputs or the folder change |
| `sort_method` | None | None, Alphabetical, Numerical (first number in the file name) or Datetime, ascending or descending |
| `frame_load_cap` | 0 | Load at most this many frames from **each** file (0 = no limit) |
| `force_rate` | 0 | Play every file at this frame rate (0 = each file's own rate) |
| `resize_to` | keep original | Load at the file's own size, or resize to `width` x `height` |
| `width` / `height` | 1344 x 768 | The frame size, when `resize_to` is **custom size** |
| `fit_mode` | fit (pad) | How a frame of a different shape is made to fit (below) |
| `resize_method` | lanczos | lanczos, bicubic, bilinear, area or nearest-exact |

### Outputs
* `images` - every frame of the loaded files, one after another (mp4, mov, mkv, webm, avi, m4v, mpg, mpeg, wmv, flv, ogv)
* `audio` - those files' sound, in the same order
* `count` - how many files were loaded

An image batch stacks frames, so several files are stacked end to end, and their sound is joined the same way. Audio is converted to the first file's sample rate and channel count, and each file's sound is trimmed or padded to the length of its own frames, so picture and sound stay lined up. A file with no audio contributes silence.

### Frame rate
`force_rate` works like the VHS loader's: frames are dropped or repeated so the clip runs at the rate you ask for **at the same speed**, not slowed down or sped up. A 2-second 24 fps clip forced to 12 fps comes back as 24 frames that still play for 2 seconds, and the sound is untouched. Leave it at 0 to take each file's own rate.

With several files, the first file's rate (forced or not) is used for all of them.

`frame_load_cap` is applied after that, so the cap is counted in output frames. Frames are big, so this is the control to reach for on long clips.

### Resizing
On **keep original** a single file is loaded at its own size and nothing is resampled. Loading several at once is the one case where resizing is forced, since an image batch cannot hold mixed frame sizes: later files are brought to the first file's size. **custom size** resizes everything to the `width` x `height` you set.

`fit_mode` only matters when the frame is a different shape from the file:

| `fit_mode` | What it does |
|---|---|
| `fit (pad)` | Scales to the largest size that still fits inside the frame, keeping the aspect ratio, centred on black. Nothing is lost or distorted; a different shape gets bars. |
| `fill (crop)` | Scales until the frame is filled, keeping the aspect ratio, and crops the overflow from the centre. No bars, but the edges are cut. |
| `stretch` | Scales straight to the frame and ignores the aspect ratio. |

`resize_method` picks the resampling: **lanczos** (sharpest, the default), bicubic, bilinear, area (good for big reductions) or nearest-exact. ComfyUI's own scaler is used when it can be imported, so these behave like Upscale Image; otherwise it falls back to torch and lanczos becomes bicubic.

### Memory
Frames are kept in RAM as float32. A 243-frame 1344x768 clip is about 3 GB, so a folder of ten of them is around 30 GB. Use `frame_load_cap`, or load one file at a time.

## Load Audio Batch From Dir

### Inputs
| Input | Default | Meaning |
|---|---|---|
| `directory` | | Folder to read |
| `audio_load_cap` | 0 | Load at most this many files (0 = no limit) |
| `start_index` | 0 | Skip this many files first (-1 = the last file) |
| `load_always` | off | Re-run every time instead of only when the inputs or the folder change |
| `sort_method` | None | None, Alphabetical, Numerical (first number in the file name) or Datetime, ascending or descending |

### Outputs
* `audio` - the loaded files joined one after another into a single AUDIO (wav, mp3, flac, ogg, opus, m4a, aac, wma, aiff, mka)
* `count` - how many files were loaded

An image batch stacks frames, but audio files differ in length, so they are joined end to end instead. Files are converted to the first file's sample rate and channel count so they can be joined.

## Differences from the Inspire node
* Text, video and audio files instead of images; no mask output. Text adds the separator and strip controls; video adds the sound, the frame cap, the forced rate and the resize controls.
* Folders that happen to match a file extension are skipped (the original crashes on them).
* The change check looks at the input values and the files' modified times, not only the input names.

## Status
* **Text:** File selection, sorting, the encodings, the join, the separator and strip were tested on real files outside ComfyUI. It has not been run inside ComfyUI yet. Please open an issue with the error text if a file fails to load.
* **Video:** File selection, sorting, decoding, the frame cap, the forced rate, all three fit modes (including the letterbox padding), the audio join and the sync trim were tested on real files outside ComfyUI, with ffmpeg and OpenCV as the decoders. It has not been run inside ComfyUI yet, and `lanczos` itself comes from ComfyUI's scaler, which was exercised here only through the torch fallback. Please open an issue with the error text if a file fails to load.
* **Audio:** The file selection and sorting were tested with a stub. The audio reading, resampling and joining have not been run on a real ComfyUI install yet. Please open an issue with the error text if a file fails to load.

## Credits
Adapted from `LoadImagesFromDirBatch` in [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) by ltdrdata (GPL-3.0). Released under the same licence (GPL-3.0, see `LICENSE`).
