#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ======================================
# 🔧 전역 설정 (여기만 바꾸면 됨)
# ======================================

PATH = "./logs_hexapod_hardware/teacher"
TAG_REW = "rollout/ep_rew_mean"

SMOOTH = 20
TITLE = None

# 그래프 크기 (inch)
FIG_WIDTH  = 7
FIG_HEIGHT = 5

# 선 두께
LINE_WIDTH_REW = 3.0

# 글씨 크기
FONT_TITLE  = 18
FONT_LABEL  = 15
FONT_TICK   = 12
FONT_LEGEND = 14   # 범례 크게
# ======================================


def moving_average(x, window_size):
    if window_size <= 1:
        return np.array(x)

    x = np.array(x, dtype=float)
    kernel = np.ones(window_size, dtype=float) / window_size
    y = np.convolve(x, kernel, mode="valid")

    pad_left = [x[0]] * ((window_size - 1) // 2)
    pad_right = [x[-1]] * (window_size - 1 - (window_size - 1) // 2)

    return np.array(pad_left + list(y) + pad_right)


def load_scalar_from_tb(path, tag):
    if os.path.isdir(path):
        event_files = [
            os.path.join(path, f) for f in os.listdir(path)
            if "tfevents" in f
        ]
        if not event_files:
            raise FileNotFoundError("tfevents 파일이 없습니다.")
        event_files.sort(key=os.path.getmtime)
        event_path = event_files[-1]
    else:
        event_path = path

    print(f"[INFO] using tfevents: {event_path}")

    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(f"tag '{tag}' 없음. 가능한 tag:\n  " +
                         "\n  ".join(ea.Tags()["scalars"]))

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return np.array(steps), np.array(values)


def main():
    steps, values = load_scalar_from_tb(PATH, TAG_REW)

    if SMOOTH > 1:
        values_s = moving_average(values, SMOOTH)
    else:
        values_s = values

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    plt.plot(
        steps, values_s,
        linewidth=LINE_WIDTH_REW,
        label=f"Episode Reward (win={SMOOTH})"
    )

    plt.xlabel("Step", fontsize=FONT_LABEL)
    plt.ylabel("Episode Reward", fontsize=FONT_LABEL)
    plt.title(
        TITLE or "Episode Reward over Training (Teacher Model)",
        fontsize=FONT_TITLE
    )

    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)

    plt.legend(fontsize=FONT_LEGEND)

    # 🔧 격자 더 얇게 (linewidth=0.3)
    plt.grid(True, linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
