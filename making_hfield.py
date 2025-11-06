import numpy as np
import os

# 간단한 테스트 hfield 생성
N = 256
H = np.zeros((N, N), dtype=np.float32)

# 간단한 패턴 생성 (디버깅용)
for i in range(N):
    for j in range(N):
        H[i, j] = 0.1 * np.sin(0.1 * i) * np.cos(0.1 * j) + 0.5

H = np.clip(H, 0.0, 1.0)
H = np.ascontiguousarray(H.astype(np.float32))

out_bin = "/Users/snowcap/Hexapod/hfield_rocky.bin"

if os.path.exists(out_bin):
    os.remove(out_bin)

H.tofile(out_bin)

print(f"Created simple hfield: {out_bin}")
print(f"File size: {os.path.getsize(out_bin)} bytes")
print(f"Data range: {H.min():.3f} to {H.max():.3f}")