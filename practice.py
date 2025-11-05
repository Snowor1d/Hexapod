import mujoco
from mujoco import viewer
import time

# ✨ 바뀐 부분: 파일에서 로드
model = mujoco.MjModel.from_xml_path("hexapod.xml")
data = mujoco.MjData(model)

# 아래부터는 네가 쓰던 기존 코드 그대로
# safe_reset(...) 등을 그대로 호출
with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        # time.sleep(0.1)
