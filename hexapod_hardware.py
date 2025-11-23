# hexapod_env.py
import math
import numpy as np
import mujoco
from gymnasium import Env, spaces
import time

# =============== XML에 맞춘 이름들 ===============
JOINTS = [
    # rear (left → right)
    "left_rear_yaw","left_rear_hip","left_rear_knee",
    "right_rear_yaw","right_rear_hip","right_rear_knee",
    # mid
    "left_mid_yaw","left_mid_hip","left_mid_knee",
    "right_mid_yaw","right_mid_hip","right_mid_knee",
    # front
    "left_front_yaw","left_front_hip","left_front_knee",
    "right_front_yaw","right_front_hip","right_front_knee",
]

ACTS = [
    "left_rear_yaw_act","left_rear_hip_act","left_rear_knee_act",
    "right_rear_yaw_act","right_rear_hip_act","right_rear_knee_act",
    "left_mid_yaw_act","left_mid_hip_act","left_mid_knee_act",
    "right_mid_yaw_act","right_mid_hip_act","right_mid_knee_act",
    "left_front_yaw_act","left_front_hip_act","left_front_knee_act",
    "right_front_yaw_act","right_front_hip_act","right_front_knee_act",
]

# jointpos 센서 이름(순서 = JOINTS 순서와 동일해야 함)
JOINTPOS_SENSORS = [
    "left_rear_yaw_pos","left_rear_hip_pos","left_rear_knee_pos",
    "right_rear_yaw_pos","right_rear_hip_pos","right_rear_knee_pos",
    "left_mid_yaw_pos","left_mid_hip_pos","left_mid_knee_pos",
    "right_mid_yaw_pos","right_mid_hip_pos","right_mid_knee_pos",
    "left_front_yaw_pos","left_front_hip_pos","left_front_knee_pos",
    "right_front_yaw_pos","right_front_hip_pos","right_front_knee_pos",
]

# 접촉(Touch) 센서 이름(6개)
CONTACT_SENSORS = [
    "touch_left_rear",
    "touch_left_mid",
    "touch_left_front",
    "touch_right_mid",
    "touch_right_front",
    "touch_right_rear",
]

IMU_ACC_NAME  = "imu_acc"
IMU_GYRO_NAME = "imu_gyro"
KEYFRAME_NAME = "home"   # XML의 keyframe 이름


def quat_to_rpy(xquat):
    # xquat = [w, x, y, z]
    w, x, y, z = xquat
    # roll
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class HexapodEnv(Env):
    """
    Teacher–Student 친화 Hexapod 환경 (모든 관절각/IMU/접촉을 센서에서 읽음)

    obs_mode:
      - "teacher": 풍부 관측 반환 (RPY 3 + q 18 + qd 18 + contacts 6 = 45)
      - "student": 실세계 대응 관측(contacts 6 + action_hist + RPY 3)
      - "teacher_student": obs는 teacher, info["student_obs"]에 student 동시 제공
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        xml_path="hexapod_hardware.xml",
        render_mode="none",
        action_repeat=50,
        target_speed=0.2,
        obs_clip=10.0,
        seed=None,
        obs_mode="teacher",           # "teacher" | "student" | "teacher_student"
        student_hist_len=1,          # 과거 액션 프레임 수
        contact_threshold=1e-4,       # 접지 감지 임계치
        max_steps=500,
        topple_deg=45.0,              # 롤/피치 한계(도)
        min_z=0.05,
        warmup_steps = 50,

        random_init_posture = True,
        joint_init_std_deg = 7.0,
        yaw_init_std_deg = 10.0,

        student_include_imu = True,
        student_use_contact = True
    ):
        assert obs_mode in ("teacher", "student", "teacher_student")
        self.obs_mode = obs_mode
        self.student_hist_len = int(student_hist_len)
        self.contact_threshold = float(contact_threshold)
        self.warmup_steps = int(warmup_steps)

        # MuJoCo 로드
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.action_repeat = int(action_repeat)
        self.v_tar = float(target_speed)
        self.obs_clip = float(obs_clip)
        self.max_steps = int(max_steps)
        self.topple_rad = math.radians(float(topple_deg))
        self.min_z = float(min_z)
        self.student_include_imu = student_include_imu
        self.student_use_contact = student_use_contact

        # 시간 간격(한 step 호출당 실제 시뮬 시간)
        self.dt = self.model.opt.timestep * self.action_repeat

        # 랜덤 시드
        self.np_random = np.random.RandomState(seed if seed is not None else 0)

        # 뷰어(선택)
        self.viewer = None
        if self.render_mode == "human":
            from mujoco import viewer as mjv
            self.viewer = mjv.launch_passive(self.model, self.data)
            torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            if torso_id != -1:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = torso_id
                self.viewer.cam.distance = 2.0

        # 액추에이터 id 및 ctrlrange
        self.act_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS])
        cr = self.model.actuator_ctrlrange[self.act_ids]
        self.ctrl_min = cr[:, 0].copy()
        self.ctrl_max = cr[:, 1].copy()
        self.act_dim = len(ACTS)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.act_dim,), dtype=np.float32)

        self.j_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS])
        self.qadr = self.model.jnt_qposadr[self.j_ids]
        self.vadr = self.model.jnt_dofadr[self.j_ids]
        
        # ---- 센서 주소 매핑(일반화) ----
        self._sens_adr = {}   # name -> (adr, dim)
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = int(self.model.sensor_adr[i])
            dim = int(self.model.sensor_dim[i])
            if name:
                self._sens_adr[name] = (adr, dim)

        def _get_sensor(name):
            adr, dim = self._sens_adr[name]
            return self.data.sensordata[adr:adr+dim]

        self._get_sensor = _get_sensor

        # 조인트 각도는 jointpos 센서에서 읽음 (순서 JOINTS와 동일)
        self._q_sensor_names = list(JOINTPOS_SENSORS)

        # 접촉 센서 이름
        self._contact_sensor_names = list(CONTACT_SENSORS)

        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if self._torso_id == -1:
            raise RuntimeError("Body 'torso' not found in model.")

        # 키프레임 id
        self.kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
        if self.kid == -1:
            raise RuntimeError(f"Keyframe '{KEYFRAME_NAME}' not found in XML.")
        
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)
        xquat0 = self.data.xquat[self._torso_id].copy()
        self._roll0, self._pitch0, self._yaw0 = quat_to_rpy(xquat0)
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, xquat0)
        self._R_W_B0 = mat.reshape(3, 3)
        self._R_imu_corr = self._R_W_B0.copy()  # (지금은 안 쓰지만 남겨둠)
        
        # 학생용 히스토리 버퍼
        self._a_hist = np.zeros((self.student_hist_len, self.act_dim), dtype=np.float32)
        
        # qd 추정용 마지막 jointpos 센서값
        self._q_sensor_last = np.zeros(len(self._q_sensor_names), dtype=np.float64)

        # ===== 관측 공간 정의 (IMU → RPY delta 3개) =====
        teacher_obs_dim = 3 + 18 + 18 + 6   # rpy(3) + q(18) + qd(18) + contacts(6)
        student_obs_dim = 0
        if self.student_include_imu:
            student_obs_dim = 6 + (self.student_hist_len * self.act_dim) + 3  # contacts + hist + rpy(3)
        else:
            student_obs_dim = 6 + (self.student_hist_len * self.act_dim)      # contacts + hist

        if self.student_use_contact:
            student_obs_dim -= 6

        if self.obs_mode == "teacher":
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(teacher_obs_dim,), dtype=np.float32)
        elif self.obs_mode == "student":
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(student_obs_dim,), dtype=np.float32)
        else:  # teacher_student는 teacher 기준
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(teacher_obs_dim,), dtype=np.float32)

        self.step_ctr = 0
        self._home_ctrl = np.zeros(self.act_dim, dtype=np.float64)
        self._prev_action = np.zeros(self.act_dim, dtype=np.float32)

        self.random_init_posture = random_init_posture
        self.joint_init_std_deg = joint_init_std_deg
        self.yaw_init_std_deg = yaw_init_std_deg
        
    # ---------- 내부 유틸 ----------
    def _set_ctrl_from_action(self, a):
        # a in [-1,1] -> ctrlrange 스케일링
        u = (a + 1.0) * 0.5 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min
        self.data.ctrl[self.act_ids] = u

    def _contacts_norm(self):
        vals = []
        for n in self._contact_sensor_names:
            v = float(self._get_sensor(n)[0])  # touch는 dim=1
            vals.append(1.0 if v > self.contact_threshold else 0.0)
        arr = np.array(vals, dtype=np.float32)
        return arr * 2.0 - 1.0

    def _read_q_from_sensors(self):
        q = np.empty(len(self._q_sensor_names), dtype=np.float64)
        for i, n in enumerate(self._q_sensor_names):
            q[i] = float(self._get_sensor(n)[0])  # jointpos dim=1
        return q

    def _estimate_qd_from_sensors(self, q_curr):
        # 유한차분
        qd = (q_curr - self._q_sensor_last) / max(self.dt, 1e-6)
        self._q_sensor_last[:] = q_curr
        return qd

    def _normalize_q_by_ctrlrange(self, q):
        # 조인트-액추에이터 1:1 정렬 가정(ACTS 순서 == JOINTS 순서)
        q_norm = np.zeros_like(q, dtype=np.float32)
        for i in range(len(q)):
            qmin, qmax = self.ctrl_min[i], self.ctrl_max[i]
            if qmax > qmin:
                q_norm[i] = 2.0 * (q[i] - qmin) / (qmax - qmin) - 1.0
            else:
                q_norm[i] = 0.0
        return np.clip(q_norm, -1.0, 1.0)

    def _teacher_obs(self):
        """
        이전: IMU acc(3) + gyro(3)를 넣었음
        변경: torso의 (roll, pitch, yaw) - 초기값 을 obs 앞부분에 사용
        """
        # ----- RPY (초기 대비 차이) -----
        roll, pitch, yaw = self._torso_rpy()  # 이미 초기 상태와의 차를 반환

        # topple_deg 기준으로 정규화: topple_rad → 1.0
        scale = self.topple_rad if self.topple_rad > 0 else math.radians(45.0)
        rpy = np.array([roll, pitch, yaw], dtype=np.float32) / scale
        rpy_norm = np.clip(rpy, -1.0, 1.0)

        # q, qd
        q  = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr]

        q_norm = np.zeros_like(q, dtype=np.float32)
        for i in range(len(q)):
            q_min, q_max = self.ctrl_min[i], self.ctrl_max[i]
            if q_max > q_min:
                q_norm[i] = 2.0 * (q[i] - q_min) / (q_max - q_min) - 1.0
            else:
                q_norm[i] = 0.0
        q_norm = np.clip(q_norm, -1.0, 1.0)

        qd_norm = np.clip(qd / 5.0, -1.0, 1.0).astype(np.float32)

        contacts_norm = self._contacts_norm()

        obs = np.concatenate([rpy_norm, q_norm, qd_norm, contacts_norm]).astype(np.float32)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _student_obs(self):
        """
        학생 관측: contacts + action_hist + (옵션) RPY
        """
        contacts_norm = self._contacts_norm()
        hist_flat = self._a_hist.reshape(-1)

        # RPY (초기 대비 차이)
        roll, pitch, yaw = self._torso_rpy()
        scale = self.topple_rad if self.topple_rad > 0 else math.radians(45.0)
        rpy = np.array([roll, pitch, yaw], dtype=np.float32) / scale
        rpy_norm = np.clip(rpy, -1.0, 1.0)

        if self.student_include_imu and self.student_use_contact:
            obs = np.concatenate([
                contacts_norm.astype(np.float32),
                hist_flat.astype(np.float32),
                rpy_norm,
            ]).astype(np.float32)
        elif self.student_include_imu and not self.student_use_contact:
            obs = np.concatenate([
                hist_flat.astype(np.float32),
                rpy_norm,
            ]).astype(np.float32)
        else:
            obs = np.concatenate([
                hist_flat.astype(np.float32)
            ]).astype(np.float32)

        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _torso_rpy(self):
        xquat = self.data.xquat[self._torso_id]
        roll, pitch, yaw = quat_to_rpy(xquat)
        # 🔹 초기 자세의 기울기(roll0, pitch0, yaw0)를 빼서 home에서 (0,0,0)이 되게
        return roll - self._roll0, pitch - self._pitch0, yaw - self._yaw0   

    def _forward_speed_body(self):
        """
        world linear velocity(qvel[0:3])를 torso body의 x축(몸 앞) 방향으로 project.
        """
        v_world = np.array([
            float(self.data.qvel[0]),
            float(self.data.qvel[1]),
            float(self.data.qvel[2]),
        ])

        xquat = self.data.xquat[self._torso_id]
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, xquat)
        R_W_B = mat.reshape(3, 3)   # body -> world

        # body x축의 world 표현 (열벡터)
        body_x_world = R_W_B[:, 0]

        # v_world를 body-x 방향으로 투영
        v_fwd = float(np.dot(v_world, body_x_world))
        return v_fwd

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        self.step_ctr = 0
        if seed is not None:
            self.np_random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)

        # 약간의 초깃값 노이즈(학습 안정화용) — 센서 jointpos 기준으로
        mujoco.mj_forward(self.model, self.data)

        # 🔹 초기자세 유지용 ctrl 스냅샷
        home_qpos_full = self.model.key_qpos[self.kid].copy()
        home_joint_q = home_qpos_full[self.qadr]

        home_ctrl_all_actuators = self.model.key_ctrl[self.kid] # (nu,)
        home_ctrl_subset = home_ctrl_all_actuators[self.act_ids] # (18,)
        
        self._home_ctrl = home_ctrl_subset.astype(np.float32)
        
        std_rad = math.radians(self.joint_init_std_deg)
        noise = self.np_random.normal(
            loc = 0.0,
            scale = std_rad,
            size = self.act_dim
        ).astype(np.float32)

        noisy_ctrl = self._home_ctrl + noise
        self._home_ctrl = np.clip(noisy_ctrl, self.ctrl_min, self.ctrl_max).astype(np.float32)
        
        if self.random_init_posture:
            self._home_ctrl = self.np_random.uniform(low=self.ctrl_min, high=self.ctrl_max).astype(np.float32)
        self.data.ctrl[self.act_ids] = self._home_ctrl

        # 내부 상태 초기화
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        if self.data.act.size:
            self.data.act[:] = 0

        mujoco.mj_forward(self.model, self.data)

        self._a_hist[:] = 0.0
        self._q_sensor_last[:] = self._read_q_from_sensors()
        self._prev_action[:] = 0.0

        # reset 시점에서의 기준 RPY 다시 잡고 싶으면 여기서도 갱신 가능
        xquat0 = self.data.xquat[self._torso_id].copy()
        self._roll0, self._pitch0, self._yaw0 = quat_to_rpy(xquat0)

        teacher_obs = self._teacher_obs()
        student_obs = self._student_obs()

        if self.obs_mode == "teacher":
            return teacher_obs, {}
        elif self.obs_mode == "student":
            return student_obs, {}
        else:
            return teacher_obs, {"student_obs": student_obs}

    def step(self, action):
        self.step_ctr += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self.student_hist_len > 0:
            if self.student_hist_len > 1:
                self._a_hist[:-1] = self._a_hist[1:]
            self._a_hist[-1] = action

        # 🔹 warmup 동안은 초기 자세 유지 (action 무시)
        if self.step_ctr <= self.warmup_steps:
            self.data.ctrl[self.act_ids] = self._home_ctrl
            if self.step_ctr == self.warmup_steps:
                xquat0 = self.data.xquat[self._torso_id].copy()
                self._roll0, self._pitch0, self._yaw0 = quat_to_rpy(xquat0)
        else:
            self._set_ctrl_from_action(action)

        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        teacher_obs = self._teacher_obs()
        student_obs = self._student_obs()

        # --- 보상 계산 ---
        xd = float(self.data.qvel[0])
        yd = float(self.data.qvel[1])
        zd = float(self.data.qvel[2])
        zpos = float(self.data.qpos[2])

        roll, pitch, yaw = self._torso_rpy()  # 이미 초기 상태 대비 차이

        # 🔹 몸 기준 앞 방향 속도
        fwd_speed = self._forward_speed_body()

        velocity_rew = (1.0 / (abs(fwd_speed - self.v_tar) + 1.0) - 1.0 / (self.v_tar + 1.0)) * 10.0

        y_pen   = (yd * 0.5)**2 * 0.5
        yaw_pen = (yaw**2) * 1.2
        pitch_pen = (pitch * 4.0) ** 2
        roll_pen  = (roll  * 3) ** 2
        z_pen = (0.010 - zpos) * 20.0 if zpos < 0.010 else 0.0
        zd_pen = (zd**2) * 2
        ctrl_pen = 0.1 * float(np.mean(np.square(action - self._prev_action)))

        reward = velocity_rew - yaw_pen - z_pen - pitch_pen - roll_pen - zd_pen
        if self.step_ctr <= self.warmup_steps:
            reward = 0.0

        terminated = False
        truncated  = self.step_ctr >= self.max_steps

        if self.obs_mode == "teacher":
            obs_out = teacher_obs
        elif self.obs_mode == "student":
            obs_out = student_obs
        else:
            obs_out = teacher_obs

        info = {
            "xd": xd,
            "forward_speed": fwd_speed,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "student_obs": student_obs,
            "reward_terms": {
                "velocity_rew": velocity_rew,
                "yaw_pen": yaw_pen,
                "pitch_pen": pitch_pen,
                "roll_pen": roll_pen,
                "z_pen": z_pen,
                "ctrl_pen": ctrl_pen,
            }
        }

        self._prev_action = action.copy()
        return obs_out, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
