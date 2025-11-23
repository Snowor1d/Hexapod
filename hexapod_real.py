# hexapod_env.py
import math
import numpy as np
import mujoco
from gymnasium import Env, spaces

JOINTS = [
    "yaw_fl","hip_fl","knee_fl",
    "yaw_fr","hip_fr","knee_fr",
    "yaw_ml","hip_ml","knee_ml",
    "yaw_mr","hip_mr","knee_mr",
    "yaw_rl","hip_rl","knee_rl",
    "yaw_rr","hip_rr","knee_rr",
]
ACTS  = [
    "a_yaw_fl","a_hip_fl","a_knee_fl",
    "a_yaw_fr","a_hip_fr","a_knee_fr",
    "a_yaw_ml","a_hip_ml","a_knee_ml",
    "a_yaw_mr","a_hip_mr","a_knee_mr",
    "a_yaw_rl","a_hip_rl","a_knee_rl",
    "a_yaw_rr","a_hip_rr","a_knee_rr",
]

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
    Teacher–Student 친화 Hexapod 환경

    obs_mode:
      - "teacher": 풍부 관측만 반환 (Box).
      - "student": 실세계 대응 관측(foot contact + past actions)만 반환 (Box).
      - "teacher_student": step()/reset()은 teacher 관측을 obs로 반환하되,
                           info["student_obs"]에 student 관측을 함께 제공.

    Student 관측:
      - foot contacts (6) in {-1, +1}
      - past actions history (student_hist_len * act_dim), 최근이 마지막에 위치
        (리셋 시 0으로 초기화; 저장되는 값은 [-1,1] 원시 액션)

    참고:
      - 보상/종료는 시뮬 내부 상태를 사용(실세계에선 보상 미사용).
      - SB3 호환: "teacher_student" 모드가 distillation 수집에 편리
                  (policy는 teacher obs로, info로 student obs 수집).
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        xml_path="hexapod.xml",
        render_mode="none",
        action_repeat=20,
        target_speed=0.35,
        obs_clip=10.0,
        seed=None,
        obs_mode="teacher",            # "teacher" | "student" | "teacher_student"
        student_hist_len=1,           # 과거 액션 프레임 수
        contact_threshold=1e-4,        # 접지 감지 임계치
        max_steps=500,
        topple_deg=45.0,               # 롤/피치 한계(도)
        min_z=0.05                     # 바닥 관통 등 실패 판정
    ):
        assert obs_mode in ("teacher", "student", "teacher_student")
        self.obs_mode = obs_mode
        self.student_hist_len = int(student_hist_len)
        self.contact_threshold = float(contact_threshold)

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

        # ids & indices
        self.act_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS])
        self.j_ids   = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    n) for n in JOINTS])
        self.qadr    = self.model.jnt_qposadr[self.j_ids]
        self.vadr    = self.model.jnt_dofadr[self.j_ids]

        # ctrlrange
        cr = self.model.actuator_ctrlrange[self.act_ids]
        self.ctrl_min = cr[:, 0].copy()
        self.ctrl_max = cr[:, 1].copy()

        # keyframe
        self.kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "safe_start")
        if self.kid == -1:
            raise RuntimeError("Keyframe 'safe_start' not found in XML.")

        # 랜덤 시드
        self.np_random = np.random.RandomState(seed if seed is not None else 0)

        # 뷰어 (선택)
        self.viewer = None
        if self.render_mode == "human":
            from mujoco import viewer as mjv
            self.viewer = mjv.launch_passive(self.model, self.data)
            torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            if torso_id != -1:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = torso_id
                self.viewer.cam.distance = 2.0

        # 캐시
        self._feet_geoms = ["foot_fl","foot_fr","foot_ml","foot_mr","foot_rl","foot_rr"]

        # 액션/관측 공간 정의
        self.act_dim = len(ACTS)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.act_dim,), dtype=np.float32)

        # Teacher 관측 차원: IMU(6) + q(18) + qd(18) + contacts(6) = 48
        teacher_obs_dim = 6 + 18 + 18 + 6

        # Student 관측 차원: contacts(6) + action_hist(student_hist_len * act_dim + imu sensordata)
        student_obs_dim = 6 + self.student_hist_len * self.act_dim + 4

        if self.obs_mode == "teacher":
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(teacher_obs_dim,), dtype=np.float32)
        elif self.obs_mode == "student":
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(student_obs_dim,), dtype=np.float32)
        else:  # "teacher_student": 외부엔 teacher obs를 반환하므로 teacher 기준
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(teacher_obs_dim,), dtype=np.float32)

        # 히스토리 버퍼(학생용)
        self._a_hist = np.zeros((self.student_hist_len, self.act_dim), dtype=np.float32)

        # 스텝 카운터
        self.step_ctr = 0

    # ---------- 내부 유틸 ----------
    def _set_ctrl_from_action(self, a):
        # a in [-1,1] -> ctrlrange 스케일링
        u = (a + 1.0) * 0.5 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min
        self.data.ctrl[self.act_ids] = u

    def _get_contacts_norm(self):
        # sensordata[6:12]가 접지 힘으로 세팅되어있다는 가정(사용자 XML에 맞춤)
        contacts = (self.data.sensordata[6:12] > self.contact_threshold).astype(np.float32)
        return contacts * 2.0 - 1.0  # {0,1}->{-1,1}

    def _teacher_obs(self):
        # IMU
        imu_acc  = self.data.sensordata[0:3]
        imu_gyro = self.data.sensordata[3:6]
        imu_acc_norm  = np.clip(imu_acc / 10.0, -1.0, 1.0)
        imu_gyro_norm = np.clip(imu_gyro / 10.0, -1.0, 1.0)

        # 관절
        q  = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr]

        # 각도 정규화 (actuator ctrlrange를 근사적 범위로 사용)
        q_norm = np.zeros_like(q)
        for i in range(len(q)):
            q_min, q_max = self.ctrl_min[i], self.ctrl_max[i]
            if q_max > q_min:
                q_norm[i] = 2.0 * (q[i] - q_min) / (q_max - q_min) - 1.0
            else:
                q_norm[i] = 0.0
            q_norm[i] = np.clip(q_norm[i], -1.0, 1.0)

        qd_norm = np.clip(qd / 5.0, -1.0, 1.0)

        contacts_norm = self._get_contacts_norm()

        obs = np.concatenate([imu_acc_norm, imu_gyro_norm, q_norm, qd_norm, contacts_norm]).astype(np.float32)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _student_obs(self):
        contacts_norm = self._get_contacts_norm()
        # action history: shape (H, act_dim) -> flatten row-major
        hist_flat = self._a_hist.reshape(-1)

        imu_acc = self.data.sensordata[0:3]
        imu_gyro = self.data.sensordata[3:6]
        imu_acc_norm = np.clip(imu_acc / 10.0, -1.0, 1.0)
        imu_gyro_norm = np.clip(imu_gyro / 10.0, -1.0, 1.0)

        obs = np.concatenate([contacts_norm, hist_flat]).astype(np.float32)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _torso_rpy(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        xquat = self.data.xquat[bid]
        return quat_to_rpy(xquat)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        self.step_ctr = 0
        if seed is not None:
            self.np_random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)

        # 약간의 초깃값 노이즈(학습 안정화용)
        for j, qa in enumerate(self.qadr):
            self.data.qpos[qa] += self.np_random.uniform(-np.deg2rad(2), np.deg2rad(2))

        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        if self.data.act.size:
            self.data.act[:] = 0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0

        mujoco.mj_forward(self.model, self.data)

        # 초기 ctrl을 현재 q에 맞춤(가벼운 안정화)
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = self.data.qpos[self.qadr[i]]

        # 학생용 히스토리 초기화
        self._a_hist[:] = 0.0

        # 관측 만들기
        teacher_obs = self._teacher_obs()
        student_obs = self._student_obs()

        if self.obs_mode == "teacher":
            return teacher_obs, {}
        elif self.obs_mode == "student":
            return student_obs, {}
        else:  # "teacher_student"
            # obs는 teacher로, student는 info에 담아서 반환
            return teacher_obs, {"student_obs": student_obs}

    def step(self, action):
        self.step_ctr += 1

        # 액션 클립 & 히스토리에 "원시([-1,1])" 액션 저장 (실세계와 동일한 포맷)
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        # 히스토리 쉬프트 후 현재 액션 push
        if self.student_hist_len > 0:
            if self.student_hist_len > 1:
                self._a_hist[:-1] = self._a_hist[1:]
            self._a_hist[-1] = action

        # 물리 시뮬
        self._set_ctrl_from_action(action)
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        # 관측(teacher/student)
        teacher_obs = self._teacher_obs()
        student_obs = self._student_obs()

        # --- 보상 계산(시뮬 내부상태) ---
        xd = float(self.data.qvel[0])          # 전진 속도``
        yd = float(self.data.qvel[1])
        zd = float(self.data.qvel[2])
        zpos = float(self.data.qpos[2])        # 몸체 높이
        qw = float(self.data.qpos[3])          # free-joint quaternion scalar

        roll, pitch, yaw = self._torso_rpy()

        # 목표 속도 추종(원래 구조 유지)
        velocity_rew = (1.0 / (abs(xd - self.v_tar) + 1.0) - 1.0 / (self.v_tar + 1.0)) * 10.0
        
        y_pen = np.square(yd * 0.5)*0.5
        # "자세" 벌점(쿼터니언 스칼라부 기반 yaw_dev 근사)
        yaw_pen = np.square(yaw) *0.5 # yaw_rate_pen를 주는 것 검토

        # 롤/피치 벌점(넘어짐 방지)
        pitch_pen = (pitch * 4.0) ** 2
        roll_pen  = (roll  * 3.0) ** 2

        # 저고도 벌점(바닥에 너무 근접)
        z_pen = 0.0
        if zpos < 0.10:
            z_pen += (0.10 - zpos) * 20.0

        zd_pen = 0
        zd_pen = np.square(zd) * 0.25
        

        # 제어 에너지(과도한 액션 억제)
        ctrl_pen = 0.01 * float(np.mean(np.square(action)))

        reward = velocity_rew - yaw_pen - y_pen - zd_pen# - pitch_pen - roll_pen - z_pen - ctrl_pen

        # --- 종료/트렁케이트 ---
        terminated = (abs(roll) > self.topple_rad) or (abs(pitch) > self.topple_rad) or (zpos < self.min_z)
        truncated  = self.step_ctr >= self.max_steps

        # --- obs 반환 모드 ---
        if self.obs_mode == "teacher":
            obs_out = teacher_obs
        elif self.obs_mode == "student":
            obs_out = student_obs
        else:  # "teacher_student"
            obs_out = teacher_obs

        info = {
            "xd": xd,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "student_obs": student_obs,   # 항상 제공 → distillation 수집에 유용
            "reward_terms": {
                "velocity_rew": velocity_rew,
                "yaw_pen": yaw_pen,
                "pitch_pen": pitch_pen,
                "roll_pen": roll_pen,
                "z_pen": z_pen,
                "ctrl_pen": ctrl_pen,
            }
        }
        return obs_out, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
