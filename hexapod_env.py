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
    metadata = {"render_modes": ["human", "none"]}
    
    def __init__(self, xml_path="hexapod.xml", render_mode="none", action_repeat=20, target_speed=0.35, obs_clip=10.0, seed=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.action_repeat = int(action_repeat)
        self.v_tar = float(target_speed)
        self.obs_clip = float(obs_clip)
        self.max_steps = 500

        self.prev_u = None
        self.max_du = 0.1
        self.alpha = 0.85
        self.step_ctr = 0
        # ids & indices
        self.act_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS])
        self.j_ids   = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    n) for n in JOINTS])
        self.qadr    = self.model.jnt_qposadr[self.j_ids]
        self.vadr    = self.model.jnt_dofadr[self.j_ids]

        
        # ctrlrange 
        cr = self.model.actuator_ctrlrange[self.act_ids]
        self.ctrl_min = cr[:, 0].copy()
        self.ctrl_max = cr[:, 1].copy()

        obs_dim = 6+18+18+6 #IMU(6) + 6개 발 접지 + 18개 힌지 q, 18개 힌지 q_dot
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTS),), dtype=np.float32)

        self.kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "safe_start")
        if self.kid == -1:
            raise RuntimeError("Keyframe 'safe_start' not found in XML.")

        self.viewer = None
        if self.render_mode == "human":
            from mujoco import viewer as mjv
            self.viewer = mjv.launch_passive(self.model, self.data)
            torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            if torso_id != -1:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = torso_id
                self.viewer.cam.distance = 2.0
        
        self.np_random = np.random.RandomState(seed if seed is not None else 0 )

        # caches
        self._feet_geoms = ["foot_fl","foot_fr","foot_ml","foot_mr","foot_rl","foot_rr"]
    
        # print("=== Sensor Info ===")
        # for i in range(self.model.nsensor):
        #     name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        #     stype = int(self.model.sensor_type[i])  # 센서 타입 enum (예: ACCELEROMETER, GYRO, TOUCH 등)
        #     dim   = int(self.model.sensor_dim[i])   # 데이터 차원
        #     adr   = int(self.model.sensor_adr[i])   # sensordata 내 시작 인덱스
        #     print(f"{i:2d} | name={name:12s} | type={stype:2d} | dim={dim} | adr={adr}")


    def _set_ctrl_from_action(self, a):
        # a가 [-1, 1] 범위라고 가정
        # [-1, 1] -> [ctrl_min, ctrl_max] 범위로 스케일링
        u = (a + 1.0) * 0.5 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min
        self.data.ctrl[self.act_ids] = u
        # self.prev_u 관련 로직 모두 제거

    def _obs(self):
        # --- IMU ---
        imu_acc = self.data.sensordata[0:3]       # [ax, ay, az] (m/s^2)
        imu_gyro = self.data.sensordata[3:6]      # [wx, wy, wz] (rad/s)

        # --- Normalize IMU ---
        imu_acc_norm = np.clip(imu_acc / 10.0, -1.0, 1.0)       # 약 ±9.81 기준 → [-1, 1]
        imu_gyro_norm = np.clip(imu_gyro / 10.0, -1.0, 1.0)     # 보통 ±10 rad/s 범위 내

        # --- Joint Angles & Velocities ---
        q = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr]

        # ctrlrange를 이용해서 각도 정규화
        q_norm = np.zeros_like(q)
        for i in range(len(q)):
            q_min, q_max = self.ctrl_min[i], self.ctrl_max[i]
            q_norm[i] = 2.0 * (q[i] - q_min) / (q_max - q_min) - 1.0
            q_norm[i] = np.clip(q_norm[i], -1.0, 1.0)

        # 속도는 대략 ±5 rad/s로 정규화
        qd_norm = np.clip(qd / 5.0, -1.0, 1.0)

        # --- Foot Contact ---
        contacts = (self.data.sensordata[6:12] > 1e-4).astype(np.float32)
        contacts_norm = contacts * 2.0 - 1.0   # {0,1} → {-1,1}  (논문 [-1,1] 범위 맞춤)

        # --- Concatenate ---
        obs = np.concatenate([
            imu_acc_norm,
            imu_gyro_norm,
            q_norm,
            qd_norm,
            contacts_norm
        ]).astype(np.float32)

        # Clip for safety
        np.clip(obs, -1.0, 1.0, out=obs)

        return obs


    def _torso_rpy(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        xquat = self.data.xquat[bid]
        return quat_to_rpy(xquat)

    # Gym API 
    def reset(self, *, seed=None, options=None):
        self.step_ctr = 0
        # if seed is not None:
        #     self.np_random.seed(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)

        # small noise -> 평지 학습 안정화? 확인 필요
        # self.data.qpos[5] += self.np_random.uniform(-np.deg2rad(5), np.deg2rad(5))
        
        for j, qa in enumerate(self.qadr):
            self.data.qpos[qa] += self.np_random.uniform(-np.deg2rad(2), np.deg2rad(2))

        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        if self.data.act.size: 
            self.data.act[:] = 0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0

        mujoco.mj_forward(self.model, self.data) 

        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = self.data.qpos[self.qadr[i]]

        self.prev_u = None

        
        return self._obs(), {}

    def step(self, action):
        
        self.step_ctr += 1

        # [-1, 1]로 클립 후 → actuator ctrlrange에 맞춰 스케일링
        action = np.clip(action, -1.0, 1.0)
        self._set_ctrl_from_action(action)

        # 시뮬레이션 진행
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        # ----- 관측에서 보상 계산에 쓸 항목들 -----
        # 선속도 x (free root의 qvel[0]이 world 좌표계 x-속도)
        xd = float(self.data.qvel[0])
        zd = float(self.data.qvel[2])

        # free joint 쿼터니언 (w, x, y, z)
        qw, qx, qy, qz = self.data.qpos[3:7]

        # RPY는 기존 유틸 사용 (torso 기준)
        roll, pitch, yaw = self._torso_rpy()

        # ----- 보상: 너가 준 구조 그대로 -----
        # 속도 일치 보상 (스케일×10)
        velocity_rew = (1.0 / (abs(xd - self.v_tar) + 1.0) - 1.0 / (self.v_tar + 1.0)) * 10.0

        # yaw 편차(자세 편차)를 쿼터니언 스칼라부(qw)로부터 계산: 2*acos(qw)
        # 수치 안전을 위해 clip
        yaw_rew = np.square(2 ** math.acos(qw)) * .7
        ctrl_pen_rew = np.mean(np.square(action)) * 0.01 # 확인필요
        zd_rew = np.square(zd) * 0.5
        # pitch, roll 페널티
        pitch_rew = (pitch * 4.0) ** 2
        roll_rew  = (roll  * 3.0) ** 2

        # 최종 보상
        reward = velocity_rew - yaw_rew - zd_rew

        done = self.step_ctr > self.max_steps

        obs = self._obs()
        info = {
            "xd": xd,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "reward_terms": {
                "velocity_rew": velocity_rew,
                "yaw_rew": yaw_rew,
                "pitch_rew": pitch_rew,
                "roll_rew": roll_rew,
            }
        }
        return obs, reward, False, done, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
