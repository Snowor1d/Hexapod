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
LEGS = ["fl","fr","ml","mr","rl","rr"]

LEG_GEOMS = {
    "fl": ["coxa_fl","femur_fl","tibia_fl","foot_fl"],
    "fr": ["coxa_fr","femur_fr","tibia_fr","foot_fr"],
    "ml": ["coxa_ml","femur_ml","tibia_ml","foot_ml"],
    "mr": ["coxa_mr","femur_mr","tibia_mr","foot_mr"],
    "rl": ["coxa_rl","femur_rl","tibia_rl","foot_rl"],
    "rr": ["coxa_rr","femur_rr","tibia_rr","foot_rr"],
}

COLOR_GRAY = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)
COLOR_BLUE = np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32)  # 앞다리 기본
COLOR_RED  = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 고장 표시

def quat_to_rpy(xquat):
    w, x, y, z = xquat
    sinr_cosp = 2*(w*x + y*z);  cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x); pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2*(w*z + x*y); cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

class HexapodEnv(Env):
    metadata = {"render_modes": ["human", "none"]}
    
    def __init__(self, xml_path="hexapod_uneven.xml", render_mode="none",
                 action_repeat=20, target_speed=0.2, obs_clip=10.0, seed=None, terrain_amp = 1.0, terrain_freq = 1.0, terrain_smooth=None):

    
        
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.action_repeat = int(action_repeat)
        self.v_tar = float(target_speed)
        self.obs_clip = float(obs_clip)
        self.max_steps = 800        

        self.step_ctr = 0
        self.act_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS])
        self.j_ids   = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    n) for n in JOINTS])
        self.kp_per_act = self.model.actuator_gainprm[self.act_ids, 0].copy()
        self._prev_u_ref = np.zeros(len(self.act_ids), dtype=np.float32)
        self.qadr    = self.model.jnt_qposadr[self.j_ids]
        self.vadr    = self.model.jnt_dofadr[self.j_ids]

        # 다리별 (3조인트/3액추에이터) 슬라이스
        self.leg_joint_slices = {leg: slice(i*3, i*3+3) for i, leg in enumerate(LEGS)}
        self.leg_act_slices   = self.leg_joint_slices.copy()

        # ctrlrange 
        cr = self.model.actuator_ctrlrange[self.act_ids]
        self.ctrl_min = cr[:, 0].copy()
        self.ctrl_max = cr[:, 1].copy()

        # 관측: IMU6 + q18 + qd18 + contacts6 + failure6
        obs_dim = 6 + 18 + 18 + 6 + 6
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
        self._feet_geoms = ["foot_fl","foot_fr","foot_ml","foot_mr","foot_rl","foot_rr"]

        # 실패 상태
        self.failed_leg_idx = None
        self.failure_mask01 = np.zeros(6, dtype=np.float32)

        # 시각화용 geom id
        self.geom_id_map = {leg: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) 
                                  for g in names] for leg, names in LEG_GEOMS.items()}

        # ★ 무력화/복구용 원본 파라미터 백업
        self._orig_gainprm = self.model.actuator_gainprm.copy()       # (nu, 10)
        self._orig_forcerange = self.model.actuator_forcerange.copy() # (nu, 2)

        # touch_torso 센서의 시작 인덱스(adr) 가져오기
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_torso")
        if sid == -1:
            raise RuntimeError("Sensor 'touch_torso' not found in XML.")
        self.torso_touch_adr = int(self.model.sensor_adr[sid])   # sensordata 시작 위치
        self.torso_touch_dim = int(self.model.sensor_dim[sid])   # 보통 1
        assert self.torso_touch_dim == 1

        # 접촉 감지 임계값 (필요시 조정)
        self.torso_touch_thresh = 1e-4

        # (선택) 디바운스용 간단한 이동평균
        self._touch_ma = 0.0
        self._touch_ma_tau = 0.2  # 0~1, 클수록 반응 빠름

        self.terrain_amp = float(terrain_amp)
        self.terrain_freq = float(terrain_freq)
        self.terrain_smooth = terrain_smooth
        
        self.hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "hill")
        assert self.hfield_id != -1, "hfield 'hill이 없음"
        self._hfield_size0 = self.model.hfield_size[self.hfield_id].copy()
        self._apply_terrain()

    def _apply_terrain(self):
        """terrain_amp / terrain_freq / terrain_smooth 값을 hfield에 반영."""
        # 원본에서 시작
        base = self._hfield_size0.copy()
        size_x, size_y, size_z, smooth = base

        # ● 높이 스케일: 더 크면 더 큰 기복(난이도 ↑)
        size_z = size_z * self.terrain_amp

        # ● 공간 주파수: 커질수록 같은 픽셀이 더 촘촘히 배치되므로 기복이 잦아짐(난이도 ↑)
        #   주파수 k배 => 가로/세로 스케일을 1/k로 줄임
        k = max(1e-6, self.terrain_freq)
        size_x = size_x / k
        size_y = size_y / k

        # ● 스무딩(접촉 필터): 클수록 표면이 매끈(난이도 ↓). None이면 기존 유지
        if self.terrain_smooth is not None:
            smooth = float(self.terrain_smooth)

        self.model.hfield_size[self.hfield_id, 0] = size_x
        self.model.hfield_size[self.hfield_id, 1] = size_y
        self.model.hfield_size[self.hfield_id, 2] = size_z
        self.model.hfield_size[self.hfield_id, 3] = smooth

        # 사이즈 변경 후 전진계산
        mujoco.mj_forward(self.model, self.data)

    def _color_leg(self, leg, rgba):
        for gid in self.geom_id_map[leg]:
            if gid != -1:
                self.model.geom_rgba[gid] = rgba

    def _apply_leg_colors(self):
        # 기본 회색
        for leg in LEGS: self._color_leg(leg, COLOR_GRAY)
        # 앞다리 파랑
        self._color_leg("fl", COLOR_BLUE)
        self._color_leg("fr", COLOR_BLUE)
        # 고장 빨강(우선)
        if self.failed_leg_idx is not None:
            self._color_leg(LEGS[self.failed_leg_idx], COLOR_RED)

    # ===== 고장(무력화) 제어 =====
    def _restore_all_actuators(self):
        self.model.actuator_gainprm[:]   = self._orig_gainprm
        self.model.actuator_forcerange[:] = self._orig_forcerange

    def _disable_leg_actuators(self, leg):
        """해당 다리의 3개 액추에이터를 완전 무력화: kp=0, forcerange=0"""
        s = self.leg_act_slices[leg]
        act_idxs = np.arange(s.start, s.stop)
        # position actuator는 gainprm[0]이 kp
        self.model.actuator_gainprm[self.act_ids[act_idxs], 0] = 0.0
        # 안전하게 힘 상한도 0으로
        self.model.actuator_forcerange[self.act_ids[act_idxs], 0] = 0.0
        self.model.actuator_forcerange[self.act_ids[act_idxs], 1] = 0.0

    def _set_ctrl_from_action(self, a):
        # [-1, 1] → [ctrl_min, ctrl_max]
        u = (a + 1.0) * 0.5 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min
        self.data.ctrl[self.act_ids] = u

        return u
        # 고장 다리라도 여기선 그냥 써도 됨(어차피 kp=0/force=0 → 토크 0)

    def _obs(self):
        imu_acc  = self.data.sensordata[0:3]
        imu_gyro = self.data.sensordata[3:6]
        imu_acc_norm  = np.clip(imu_acc / 10.0,  -1.0, 1.0)
        imu_gyro_norm = np.clip(imu_gyro / 10.0, -1.0, 1.0)

        q  = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr]

        q_norm = np.zeros_like(q)
        for i in range(len(q)):
            q_min, q_max = self.ctrl_min[i], self.ctrl_max[i]
            q_norm[i] = 2.0 * (q[i] - q_min) / (q_max - q_min) - 1.0
            q_norm[i] = np.clip(q_norm[i], -1.0, 1.0)
        qd_norm = np.clip(qd / 5.0, -1.0, 1.0)

        contacts = (self.data.sensordata[6:12] > 1e-4).astype(np.float32)
        contacts_norm = contacts * 2.0 - 1.0

        failure_pm1 = (self.failure_mask01 * 2.0 - 1.0).astype(np.float32)

        obs = np.concatenate([
            imu_acc_norm,
            imu_gyro_norm,
            q_norm,
            qd_norm,
            contacts_norm,
            failure_pm1,
        ]).astype(np.float32)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _torso_rpy(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        xquat = self.data.xquat[bid]
        return quat_to_rpy(xquat)

    def reset(self, *, seed=None, options=None):
        self.step_ctr = 0

        # 액추에이터 파라미터 원복 후 초기화
        self._restore_all_actuators()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)
        for j, qa in enumerate(self.qadr):
            self.data.qpos[qa] += self.np_random.uniform(-np.deg2rad(2), np.deg2rad(2))
        self.data.qvel[:] = 0; self.data.qacc[:] = 0
        if self.data.act.size:  self.data.act[:]  = 0
        if self.data.ctrl.size: self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # 초기 ctrl = 현재 qpos (학습 안정화용)
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = self.data.qpos[self.qadr[i]]

        # 무작위 1다리 고장
        self.failed_leg_idx = int(self.np_random.randint(0, 6))
        self.failure_mask01[:] = 0
        self.failure_mask01[self.failed_leg_idx] = 1.0

        # 실제 무력화 적용
        self._disable_leg_actuators(LEGS[self.failed_leg_idx])

        # 시각화 색상
        self._apply_leg_colors()

        return self._obs(), {}

    def step(self, action):
        self.step_ctr += 1

        action = np.clip(action, -1.0, 1.0)
        u_ref = self._set_ctrl_from_action(action)
        touch_hits = 0
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)

            force = float(self.data.sensordata[self.torso_touch_adr])
            if force > self.torso_touch_thresh:
                touch_hits += 1

            if self.viewer is not None:
                self.viewer.sync()

        
        touch_ratio = touch_hits / float(self.action_repeat)
        touch_pen = touch_ratio * 5
        
        
        #print("touch ratio : ", touch_ratio)

        xd = float(self.data.qvel[0])
        zd = float(self.data.qvel[2])
        zpos = float(self.data.qpos[2])
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll, pitch, yaw = self._torso_rpy()
        
        # 보상
        velocity_rew = (1.0 / (abs(xd - self.v_tar) + 1.0) - 1.0 / (self.v_tar + 1.0)) * 10.0
        yaw_rew = np.square(yaw*0.3)
        #print(yaw_rew)
        roll_rew = (roll*0.5)**2
        #print("yaw_rew : ", yaw_rew)
        #print(yaw)
        q_curr = self.data.qpos[self.qadr] #현재 관절각
        q_err = u_ref - q_curr
        kp = self.kp_per_act
        effort_proxy = kp * q_err
        zd_rew = np.square(zd) * 0.5

        # (선택) 무력화된 다리의 액션은 페널티에서 제외
        if self.failed_leg_idx is not None:
            s = self.leg_act_slices[LEGS[self.failed_leg_idx]]
            mask = np.ones_like(effort_proxy, dtype=bool)
            mask[s] = False
        else:
            mask = np.ones_like(effort_proxy, dtype=bool)

        err_pen = 0.0001 * np.mean(np.square(effort_proxy[mask]))
       # reward = velocity_rew - yaw_rew - roll_rew - err_pen# 필요시 ctrl_pen_rew 등 추가
        #print("velocity_rew : ", velocity_rew)
        #print("yaw_rew : ", yaw_rew)
        #print("roll_rew : ", roll_rew)
        #print("err_pen : ", err_pen)
        print("velocity_rew : ", velocity_rew)
        print("yaw_rew : ", yaw_rew)
        print("touch_pen : ", touch_pen)
        reward = velocity_rew - yaw_rew - touch_pen# - roll_rew - err_pen - touch_pen - zd_rew
        done = self.step_ctr > self.max_steps

        obs = self._obs()
       # print(obs)
        info = {
            "xd": xd,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "failed_leg": LEGS[self.failed_leg_idx],
            "reward_terms": {
                "velocity_rew": velocity_rew,
                "yaw_rew": yaw_rew,
                "err_pen": err_pen,
            }
        }
        return obs, reward, False, done, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
