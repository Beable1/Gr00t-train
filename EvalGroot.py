#!/usr/bin/env python3
"""
GR00T Delta EE Controller for R1 Humanoid Robot  (via RunPod WebSocket)

Same architecture as diffusion_policy_control_delta.py, but instead of
running the model locally, it sends observation (delta EE state + image)
to a RunPod GR00T server and receives the delta action back.

Inference flow (10 Hz):
  1. /joint_states -> FK -> current EE state
  2. delta_state = current_ee - prev_ee           (17 dims)
  3. Send delta_state + image -> RunPod WebSocket
  4. Receive delta_action from RunPod              (17 dims)
  5. target_ee = current_ee + delta_action
  6. IK(target_ee[:6]) -> arm joints (6)
  7. gripper = current_gripper + delta_action[6:]  (direct)
  8. Send /joint_command

Usage:
  python GR00T/EvalGroot.py

Commands:
  ENTER  - Start/stop control (goes home first)
  h      - Go to home position
  q      - Quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import numpy as np
import threading
import time
import cv2
import copy
import json
import base64

import pinocchio as pin
import websocket

try:
    import websocket._handshake
    websocket._handshake.get_default_user_agent = lambda: "Mozilla/5.0"
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RUNPOD_WS_URL = "ws://194.68.245.42:22066"


# ---------------------------------------------------------------------------
# IKSolver – same as data_collection_keyboard_delta.py (with custom limits)
# ---------------------------------------------------------------------------
class IKSolver:
    def __init__(self, urdf_path, ee_joint_name, controlled_joints):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getJointId(ee_joint_name)
        self.controlled_joints = controlled_joints

        self.q_idx = []
        self.v_idx = []
        for name in self.controlled_joints:
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                self.q_idx.append(self.model.joints[j_id].idx_q)
                self.v_idx.append(self.model.joints[j_id].idx_v)

        try:
            if self.model.existJointName("right_shoulder_link_joint"):
                j_id = self.model.getJointId("right_shoulder_link_joint")
                idx_q = self.model.joints[j_id].idx_q
                if idx_q < self.model.nq:
                    self.model.lowerPositionLimit[idx_q] = -1.1
                    self.model.upperPositionLimit[idx_q] = 1.1
        except Exception:
            pass

    def solve(self, q_current_full, target_pos, target_rot=None, max_iter=50, eps=1e-3):
        q = np.copy(q_current_full)
        target_se3 = pin.SE3(
            target_rot if target_rot is not None else np.eye(3), target_pos
        )
        damp = 1e-4
        dt = 0.5

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            iMd = self.data.oMi[self.ee_id].actInv(target_se3)
            err = pin.log(iMd).vector

            if np.linalg.norm(err) < eps:
                break

            J = pin.computeJointJacobian(self.model, self.data, q, self.ee_id)
            J_masked = np.zeros_like(J)
            for vj in self.v_idx:
                J_masked[:, vj] = J[:, vj]

            v = J_masked.T @ np.linalg.solve(
                J_masked @ J_masked.T + damp * np.eye(6), err
            )
            q = pin.integrate(self.model, q, v * dt)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        return q


# ---------------------------------------------------------------------------
# Controller Node
# ---------------------------------------------------------------------------
class GrootDeltaController(Node):

    URDF_PATH = "/home/beable/IsaacLab/IsaacLab/r1-new/urdf/r1-new.urdf"
    EE_JOINT = "wrist_pitch_joint_r"

    RIGHT_ARM_JOINTS = [
        "right_shoulder_link_joint",
        "right_arm_top_link_joint",
        "right_arm_bottom_link_joint",
        "right_forearm_link_joint",
        "wrist_pitch_joint_r",
        "wrist_roll_joint_r",
        "thumb_joint_roll_r",
        "index_proximal_joint_r",
        "middle_proximal_joint_r",
        "ring_proximal_joint_r",
        "little_proximal_joint_r",
        "thumb_proximal_joint_r",
        "index_proximal_joint_r_1",
        "middle_proximal_joint_r_1",
        "ring_proximal_joint_r_1",
        "little_proximal_joint_r_1",
        "thumb_proximal_joint_r_1",
    ]

    IK_CONTROLLED_JOINTS = [
        "right_shoulder_link_joint",
        "right_arm_top_link_joint",
        "right_arm_bottom_link_joint",
        "right_forearm_link_joint",
        "wrist_roll_joint_r",
        "wrist_pitch_joint_r",
    ]

    HOME_POSITIONS = {
        "right_shoulder_link_joint": 0.17,
        "right_arm_top_link_joint": -1.46,
        "right_arm_bottom_link_joint": 0.35,
        "right_forearm_link_joint": 1.7,
        "wrist_pitch_joint_r": 0.0,
        "wrist_roll_joint_r": 0.0,
        "thumb_joint_roll_r": 0.0,
        "index_proximal_joint_r": 0.0,
        "middle_proximal_joint_r": 0.0,
        "ring_proximal_joint_r": 0.0,
        "little_proximal_joint_r": 0.0,
        "thumb_proximal_joint_r": 0.0,
        "index_proximal_joint_r_1": 0.0,
        "middle_proximal_joint_r_1": 0.0,
        "ring_proximal_joint_r_1": 0.0,
        "little_proximal_joint_r_1": 0.0,
        "thumb_proximal_joint_r_1": 0.0,
    }

    def __init__(self, runpod_url: str):
        super().__init__("groot_delta_controller")

        self.runpod_url = runpod_url
        self.vid_H = 360
        self.vid_W = 640
        self.state_dim = 17

        # ---- Pinocchio FK model ----
        self.pin_model = pin.buildModelFromUrdf(self.URDF_PATH)
        self.pin_data = self.pin_model.createData()
        if self.pin_model.existJointName(self.EE_JOINT):
            self.ee_joint_id = self.pin_model.getJointId(self.EE_JOINT)
        else:
            self.ee_joint_id = self.pin_model.getFrameId(self.EE_JOINT)

        # ---- IK solver ----
        self.ik_solver = IKSolver(
            self.URDF_PATH, self.EE_JOINT, self.IK_CONTROLLED_JOINTS
        )

        # ---- Sensor state ----
        self.current_joint_positions = {n: 0.0 for n in self.RIGHT_ARM_JOINTS}
        self.current_rgb_image = np.zeros(
            (self.vid_H, self.vid_W, 3), dtype=np.uint8
        )
        self.previous_ee_state: np.ndarray | None = None

        # ---- WebSocket client ----
        self.ws_client: websocket.WebSocket | None = None
        self._connect_to_runpod()

        # ---- ROS2 subscribers / publisher ----
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self.create_subscription(Image, "/rgb", self._rgb_cb, 10)
        self.joint_cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        # ---- Control loop ----
        self.control_rate = 10.0
        self.create_timer(1.0 / self.control_rate, self._control_loop)
        self.is_running = False
        self.inference_count = 0

        self.get_logger().info(
            f"GR00T Controller ready  (RunPod: {self.runpod_url})"
        )
        self.get_logger().info("Commands:  ENTER=toggle  h=home  q=quit")

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    def _connect_to_runpod(self):
        try:
            self.get_logger().info(f"Connecting to RunPod at {self.runpod_url} ...")
            self.ws_client = websocket.create_connection(self.runpod_url, timeout=10)
            self.get_logger().info("Connected to RunPod.")
        except Exception as e:
            self.get_logger().error(f"RunPod connection failed: {e}")
            self.ws_client = None

    def _send_to_runpod(self, state_vec: list[float], image_bgr: np.ndarray) -> list[float] | None:
        if self.ws_client is None:
            self._connect_to_runpod()
        if self.ws_client is None:
            return None

        try:
            _, buf = cv2.imencode(
                ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            image_b64 = base64.b64encode(buf).decode("utf-8")

            payload = json.dumps({"state": state_vec, "image": image_b64})
            self.ws_client.send(payload)
            resp_str = self.ws_client.recv()
            resp = json.loads(resp_str)

            if "error" in resp:
                self.get_logger().error(f"RunPod error: {resp['error']}")
                return None

            action = resp.get("action", None)
            if isinstance(action, list) and len(action) == 17:
                return [float(x) for x in action]

            self.get_logger().error(f"Invalid action from RunPod: {action}")
            return None
        except Exception as e:
            self.get_logger().error(f"WebSocket error: {e}")
            self.ws_client = None
            return None

    # ------------------------------------------------------------------
    # FK  (identical to diffusion_policy_control_delta.py)
    # ------------------------------------------------------------------
    def _joint_positions_to_ee(self, joint_positions: list[float]) -> np.ndarray:
        q = pin.neutral(self.pin_model)
        for name, p in zip(self.RIGHT_ARM_JOINTS, joint_positions):
            if self.pin_model.existJointName(name):
                j_id = self.pin_model.getJointId(name)
                idx_q = self.pin_model.joints[j_id].idx_q
                if idx_q < self.pin_model.nq:
                    q[idx_q] = p

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        se3 = self.pin_data.oMi[self.ee_joint_id]
        pos = se3.translation.copy()
        rpy = pin.rpy.matrixToRpy(se3.rotation)
        gripper = np.array(joint_positions[6:], dtype=np.float64)
        return np.concatenate([pos, rpy, gripper])

    # ------------------------------------------------------------------
    # IK  – delta_action (17) -> joint command (17)
    # ------------------------------------------------------------------
    def _apply_delta_action(self, delta_action: np.ndarray) -> list[float]:
        current_joints = [
            self.current_joint_positions[n] for n in self.RIGHT_ARM_JOINTS
        ]
        current_ee = self._joint_positions_to_ee(current_joints)

        target_pos = current_ee[:3] + delta_action[:3]

        target_rpy = current_ee[3:6] + delta_action[3:6]
        for i in range(3):
            target_rpy[i] = np.arctan2(np.sin(target_rpy[i]), np.cos(target_rpy[i]))
        target_rot = pin.rpy.rpyToMatrix(
            float(target_rpy[0]), float(target_rpy[1]), float(target_rpy[2])
        )

        q_init = pin.neutral(self.ik_solver.model)
        for name, p in zip(self.RIGHT_ARM_JOINTS, current_joints):
            if self.ik_solver.model.existJointName(name):
                j_id = self.ik_solver.model.getJointId(name)
                idx_q = self.ik_solver.model.joints[j_id].idx_q
                if idx_q < self.ik_solver.model.nq:
                    q_init[idx_q] = p

        q_result = self.ik_solver.solve(q_init, target_pos, target_rot)

        arm_joints: list[float] = []
        for name in self.RIGHT_ARM_JOINTS[:6]:
            j_id = self.ik_solver.model.getJointId(name)
            idx_q = self.ik_solver.model.joints[j_id].idx_q
            arm_joints.append(float(q_result[idx_q]))

        current_gripper = np.array(current_joints[6:])
        target_gripper = current_gripper + delta_action[6:]

        return arm_joints + target_gripper.tolist()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _joint_states_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.current_joint_positions:
                self.current_joint_positions[name] = msg.position[i]

    def _rgb_cb(self, msg: Image):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, -1)
            )
            enc = msg.encoding.lower()
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif enc == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            self.current_rgb_image = cv2.resize(
                img, (self.vid_W, self.vid_H), cv2.INTER_LINEAR
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Observation (same delta logic as diffusion_policy_control_delta.py)
    # ------------------------------------------------------------------
    def _prepare_observation(self) -> tuple[list[float], np.ndarray]:
        current_joints = [
            self.current_joint_positions[n] for n in self.RIGHT_ARM_JOINTS
        ]
        current_ee = self._joint_positions_to_ee(current_joints)

        if self.previous_ee_state is None:
            delta_state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            delta_state = (current_ee - self.previous_ee_state).astype(np.float32)
            for i in range(3, 6):
                delta_state[i] = np.arctan2(
                    np.sin(delta_state[i]), np.cos(delta_state[i])
                )
        self.previous_ee_state = current_ee.copy()

        return delta_state.tolist(), self.current_rgb_image.copy()

    # ------------------------------------------------------------------
    # Control loop  (10 Hz)
    # ------------------------------------------------------------------
    def _control_loop(self):
        if not self.is_running:
            return
        try:
            state_vec, image_bgr = self._prepare_observation()

            delta_action_list = self._send_to_runpod(state_vec, image_bgr)

            if delta_action_list is None:
                self.get_logger().warn("No action from RunPod, skipping step.")
                return

            delta_action = np.array(delta_action_list, dtype=np.float64)

            joint_cmd = self._apply_delta_action(delta_action)
            self._send_joint_command(joint_cmd)

            self.inference_count += 1
            if self.inference_count % 10 == 0:
                self.get_logger().info(
                    f"#{self.inference_count}  delta_xyz=[{delta_action[0]:.4f}, "
                    f"{delta_action[1]:.4f}, {delta_action[2]:.4f}]"
                )

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_control()

    # ------------------------------------------------------------------
    # Joint command publisher
    # ------------------------------------------------------------------
    def _send_joint_command(self, positions: list[float]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = list(self.RIGHT_ARM_JOINTS)
        msg.position = positions
        msg.velocity = [0.0] * len(self.RIGHT_ARM_JOINTS)
        msg.effort = [0.0] * len(self.RIGHT_ARM_JOINTS)
        self.joint_cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Start / stop / home
    # ------------------------------------------------------------------
    def go_home(self, duration_sec: float = 3.0):
        self.get_logger().info("Moving to home position...")
        home = [self.HOME_POSITIONS[n] for n in self.RIGHT_ARM_JOINTS]
        steps = int(duration_sec * self.control_rate)
        for _ in range(steps):
            self._send_joint_command(home)
            time.sleep(1.0 / self.control_rate)
        self.get_logger().info("Home position reached")

    def start_control(self):
        self.previous_ee_state = None
        self.inference_count = 0

        current_joints = [
            self.current_joint_positions[n] for n in self.RIGHT_ARM_JOINTS
        ]
        self.previous_ee_state = self._joint_positions_to_ee(current_joints).copy()

        self.is_running = True
        self.get_logger().info("Control STARTED")

    def stop_control(self):
        self.is_running = False
        self.get_logger().info("Control STOPPED")

    def toggle_control(self):
        if self.is_running:
            self.stop_control()
        else:
            self.start_control()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    controller = GrootDeltaController(runpod_url=RUNPOD_WS_URL)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    print("\n" + "=" * 60)
    print("GR00T DELTA EE CONTROLLER  (via RunPod)")
    print("=" * 60)
    print("\nCommands:")
    print("  h      - Go to home position")
    print("  ENTER  - Start/stop control (goes home first)")
    print("  q      - Quit")
    print("-" * 60)

    time.sleep(1.0)

    try:
        while rclpy.ok():
            cmd = input("\n> ").strip().lower()
            if cmd == "q":
                print("Exiting...")
                break
            elif cmd == "h":
                controller.go_home()
            else:
                controller.toggle_control()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        controller.stop_control()
        if controller.ws_client is not None:
            try:
                controller.ws_client.close()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        executor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
