import abc
import warnings
from os import path

import glfw
import gym
import numpy as np
from d4rl.kitchen.adept_envs.simulation import module
from gym import error
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                "You must call env.set_task before using env." + func.__name__
            )
        return func(*args, **kwargs)

    return inner


DEFAULT_SIZE = 500


class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 150

    def __init__(self, model_path, frame_skip, rgb_array_res=(640, 480)):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip

        self._use_dm_backend = True
        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_path.endswith(".mjb"):
                self.sim = dm_mujoco.Physics.from_binary_path(model_path)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_path)
            self.model = self.sim.model
            self._patch_mjlib_accessors(self.model, self.sim.data)
        else:  # Use mujoco_py
            mujoco_py = module.get_mujoco_py()
            self.model = mujoco_py.load_model_from_path(model_path)
            self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self._rgb_array_res = rgb_array_res

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.seed()

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self._use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API."""
        assert self._use_dm_backend
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), name.encode()
            )
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
            return obj_id

        if not hasattr(model, "body_name2id"):
            model.body_name2id = lambda name: name2id("body", name)

        if not hasattr(model, "geom_name2id"):
            model.geom_name2id = lambda name: name2id("geom", name)

        if not hasattr(model, "site_name2id"):
            model.site_name2id = lambda name: name2id("site", name)

        if not hasattr(model, "joint_name2id"):
            model.joint_name2id = lambda name: name2id("joint", name)

        if not hasattr(model, "actuator_name2id"):
            model.actuator_name2id = lambda name: name2id("actuator", name)

        if not hasattr(model, "camera_name2id"):
            model.camera_name2id = lambda name: name2id("camera", name)

        if not hasattr(data, "body_xpos"):
            data.body_xpos = data.xpos

        if not hasattr(data, "body_xquat"):
            data.body_xquat = data.xquat

        if not hasattr(data, "get_body_xpos"):
            data.get_body_xpos = lambda name: data.body_xpos[model.body_name2id(name)]

        if not hasattr(data, "get_body_xquat"):
            data.get_body_xquat = lambda name: data.body_xquat[model.body_name2id(name)]

        if not hasattr(data, "get_mocap_pos"):
            data.get_mocap_pos = lambda name: data.mocap_pos[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "get_mocap_quat"):
            data.get_mocap_quat = lambda name: data.mocap_quat[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "set_mocap_pos"):

            def set_mocap_pos(name, value):
                data.mocap_pos[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_pos = lambda name, value: set_mocap_pos(name, value)

        if not hasattr(data, "set_mocap_quat"):

            def set_mocap_quat(name, value):
                data.mocap_quat[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_quat = lambda name, value: set_mocap_quat(name, value)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        if self._use_dm_backend:
            self.sim.set_state(np.concatenate((qpos, qvel)))
        else:
            old_state = self.sim.get_state()
            new_state = mujoco_py.MjSimState(
                old_state.time, qpos, qvel, old_state.act, old_state.udd_state
            )
            self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, "curr_path_length", 0) > self.max_path_length:
            raise ValueError(
                "Maximum path length allowed by the benchmark has been exceeded"
            )
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render(self, mode="human"):
        if mode == "human":
            self._get_viewer(mode).render()
        elif mode == "rgb_array":
            return self.sim.render(
                *self._rgb_array_res, mode="offscreen", camera_name="topview"
            )[:, :, ::-1]
        else:
            raise ValueError("mode can only be either 'human' or 'rgb_array'")

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
