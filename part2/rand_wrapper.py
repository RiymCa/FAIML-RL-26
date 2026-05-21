import numpy as np
import gymnasium as gym
from collections import deque

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper che applica Domain Randomization (UDR o ADR) all'ambiente Panda-Gym.
    """
    def __init__(
            self,
            env,
            mass_init,
            mass_range=(0.1, 10.0),
            mode="none",  # "none", "udr", o "adr"
            env_type="source", # "source" o "target"
            p_b=0.5,  # Probabilità di campionare ai bordi (ADR)
            buffer_size=10,  # Dimensione della memoria per valutare la difficoltà 30/50
            high_threshold=0.8,  # Soglia di successo per espandere la difficoltà  0.7
            low_threshold=0.2,  # Soglia di fallimento per ridurre la difficoltà
            step_size=0.2,  # Quanto allargare/stringere il range ad ogni step     0.05
            verbose=False
    ):
        super().__init__(env)

        self.verbose = verbose
        self.mode = "none" if mode not in ["udr", "adr", "none"] else mode
        self.env_type = "source" if env_type not in ["source", "target"] else env_type

        self.nominal_mass = mass_init
        self.global_mass_min, self.global_mass_max = mass_range

        if mode == "adr":
            self.current_mass_min = max(self.global_mass_min, self.nominal_mass - step_size / 2)
            self.current_mass_max = min(self.global_mass_max, self.nominal_mass + step_size / 2)
        else:
            self.current_mass_min = self.global_mass_min
            self.current_mass_max = self.global_mass_max

        self.p_b = p_b
        self.buffer_size = buffer_size
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.step_size = step_size

        self.low_buffer = deque(maxlen=buffer_size)
        self.high_buffer = deque(maxlen=buffer_size)

        self.last_sample_type = "default"
        self.np_random = np.random.default_rng()

    # -----------------------
    # Campionamento Massa
    # -----------------------
    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "default"
            return self.nominal_mass

        if self.mode == "adr":
            if self.np_random.random() < self.p_b:
                if self.np_random.random() < 0.5:
                    self.last_sample_type = "low_boundary"
                    return self.current_mass_min
                else:
                    self.last_sample_type = "high_boundary"
                    return self.current_mass_max

        # UDR
        self.last_sample_type = "uniform"
        return self.np_random.uniform(self.current_mass_min, self.current_mass_max)

    # -----------------------
    # Step: Aggiornamento ADR
    # -----------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        if self.mode == "adr" and done:
            success = float(info.get("is_success", False))

            if self.last_sample_type == "low_boundary":
                self.low_buffer.append(success)

                if len(self.low_buffer) >= self.buffer_size:
                    avg_perf = np.mean(self.low_buffer)

                    if avg_perf >= self.high_threshold:
                        self.current_mass_min = max(self.global_mass_min, self.current_mass_min - self.step_size)
                    elif avg_perf <= self.low_threshold:
                        self.current_mass_min = min(self.nominal_mass, self.current_mass_min + self.step_size)

                    self.low_buffer.clear()

            elif self.last_sample_type == "high_boundary":
                self.high_buffer.append(success)

                if len(self.high_buffer) >= self.buffer_size:
                    avg_perf = np.mean(self.high_buffer)

                    if avg_perf >= self.high_threshold:
                        self.current_mass_max = min(self.global_mass_max, self.current_mass_max + self.step_size)
                    elif avg_perf <= self.low_threshold:
                        self.current_mass_max = max(self.nominal_mass, self.current_mass_max - self.step_size)

                    self.high_buffer.clear()

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset: Applicazione Fisica
    # -----------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        obs, info = super().reset(seed=seed, options=options)

        new_mass = self._sample_mass()

        sim = self.env.unwrapped.task.sim
        object_body_id = sim._bodies_idx["object"]
        sim.physics_client.changeDynamics(
            bodyUniqueId=object_body_id,
            linkIndex=-1,
            mass=float(new_mass),
        )

        if self.verbose:
            width = self.current_mass_max - self.current_mass_min
            print(
                f"[{self.mode}] mass={new_mass:.2f} | "
                f"range=[{self.current_mass_min:.2f}, {self.current_mass_max:.2f}] | "
                f"type={self.last_sample_type:<13} | " 
                f"width={width:.2f}"
            )

        return obs, info