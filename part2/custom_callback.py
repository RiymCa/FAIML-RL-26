from stable_baselines3.common.callbacks import EvalCallback
import os


class SyncEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, vec_env, **kwargs):
        super().__init__(eval_env, best_model_save_path=best_model_save_path, **kwargs)
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.obs_rms = self.vec_env.obs_rms

        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.best_mean_reward == self.last_mean_reward:
                self.vec_env.save(os.path.join(self.best_model_save_path, "vec_normalize.pkl"))

        return result