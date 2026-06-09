import os

# EvalCallBack is an evaluation function that every N steps on the training evaluates the model and if it gets a
# record score it saves it in best_model.zip
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

"""
    This class has the same goal of EvalCallback, saving the model, but it allows us to save also the normalization
    file.
    
    How it works: 
        - the model receives the spatial coordinates of the position of the arm, the block and the target spot,
        then it normalizes them keeping track of mean and std of each one, that both changes during training.
        - It outputs a choice of distances [dx, dy, dz] not normalized, normal.
        - After a while it gets evaluated here, if it performs better than the previous one we save the model and the
        stats. (Note that the stats and the model weights keep changing later in the training, in the testing phase
        we only take the best performing model with the stats)
    
    The idea behind it is that sometimes during training the models reach an optimum solution, but if it keeps trying
    it might start to explore other actions or overfitting causing it to modify its parameter and loosing accuracy,
    in this way we save the best model and everytime compare it to a new one. 
"""
class SyncEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, vec_env, **kwargs):
        super().__init__(eval_env, best_model_save_path=best_model_save_path, **kwargs)
        self.vec_env = vec_env
        self._last_saved_reward = float("-inf")

    """
        Check if is time to make an evaluation, updating the mean and std of the env and eval env.
        Then it gets the result from super.on_step and if the result are the best we have had it saves the model and
        its normalization file.
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            assert isinstance(self.eval_env, VecNormalize)
            self.eval_env.obs_rms = self.vec_env.obs_rms

        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.best_mean_reward > self._last_saved_reward:
                self._last_saved_reward = self.best_mean_reward
                self.eval_env.save(os.path.join(self.best_model_save_path, "vec_normalize.pkl"))

        return result