from cas4160.infrastructure.utils import *


# Replaybuffer for preference triplet
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.max_size = capacity
        self.size = 0
        self.observations_1 = None
        self.observations_2 = None
        self.actions_1 = None
        self.actions_2 = None
        self.prefs = None

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations_1": self.observations_1[rand_indices],
            "observations_2": self.observations_2[rand_indices],
            "actions_1": self.actions_1[rand_indices],
            "actions_2": self.actions_2[rand_indices],
            "prefs": self.prefs[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation_1: np.ndarray,
        observation_2: np.ndarray,
        action_1: np.ndarray,
        action_2: np.ndarray,
        prefs: np.ndarray,
    ):
        """
        Insert a two trajectories into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation_1=observation_1,
                observation_2=observation_2,
                action_1=action_1,
                action_2=action_2,
                prefs=prefs
            )
        """
        if isinstance(prefs, float):
            prefs = np.array([prefs], dtype=np.float32)

        if self.observations_1 is None:
            self.observations_1 = np.empty((self.max_size, *observation_1.shape), dtype=observation_1.dtype)
            self.observations_2 = np.empty((self.max_size, *observation_2.shape), dtype=observation_2.dtype)
            self.actions_1 = np.empty((self.max_size, *action_1.shape), dtype=action_1.dtype)
            self.actions_2 = np.empty((self.max_size, *action_2.shape), dtype=action_2.dtype)
            self.prefs = np.empty((self.max_size, *prefs.shape), dtype=prefs.dtype)

        assert observation_1.shape == self.observations_1.shape[1:]
        assert observation_2.shape == self.observations_2.shape[1:]
        assert action_1.shape == self.actions_1.shape[1:]
        assert action_2.shape == self.actions_2.shape[1:]

        self.observations_1[self.size % self.max_size] = observation_1
        self.observations_2[self.size % self.max_size] = observation_2
        self.actions_1[self.size % self.max_size] = action_1
        self.actions_2[self.size % self.max_size] = action_2
        self.prefs[self.size % self.max_size] = prefs
        self.size += 1