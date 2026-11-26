"""Run with the pinned Dreamer checkout on PYTHONPATH."""
import importlib.util
import pickle
import tempfile
import unittest

import numpy as np
if importlib.util.find_spec("embodied") is None:
    raise unittest.SkipTest("Put the pinned Dreamer checkout on PYTHONPATH to run the replay tests")
from embodied.core.replay import Replay
from reward_replay import install_reward_sampling


class RetainedReplayTest(unittest.TestCase):
    def test_rare_sequence_survives_eviction_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            def build(online=False):
                replay = Replay(length=5, capacity=19, directory=folder, chunksize=7,
                                online=online, save_wait=True)
                return install_reward_sampling(replay, 1, 1., 7, retain=True)

            def insert(replay, index, reward):
                replay.add(dict(reward=np.float32(reward), action=np.int32(index % 9),
                                image=np.array([index], np.int32), is_first=np.bool_(index == 0),
                                is_last=np.bool_(False)), worker=0)

            earlier = {9: 1., 17: 4., 29: 3.}
            replay = build()
            for index in range(110):
                insert(replay, index, earlier.get(index, .02 if index % 8 == 0 else 0.))
            stats = replay.stats()
            self.assertEqual(stats["live_large_windows"], 0)
            self.assertEqual(stats["retained_windows"], 3)
            batch = replay.sample(32)
            for images, actions, rewards in zip(batch["image"][:, :, 0], batch["action"], batch["reward"]):
                end = int(images[-1])
                self.assertEqual(rewards[-1], earlier[end])
                np.testing.assert_array_equal(images, np.arange(end - 4, end + 1))
                np.testing.assert_array_equal(actions, images % 9)
            for mode in ("report", "eval"):
                self.assertLess(replay.sample(32, mode)["reward"].max(), 1.)

            restored = build(online=True)
            restored.load(pickle.loads(pickle.dumps(replay.save())))
            self.assertEqual(restored.stats()["retained_windows"], 3)
            restored.queue.append(restored.items[max(restored.items)])
            batch = restored.sample(2)
            self.assertLess(batch["reward"][0].max(), 1.)
            self.assertGreaterEqual(batch["reward"][1, -1], 1.)
            self.assertEqual(restored.stats()["retained_draws"], 1)
            for index in range(110, 121):
                insert(restored, index, 4. if index == 116 else 0.)
            restored.queue.clear()
            self.assertGreater(restored.stats()["live_large_windows"], 0)
            restored.sample(32)
            self.assertEqual(restored.stats()["retained_draws"], 0)


if __name__ == "__main__":
    unittest.main()
