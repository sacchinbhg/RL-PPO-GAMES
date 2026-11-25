"""Independent component checks. Run with python -m unittest test_world_model."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiment import Run
from policy import N_FEATURES, Policy
from world_model import ACTIONS, HORIZON, STATE, Dynamics, Replay, WorldLearner, draw_button, window_starts


class Components(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)

    def test_window_alignment(self):
        ends = torch.zeros(6, 2)
        ends[2, 0] = 1
        ends[3, 1] = 1
        self.assertEqual(window_starts(ends).tolist(), [0, 1, 3, 6])

    def test_discrete_action_and_gradient(self):
        logits = torch.randn(32, ACTIONS, requires_grad=True)
        y = draw_button(logits, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(y, F.one_hot(y.argmax(1), ACTIONS).float()))
        (y * torch.arange(ACTIONS)).sum().backward()
        self.assertGreater(logits.grad.norm().item(), 0)

    def test_buffer_wrap_and_pre_reset_successor(self):
        buf = Replay("cpu", capacity=3)
        x = torch.zeros(2, STATE)
        a = torch.tensor([1, 2])
        g = torch.zeros(2, N_FEATURES)
        buf.add(x, a, g, x+1, a.float())
        buf.add(x+2, a, g, x+3, a.float())
        self.assertEqual((buf.size, buf.cursor), (3, 1))
        start, _, _, end, _ = buf.sample(torch.Generator().manual_seed(9), 100)
        torch.testing.assert_close(end, start+1)
        self.assertEqual(set(start[:, 0].tolist()), {0., 2.})

    def test_archive_path_replacement_and_goal_credit(self):
        k = (0, -1, -1, 0, 0, 0, 0, 0, 0)
        target = (5, 1, 0, 1, 2, 2, 0, 0, 0)
        img = np.zeros((40,60,3), np.uint8)
        pool = SimpleNamespace(initial=[(img, k)], request=lambda *args: b"save")
        run = Run(pool, SimpleNamespace(seed=23, mode="world", episode=20))
        run.episodes[0] = run.episode(k, b"\x02\x01", target)
        r, _ = run.observe(0, (img, target), 7)
        self.assertEqual(r, 4.25)
        self.assertEqual(run.observe(0, (img, target), 1)[0], 0.)
        self.assertEqual(run.archive[target]["path"], bytes([2, 1, 7]))
        run.archive[target]["trials"] = 3
        run.episodes[0] = run.episode(k)
        run.observe(0, (img, target), 4)
        self.assertEqual(run.archive[target]["path"], bytes([4]))
        self.assertEqual(run.archive[target]["trials"], 3)
        self.assertEqual(run.events["goals_reached"], 1)

    def test_world_optimizes_only_actor_path_and_restarts(self):
        policy = Policy()
        opt = torch.optim.Adam(policy.parameters(), lr=.0003)
        before_rng = torch.get_rng_state()
        world = WorldLearner(19, torch.device("cpu"))
        torch.testing.assert_close(before_rng, torch.get_rng_state())
        x = torch.rand(2048, STATE)
        actions = torch.randint(ACTIONS, (2048,))
        goals = torch.zeros(2048, N_FEATURES)
        y = (.9*x + .01*actions[:,None]).clamp(0,1)
        rewards = (actions == 1).float()
        world.replay.add(x, actions, goals, y, rewards)
        world.fit(updates=2)
        old = {k: v.clone() for k,v in world.model.state_dict().items()}
        old_critic = policy.critic.weight.clone()
        old_actor = policy.actor.weight.clone()
        img, ram = torch.rand(24,3,40,60)*255, torch.rand(24,2*N_FEATURES)
        self.assertEqual(world.improve(policy, opt, img, ram, torch.ones(24)), (0.,0.))
        norm, _ = world.improve(policy, opt, img, ram, torch.full((24,), HORIZON))
        self.assertGreater(norm, 0.)
        self.assertFalse(torch.equal(old_actor, policy.actor.weight))
        torch.testing.assert_close(old_critic, policy.critic.weight, rtol=0, atol=0)
        self.assertIsNone(policy.critic.weight.grad)
        for k,v in world.model.state_dict().items():
            torch.testing.assert_close(old[k], v, rtol=0, atol=0)
        clone = WorldLearner(31, torch.device("cpu"))
        clone.restore(world.checkpoint())
        self.assertEqual(clone.imagined_steps, 24 * HORIZON)
        self.assertEqual(clone.model_updates, 2)
        for m,n in zip(clone.replay.sample(clone.rng, 10), world.replay.sample(world.rng, 10)):
            torch.testing.assert_close(m, n)

    def test_action_blind_world_produces_zero_policy_gradient(self):
        policy = Policy()
        opt = torch.optim.Adam(policy.parameters())
        world = WorldLearner(19, torch.device("cpu"))
        x = torch.zeros(2048, STATE)
        world.replay.add(x, torch.zeros(2048), torch.zeros(2048,N_FEATURES), x, torch.zeros(2048))
        world.model.requires_grad_(False)
        world.model.button.weight.zero_()
        before = policy.actor.weight.clone()
        norm, _ = world.improve(policy, opt, torch.rand(4,3,40,60)*255,
                                torch.zeros(4,2*N_FEATURES), torch.full((4,), HORIZON+1))
        self.assertEqual(norm, 0.)
        torch.testing.assert_close(policy.actor.weight, before, rtol=0, atol=0)

    def test_full_bptt_actor_receives_three_gradients_and_separate_momentum(self):
        policy = Policy()
        main_opt = torch.optim.Adam(policy.parameters())
        world = WorldLearner(5, torch.device("cpu"), "full-bptt")
        state = torch.zeros(2048, STATE)
        world.replay.add(state, torch.zeros(2048), torch.zeros(2048, N_FEATURES), state, torch.zeros(2048))
        world.model.requires_grad_(False)
        gradients = []
        def trace(module, inputs, output):
            if module.weight.requires_grad and output.requires_grad:
                output.register_hook(lambda g: gradients.append(g.norm().item()))
        handle = policy.actor.register_forward_hook(trace)
        image, ram = torch.rand(8,3,40,60)*255, torch.zeros(8,2*N_FEATURES)
        world.improve(policy, main_opt, image, ram, torch.full((8,), HORIZON+1))
        handle.remove()
        self.assertEqual(len(gradients), 3)
        self.assertGreater(min(gradients), 0.)
        self.assertIsNone(policy.critic.weight.grad)
        self.assertEqual(len(main_opt.state), 0)
        self.assertGreater(len(world.imagined_optimizer.state), 0)
        clone = WorldLearner(5, torch.device("cpu"), "full-bptt")
        clone.restore(world.checkpoint())
        clone.model.requires_grad_(False)
        clone.improve(policy, main_opt, image, ram, torch.full((8,), HORIZON+1))
        self.assertTrue(all(item["step"].item() == 2 for item in clone.imagined_optimizer.state.values()))


if __name__ == "__main__":
    unittest.main()
