"""Mehregan Fig 3 per-pulse actor–critic shapes and pulse construction."""
from __future__ import annotations

import numpy as np
import torch

from controllers.ddpg.pulse_placement import (
    PulseActor,
    PulseCritic,
    pulse_idbs,
    regular_slots,
)
from envs.plant.dbs import DBS_AMPLITUDE_NA_PER_CM2


def test_actor_logits_and_slots_follow_fig3a():
    actor = PulseActor(state_len=40, n_pulses=3, n_slots=40)
    logits = actor(torch.zeros(2, 40))
    assert logits.shape == (2, 3 * 40)  # f·l·l/1000 with f·l/1000 = 3, l = 40
    slots = PulseActor.slots_from_logits(logits, 3, 40)
    assert slots.shape == (2, 3)
    assert int(slots.min()) >= 0 and int(slots.max()) < 40


def test_init_regular_biases_each_pulse_to_its_regular_slot():
    actor = PulseActor(state_len=2000, n_pulses=90, n_slots=2000)
    actor.init_regular(mean_hz=45.0, bias_scale=50.0)
    with torch.no_grad():
        slots = PulseActor.slots_from_logits(actor(torch.zeros(1, 2000)), 90, 2000)[0]
    expected = regular_slots(mean_hz=45.0, n_pulses=90, n_slots=2000)
    assert np.array_equal(slots.numpy(), expected)


def test_critic_scores_state_and_full_logit_vector():
    critic = PulseCritic(state_len=40, n_logits=120)
    assert critic(torch.zeros(4, 40), torch.zeros(4, 120)).shape == (4,)


def test_pulse_idbs_places_one_pulse_per_distinct_slot():
    idbs = pulse_idbs(np.array([0, 10, 10, 1999]), step_s=2.0, dt_ms=0.02)
    onsets = np.flatnonzero(np.diff(np.concatenate([[0.0], idbs])) > 0)
    assert onsets.tolist() == [0, 500, 99950]  # slot ms / 0.02 ms; duplicate slot coincides
    assert idbs.max() == DBS_AMPLITUDE_NA_PER_CM2
    assert idbs.size == 100001
