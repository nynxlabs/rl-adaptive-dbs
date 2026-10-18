"""CTX-BG-TH network integrator (Kumaravelu et al., 2016).

The integrator is a translation of the Kumaravelu et al. (2016) MATLAB model and
lives in the private ``kumaravelu_plant`` package (``kumaravelu_plant.network``)
while the model's license is pending; see LICENSING.md. These names resolve to
it on first use. ``matlab_rng`` here is ours.
"""

from envs.plant._kumaravelu import deferred

NetworkInitDraws = deferred("network.integrator", "NetworkInitDraws")
NetworkState = deferred("network.integrator", "NetworkState")
integrate_network = deferred("network.integrator", "integrate_network")
initialize_network_state = deferred("network.integrator", "initialize_network_state")

__all__ = [
    "NetworkInitDraws",
    "NetworkState",
    "integrate_network",
    "initialize_network_state",
]
