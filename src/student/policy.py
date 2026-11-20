"""
policy.py — QIS-NET Project 2 policy (STUDENT FILE: replace the baseline).

The policy decides, each time a fresh pair arrives on a link, what to do:
swap immediately, purify first, hold, or discard aged pairs. Your variant's
SLOs (rate, fidelity, p95 latency) will typically NOT be met by the
baseline policy below.

Design space (see spec section 3.2):
  * purify while the freshest link fidelity is below a threshold AND a
    second pair is available on the same link;
  * hold vs swap rules (e.g., wait for a purified partner);
  * discard pairs whose current fidelity has fallen below a floor.
"""

from abc import ABC, abstractmethod

from qnetsim import DeliveredPair


class Policy(ABC):
    @abstractmethod
    def on_pair_ready(self, link, pair, state, sim): ...


class BaselinePolicy(Policy):
    """Swap-as-soon-as-possible; never purify. PROVIDED as the baseline.
    It meets the reference scenarios' expectations but is deliberately
    insufficient for most variant SLOs."""

    def __init__(self, params):
        self.params = params
        # map each repeater node to its adjacent link pairs for swapping
        self.chains = params["policy"]["swap_chains"]  # list of [left, right]

    def on_pair_ready(self, link, pair, state, sim):
        for left, right in self.chains:
            if state.pairs.get(left) and state.pairs.get(right):
                sim.swap(left, right)


class OneHopPolicy(Policy):
    """Mark as delivered always, destroy pair on delivery"""

    def on_pair_ready(self, link, pair, state, sim):
        sim.delivered.append(
            DeliveredPair(
                request_id=-1, delivered_at=state.now, fidelity=pair.f0, consumed=1
            )
        )
        state.pairs[link.link_id].remove(pair)


class TwoHopPolicy(Policy):
    """Swap the oldest pair on each link as soon as both links hold one"""

    def on_pair_ready(self, link, pair, state, sim):
        while state.pairs.get(0) and state.pairs.get(1):
            sim.swap(0, 1)


class VariantPolicy(BaselinePolicy):
    """Prioritises the freshest pair first"""

    def on_pair_ready(self, link, pair, state, sim):
        for left, right in self.chains:
            # hold/swap condition
            if state.pairs.get(left) and state.pairs.get(right):
                sim.swap(left, right, left_index=-1, right_index=-1)
        link1 = state.pairs.get(1, [])

        # discard low-fidelity pairs
        while link1 and state.current_fidelity(link1[0]) < 0.5:
            sim.discard(1, 0)


def make_policy(params) -> Policy:
    """Returns the policy for variant simulator"""
    return VariantPolicy(params)
