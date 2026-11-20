"""
simulator.py — QIS-NET Project 2 runner (STUDENT FILE: complete the TODOs).

Drive the provided harness (qnetsim.py) to implement the three graded
entry points. Signatures are normative; the autograder imports them.

Only numpy + standard library + qnetsim may be used.
"""

import math
from collections import deque

import numpy as np
from policy import OneHopPolicy, Policy, TwoHopPolicy
from qnetsim import Link, Simulator, summarise

# ---------------------------------------------------------------- helpers


def make_links(params, which="variant") -> list[Link]:
    """Build Link objects from the variant parameter dict.

    `which="one_hop"` / `"two_hop"` select the reference scenarios defined in
    params["reference"]; `which="variant"` builds the full variant topology.
    PROVIDED — you should not need to change this.
    """
    links = []
    if which == "one_hop":
        ref = params["reference"]["one_hop"]
        links.append(
            Link(
                link_id=0,
                nodes=(0, 1),
                attempt_rate=ref["attempt_rate"],
                p_success=ref["p_success"],
                f_link=params["physics"]["f_link"],
            )
        )
    elif which == "two_hop":
        ref = params["reference"]["two_hop"]
        for i, lp in enumerate(ref["links"]):
            links.append(
                Link(
                    link_id=i,
                    nodes=(i, i + 1),
                    attempt_rate=lp["attempt_rate"],
                    p_success=lp["p_success"],
                    f_link=params["physics"]["f_link"],
                )
            )
    else:
        for i, lp in enumerate(params["topology"]["links"]):
            links.append(
                Link(
                    link_id=i,
                    nodes=tuple(lp["nodes"]),
                    attempt_rate=lp["attempt_rate"],
                    p_success=lp["p_success"],
                    f_link=params["physics"]["f_link"],
                )
            )
    return links


def request_times(params, sim_time_ms, seed) -> list[float]:
    """Deterministic request arrival times (ms) for the variant workload.
    Uses a generator derived from `seed` so runs are reproducible.
    PROVIDED."""
    wl = params["workload"]
    rng = np.random.default_rng(seed + 7919)
    t, times = 0.0, []
    while True:
        t += rng.exponential(1000.0 / wl["arrival_rate_per_s"])
        if t > sim_time_ms:
            break
        times.append(t)
    return times


# ---------------------------------------------------------------- graded entry points


def _reference_scenario(
    params: dict, which: str, policy: Policy, sim_time_ms: int, seed: int
) -> dict:
    """Abstraction of two reference scenarios"""

    links = make_links(params, which)
    sim = Simulator(
        links=links,
        t2=math.inf,
        q_swap=1.0,
        f_swap=params["physics"]["f_swap"],
        purification_allowed=False,
        seed=seed,
    )
    delivered = sim.run(sim_time_ms, policy)
    summary = summarise(delivered, sim_time_ms)

    return {"rate": summary["rate"], "mean_fidelity": summary["mean_fidelity"]}


def run_one_hop(params, sim_time_ms, seed) -> dict:
    """Reference scenario: a single elementary link, no swapping."""

    return _reference_scenario(params, "one_hop", OneHopPolicy(), sim_time_ms, seed)


def run_two_hop(params, sim_time_ms, seed) -> dict:
    """Reference scenario: two links, one repeater, immediate swap."""

    return _reference_scenario(params, "two_hop", TwoHopPolicy(), sim_time_ms, seed)


def run_variant_scenario(params, sim_time_ms, seed, policy) -> dict:
    """Full variant scenario with VariantPolicy"""

    physics = params["physics"]
    links = make_links(params, "variant")  # topology between 0,1,2

    simulator = Simulator(
        links=links,
        t2=physics["t2_ms"],
        q_swap=physics["q_swap"],
        f_swap=physics["f_swap"],
        purification_allowed=physics["purification_allowed"],
        seed=seed,
    )

    delivered = simulator.run(sim_time_ms=sim_time_ms, policy=policy)
    delivery_timings = [
        i.delivered_at for i in delivered
    ]  # should be already sorted bc of append immediately after delivery
    request_timings = request_times(params, sim_time_ms, seed)

    # FIFO matching: buffered pairs serve requests instantly, else requests queue
    buffer = 0
    waiting = deque()
    delays = []

    delivery_idx, request_idx = 0, 0

    # merge both sorted timelines, handling the earlier event each step
    while delivery_idx < len(delivery_timings) or request_idx < len(request_timings):
        next_delivery = (
            delivery_timings[delivery_idx]
            if delivery_idx < len(delivery_timings)
            else math.inf
        )
        next_request = (
            request_timings[request_idx]
            if request_idx < len(request_timings)
            else math.inf
        )

        if next_delivery <= next_request:
            # pair delivered: serve oldest waiting request, else buffer it
            if waiting:
                customer_time = waiting.popleft()
                delays.append(next_delivery - customer_time)
            else:
                buffer += 1
            delivery_idx += 1
        else:
            # request arrives: take a buffered pair, else join the queue
            if buffer > 0:
                buffer -= 1
                delays.append(0.0)
            else:
                waiting.append(next_request)
            request_idx += 1

    # p95 over served requests only (unserved ones never got a latency)
    if delays:
        delays_sorted = sorted(delays)
        p95_index = math.ceil(0.95 * len(delays_sorted)) - 1
        p95_latency = delays_sorted[p95_index]
    else:
        p95_latency = math.inf

    n_delivered = len(delivered)
    summary = summarise(delivered=delivered, sim_time_ms=sim_time_ms)

    return {
        "rate": n_delivered / (sim_time_ms / 1000),
        "mean_fidelity": summary["mean_fidelity"],
        "p95_latency_ms": p95_latency,
        "pairs_consumed_per_delivery": simulator.state.stats["generated"]
        / max(1, n_delivered),
    }
