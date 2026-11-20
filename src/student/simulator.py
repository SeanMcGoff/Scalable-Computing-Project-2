"""
simulator.py — QIS-NET Project 2 runner (STUDENT FILE: complete the TODOs).

Drive the provided harness (qnetsim.py) to implement the three graded
entry points. Signatures are normative; the autograder imports them.

Only numpy + standard library + qnetsim may be used.
"""

import math
from collections import deque

import numpy as np
from policy import OneHopPolicy, TwoHopPolicy
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


def run_one_hop(params, sim_time_ms, seed) -> dict:
    """Reference scenario: a single elementary link, no swapping.
    Return {"rate": ..., "mean_fidelity": ...}.

    Every generated pair counts as delivered (recorded by OneHopPolicy).
    Analytic check: rate = attempt_rate * p_success, fidelity = f_link.
    """
    links = make_links(params, "one_hop")
    sim = Simulator(
        links=links,
        t2=math.inf,
        q_swap=1.0,
        f_swap=params["physics"]["f_swap"],
        purification_allowed=False,
        seed=seed,
    )
    delivered = sim.run(sim_time_ms=sim_time_ms, policy=OneHopPolicy())
    summary = summarise(delivered=delivered, sim_time_ms=sim_time_ms)

    result = {"rate": summary["rate"], "mean_fidelity": summary["mean_fidelity"]}
    return result


def run_two_hop(params, sim_time_ms, seed) -> dict:
    """Reference scenario: two links, one repeater, immediate swap,
    T2 = infinity, q_swap = 1 (params['reference']['two_hop']).
    Return {"rate": ..., "mean_fidelity": ...}.

    TwoHopPolicy swaps as soon as both links hold a pair.
    Analytic check: rate = min over links of attempt_rate * p_success,
    fidelity = f_link^2 * f_swap.
    """
    links = make_links(params, "two_hop")
    sim = Simulator(
        links=links,
        t2=math.inf,
        q_swap=1.0,
        f_swap=params["physics"]["f_swap"],
        purification_allowed=False,
        seed=seed,
    )
    delivered = sim.run(sim_time_ms=sim_time_ms, policy=TwoHopPolicy())
    summary = summarise(delivered=delivered, sim_time_ms=sim_time_ms)

    result = {"rate": summary["rate"], "mean_fidelity": summary["mean_fidelity"]}
    return result


def run_variant_scenario(params, sim_time_ms, seed, policy) -> dict:
    """Full variant scenario with YOUR policy.
    Return {"rate": ..., "mean_fidelity": ..., "p95_latency_ms": ...,
            "pairs_consumed_per_delivery": ...}.
    """
    physics = params["physics"]
    links = make_links(params, "variant")  # topology between 1,2,3

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

    result = {
        "rate": n_delivered / (sim_time_ms / 1000),
        "mean_fidelity": summary["mean_fidelity"],
        "p95_latency_ms": p95_latency,
        "pairs_consumed_per_delivery": simulator.state.stats["generated"]
        / max(1, n_delivered),
    }
    return result
