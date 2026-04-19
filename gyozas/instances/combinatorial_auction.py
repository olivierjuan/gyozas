from collections.abc import Callable
from logging import getLogger

import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model, quicksum

from gyozas.instances.instance_generator import InstanceGenerator, sanitize_rng

IntGenerator = Callable[[np.random.Generator], int]
IntOrIntGenerator = int | IntGenerator


def _arg_choice_without_replacement(n_samples, weights, rng) -> NDArray[np.int64]:
    wc = np.cumsum(weights)
    indices = []
    for _ in range(n_samples):
        u = rng.uniform(0, wc[-1])
        idx = np.searchsorted(wc, u, side="right")
        indices.append(idx)
    return np.array(indices)


def _choose_next_item(bundle_mask, interests, compats, rng) -> int:
    compats_masked = compats * bundle_mask
    compats_masked_mean = np.mean(compats_masked, axis=1)
    probs = (1 - bundle_mask) * interests * compats_masked_mean
    return _arg_choice_without_replacement(1, probs, rng)[0]


def _get_bundle_price(bundle, private_values, integers, additivity) -> float:
    bundle_sum = np.sum(private_values[bundle])
    bundle_power = len(bundle) ** (1.0 + additivity)
    price = bundle_sum + bundle_power
    if integers:
        price = np.floor(price)
    return price


def _get_bundle(
    compats, private_interests, private_values, n_items, integers, additivity, add_item_prob, rng
) -> tuple[list, float]:
    item = _arg_choice_without_replacement(1, private_interests, rng)[0]
    bundle_mask = np.zeros(n_items, dtype=int)
    bundle_mask[item] = 1
    while True:
        sampled_prob = rng.uniform(0.0, 1.0)
        if sampled_prob >= add_item_prob:
            break
        if np.sum(bundle_mask) == n_items:
            break
        item = _choose_next_item(bundle_mask, private_interests, compats, rng)
        bundle_mask[item] = 1
    bundle = np.nonzero(bundle_mask)[0].tolist()
    price = _get_bundle_price(bundle, private_values, integers, additivity)
    return bundle, price


def _get_substitute_bundles(
    bundle, compats, private_interests, private_values, n_items, integers, additivity, rng
) -> list:
    sub_bundles = []
    for item in bundle:
        sub_bundle_mask = np.zeros(n_items, dtype=int)
        sub_bundle_mask[item] = 1
        while True:
            if np.sum(sub_bundle_mask) >= len(bundle):
                break
            item2 = _choose_next_item(sub_bundle_mask, private_interests, compats, rng)
            sub_bundle_mask[item2] = 1
        sub_bundle = np.nonzero(sub_bundle_mask)[0].tolist()
        sub_price = _get_bundle_price(sub_bundle, private_values, integers, additivity)
        sub_bundles.append((sub_bundle, sub_price))
    return sub_bundles


def _add_bundles(
    bidder_bids,
    sub_bundles,
    values,
    bundle,
    price,
    bid_index,
    logger,
    budget_factor,
    resale_factor,
    max_n_sub_bids,
    n_bids,
) -> None:
    budget = budget_factor * price
    min_resale_value = resale_factor * np.sum(values[bundle])
    sub_bundles.sort(key=lambda x: x[1], reverse=True)
    for sub_bundle, sub_price in sub_bundles:
        if len(bidder_bids) >= max_n_sub_bids + 1 or bid_index + len(bidder_bids) >= n_bids:
            break
        if sub_price < 0:
            logger.warning("negatively priced substitutable bundle avoided")
            continue
        if sub_price > budget:
            logger.warning("over priced substitutable bundle avoided")
            continue
        if np.sum(values[sub_bundle]) < min_resale_value:
            logger.warning("substitutable bundle below min resale value avoided")
            continue
        if tuple(sub_bundle) in bidder_bids:
            logger.warning("duplicated substitutable bundle avoided")
            continue
        bidder_bids[tuple(sub_bundle)] = sub_price


def _add_dummy_item(n_dummy_items, bidder_bids, n_items) -> int:
    dummy_item = 0
    if len(bidder_bids) > 2:
        dummy_item = n_items + n_dummy_items[0]
        n_dummy_items[0] += 1
    return dummy_item


def _add_bids(bids, bidder_bids, bid_index, dummy_item) -> None:
    for b, p in bidder_bids.items():
        bund_copy = list(b)
        if dummy_item:
            bund_copy.append(dummy_item)
        bids[bid_index[0]] = (bund_copy, p)
        bid_index[0] += 1


def _get_bids(
    values,
    compats,
    max_value,
    n_items,
    n_bids,
    max_n_sub_bids,
    integers,
    value_deviation,
    additivity,
    add_item_prob,
    budget_factor,
    resale_factor,
    logger,
    rng,
) -> tuple[list, int]:
    n_dummy_items = [0]
    bid_index = [0]
    bids = [None] * n_bids
    while bid_index[0] < n_bids:
        private_interests = rng.uniform(0.0, 1.0, n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)
        bidder_bids = {}
        bundle, price = _get_bundle(
            compats, private_interests, private_values, n_items, integers, additivity, add_item_prob, rng
        )
        if price < 0:
            logger.warning("negatively priced bundle avoided")
            continue
        bidder_bids[tuple(bundle)] = price
        substitute_bundles = _get_substitute_bundles(
            bundle, compats, private_interests, private_values, n_items, integers, additivity, rng
        )
        _add_bundles(
            bidder_bids,
            substitute_bundles,
            values,
            bundle,
            price,
            bid_index[0],
            logger,
            budget_factor,
            resale_factor,
            max_n_sub_bids,
            n_bids,
        )
        dummy_item = _add_dummy_item(n_dummy_items, bidder_bids, n_items)
        _add_bids(bids, bidder_bids, bid_index, dummy_item)
    return bids, n_dummy_items[0]


class CombinatorialAuctionGenerator(InstanceGenerator):
    """Generator for random Combinatorial Auction winner determination problem instances."""

    def __init__(
        self,
        n_items: IntOrIntGenerator = 100,
        n_bids: IntOrIntGenerator = 500,
        min_value: IntOrIntGenerator = 1,
        max_value: IntOrIntGenerator = 100,
        max_n_sub_bids: IntOrIntGenerator = 5,
        integers=False,
        value_deviation=0.5,
        additivity=0.2,
        add_item_prob=0.65,
        budget_factor=1.5,
        resale_factor=0.5,
        warnings=False,
        rng=None,
    ) -> None:
        self.n_items = n_items
        self.n_bids = n_bids
        self.min_value = min_value
        self.max_value = max_value
        self.max_n_sub_bids = max_n_sub_bids
        self.integers = integers
        self.value_deviation = value_deviation
        self.additivity = additivity
        self.add_item_prob = add_item_prob
        self.budget_factor = budget_factor
        self.resale_factor = resale_factor
        self.warnings = warnings
        super().__init__(rng=rng)

    def __next__(self) -> Model:
        return self.generate_instance(
            n_items=self.n_items,
            n_bids=self.n_bids,
            min_value=self.min_value,
            max_value=self.max_value,
            max_n_sub_bids=self.max_n_sub_bids,
            integers=self.integers,
            value_deviation=self.value_deviation,
            additivity=self.additivity,
            add_item_prob=self.add_item_prob,
            budget_factor=self.budget_factor,
            resale_factor=self.resale_factor,
            warnings=self.warnings,
            rng=self.rng,
        )

    def generate_instance(
        self,
        n_items: IntOrIntGenerator = 10,
        n_bids: IntOrIntGenerator = 20,
        min_value: IntOrIntGenerator = 1,
        max_value: IntOrIntGenerator = 10,
        max_n_sub_bids: IntOrIntGenerator = 2,
        integers=True,
        value_deviation=0.1,
        additivity=0.0,
        add_item_prob=0.5,
        budget_factor=1.5,
        resale_factor=0.5,
        warnings=False,
        rng=None,
    ) -> Model:
        rng = sanitize_rng(rng, default=self.rng)
        if isinstance(n_items, Callable):
            n_items = n_items(rng)  # ty: ignore[call-top-callable]
        if isinstance(n_bids, Callable):
            n_bids = n_bids(rng)  # ty: ignore[call-top-callable]
        if isinstance(min_value, Callable):
            min_value = min_value(rng)  # ty: ignore[call-top-callable]
        if isinstance(max_value, Callable):
            max_value = max_value(rng)  # ty: ignore[call-top-callable]
        if isinstance(max_n_sub_bids, Callable):
            max_n_sub_bids = max_n_sub_bids(rng)  # ty: ignore[call-top-callable]
        if not (max_value >= min_value):
            raise ValueError("Parameters max_value and min_value must be defined such that: min_value <= max_value.")
        if not (0 <= add_item_prob <= 1):
            raise ValueError("Parameter add_item_prob must be in range [0,1].")
        logger = getLogger(__name__)
        if not warnings:
            logger.setLevel("ERROR")
        # Generate data
        rand_val = rng.uniform(0.0, 1.0, n_items)
        values = min_value + (max_value - min_value) * rand_val
        compats_rand = rng.uniform(0.0, 1.0, (n_items, n_items))
        compats = np.triu(compats_rand, 1)
        compats = compats + compats.T
        compats = compats / np.sum(compats, axis=1, keepdims=True)
        bids, n_dummy_items = _get_bids(
            values,
            compats,
            max_value,
            n_items,
            n_bids,
            max_n_sub_bids,
            integers,
            value_deviation,
            additivity,
            add_item_prob,
            budget_factor,
            resale_factor,
            logger,
            rng,
        )
        model = Model(f"CombinatorialAuction-{n_items}-{n_bids}")
        model.setMaximize()
        # Build bids_per_item mapping
        bids_per_item = [[] for _ in range(n_items + n_dummy_items)]
        for i, (bundle, _) in enumerate(bids):
            for item in bundle:
                bids_per_item[item].append(i)
        # Variables
        vars = []
        for i, (_, price) in enumerate(bids):
            v = model.addVar(vtype="BINARY", obj=price, name=f"x_{i}")
            vars.append(v)
        # Constraints
        for idx, item_bids in enumerate(bids_per_item):
            if item_bids:
                model.addCons(quicksum(vars[j] for j in item_bids) <= 1, name=f"c_{idx}")
        return model
