"""v1a: F's proven skeleton + three safe evaluation fixes.

Reverts v1's two harmful deltas (own-actual-VP scoring inflated dev buying;
flat longest-road weight caused road obsession). Keeps:

  1. reach-0 production weight 8e3 (F weights it 0)
  2. knights in hand count toward the Largest Army race (5.0 each)
  3. port ownership valued (specialty ports weighted by matching production)

Dominant VP term uses PUBLIC victory points exactly like F, and the
longest-road weight stays conditional like F.
"""
import random

from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
from catanatron.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)

TRANSLATE_VARIETY = 4

WEIGHTS = {
    "public_vps": 3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    "reachable_production_0": 8e3,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10.0,
    "knight_in_hand": 5.0,
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
    "port_specialty": 2.5e3,
    "port_generic": 1.2e3,
}

_PRODUCTION = build_production_features(True)


def value_production(sample, player_name, include_variety=True):
    proba_point = 2.778 / 100
    features = [f"EFFECTIVE_{player_name}_{r}_PRODUCTION" for r in RESOURCES]
    prod_sum = sum(sample[f] for f in features)
    prod_variety = sum(sample[f] != 0 for f in features) * TRANSLATE_VARIETY * proba_point
    return prod_sum + (prod_variety if include_variety else 0)


def evaluate(game, p0_color):
    w = WEIGHTS
    st = game.state
    key = player_key(st, p0_color)

    production_sample = _PRODUCTION(game, p0_color)
    production = value_production(production_sample, "P0")
    enemy_production = value_production(production_sample, "P1", False)

    reachability_sample = reachability_features(game, p0_color, 2)
    r0 = sum(reachability_sample[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
    r1 = sum(reachability_sample[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)

    hand_sample = resource_hand_features(game, p0_color)
    distance_to_city = (
        max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
        + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
    ) / 5.0
    distance_to_settlement = (
        max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
        + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
        + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
        + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
    ) / 4.0
    hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

    num_in_hand = player_num_resource_cards(st, p0_color)
    discard_penalty = w["discard_penalty"] if num_in_hand > 7 else 0

    buildings = st.buildings_by_color[p0_color]
    owned_nodes = set(buildings[SETTLEMENT] + buildings[CITY])

    owned_tiles = set()
    for n in owned_nodes:
        owned_tiles.update(st.board.map.adjacent_tiles[n])
    num_tiles = len(owned_tiles)

    num_buildable = len(st.board.buildable_node_ids(p0_color))
    longest_road_factor = w["longest_road"] if num_buildable == 0 else 0.1

    port_bonus = 0.0
    for resource, nodes in st.board.map.port_nodes.items():
        if not (owned_nodes & nodes):
            continue
        if resource is None:
            port_bonus += w["port_generic"]
        else:
            prod = production_sample[f"EFFECTIVE_P0_{resource}_PRODUCTION"]
            port_bonus += w["port_specialty"] * (1.0 if prod > 0.11 else 0.4)

    return float(
        st.player_state[f"{key}_VICTORY_POINTS"] * w["public_vps"]
        + production * w["production"]
        + enemy_production * w["enemy_production"]
        + r0 * w["reachable_production_0"]
        + r1 * w["reachable_production_1"]
        + hand_synergy * w["hand_synergy"]
        + num_buildable * w["buildable_nodes"]
        + num_tiles * w["num_tiles"]
        + num_in_hand * w["hand_resources"]
        + discard_penalty
        + get_longest_road_length(st, p0_color) * longest_road_factor
        + player_num_dev_cards(st, p0_color) * w["hand_devs"]
        + get_played_dev_cards(st, p0_color, "KNIGHT") * w["army_size"]
        + st.player_state[f"{key}_KNIGHT_IN_HAND"] * w["knight_in_hand"]
        + port_bonus
    )


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = evaluate(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
