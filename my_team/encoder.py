"""Shared state encoder for the value net (~115 compact features).

Used IDENTICALLY by the self-play data generator and the deployed agent:
the net is trained on whatever this function produces, so the two must never
drift. Perspective: always "me first" — features of the querying player,
then the opponent.

Feature groups: effective/total production per resource (robber-aware and
not), reachable production at 0/1 roads, exact hands (the opponent's is
derivable), build distances, VPs, played devs, army/road state, buildable
nodes, pieces left, port ownership, bank, dev deck, robber-on-best-tile
flags, turn number.
"""
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
from catanatron.models.map import number_probability
from catanatron.features import (
    build_production_features,
    reachability_features,
)

_EFF = build_production_features(True)    # robber-aware
_TOT = build_production_features(False)   # raw
_DEV_TYPES = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")


def _distances(hand):
    wood, brick, sheep, wheat, ore = hand
    d_settle = (max(1 - wood, 0) + max(1 - brick, 0) + max(1 - sheep, 0) + max(1 - wheat, 0)) / 4
    d_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5
    d_dev = (max(1 - sheep, 0) + max(1 - wheat, 0) + max(1 - ore, 0)) / 3
    return d_settle, d_city, d_dev


def encode(game, me):
    st = game.state
    ps = st.player_state
    enemy = st.colors[1 - st.color_to_index[me]]

    eff = _EFF(game, me)
    tot = _TOT(game, me)
    reach = reachability_features(game, me, 2)

    feats = []
    for r in RESOURCES:
        feats.append(eff[f"EFFECTIVE_P0_{r}_PRODUCTION"])
    for r in RESOURCES:
        feats.append(eff[f"EFFECTIVE_P1_{r}_PRODUCTION"])
    for r in RESOURCES:
        feats.append(tot[f"TOTAL_P0_{r}_PRODUCTION"])
    for r in RESOURCES:
        feats.append(tot[f"TOTAL_P1_{r}_PRODUCTION"])
    for level in (0, 1):
        for r in RESOURCES:
            feats.append(reach[f"P0_{level}_ROAD_REACHABLE_{r}"])
    for level in (0, 1):
        for r in RESOURCES:
            feats.append(reach[f"P1_{level}_ROAD_REACHABLE_{r}"])

    for color in (me, enemy):
        key = f"P{st.color_to_index[color]}"
        hand = [ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES]
        feats.extend(h / 10.0 for h in hand)
        feats.append(sum(hand) / 10.0)
        d_settle, d_city, d_dev = _distances(hand)
        feats.extend((d_settle, d_city, d_dev))
        feats.append(ps[f"{key}_VICTORY_POINTS"] / 10.0)
        for d in _DEV_TYPES:
            feats.append(ps.get(f"{key}_PLAYED_{d}", 0) / 5.0)
        feats.append(ps[f"{key}_LONGEST_ROAD_LENGTH"] / 15.0)
        feats.append(1.0 if ps[f"{key}_HAS_ARMY"] else 0.0)
        feats.append(1.0 if ps[f"{key}_HAS_ROAD"] else 0.0)
        feats.append(ps[f"{key}_ROADS_AVAILABLE"] / 15.0)
        feats.append(ps[f"{key}_SETTLEMENTS_AVAILABLE"] / 5.0)
        feats.append(ps[f"{key}_CITIES_AVAILABLE"] / 4.0)
        feats.append(1.0 if ps[f"{key}_HAS_ROLLED"] else 0.0)

    me_key = f"P{st.color_to_index[me]}"
    en_key = f"P{st.color_to_index[enemy]}"
    for d in _DEV_TYPES:
        feats.append(ps[f"{me_key}_{d}_IN_HAND"] / 5.0)
    feats.append(sum(ps[f"{en_key}_{d}_IN_HAND"] for d in _DEV_TYPES) / 8.0)
    feats.append(ps[f"{me_key}_ACTUAL_VICTORY_POINTS"] / 10.0)

    feats.append(len(st.board.buildable_node_ids(me)) / 20.0)
    feats.append(len(st.board.buildable_node_ids(enemy)) / 20.0)

    port_nodes = st.board.map.port_nodes
    for color in (me, enemy):
        owned = set(st.buildings_by_color[color][SETTLEMENT] + st.buildings_by_color[color][CITY])
        for resource in RESOURCES:
            feats.append(1.0 if (owned & port_nodes.get(resource, set())) else 0.0)
        feats.append(1.0 if (owned & port_nodes.get(None, set())) else 0.0)

    for k, r in enumerate(RESOURCES):
        feats.append(st.resource_freqdeck[k] / 19.0)
    feats.append(len(st.development_listdeck) / 25.0)

    robber_tid = st.board.map.tiles[st.board.robber_coordinate].id
    for color in (me, enemy):
        best_prob, best_tid = 0.0, None
        for building_type, mult in ((SETTLEMENT, 1), (CITY, 2)):
            for node in st.buildings_by_color[color][building_type]:
                for t in st.board.map.adjacent_tiles[node]:
                    if getattr(t, "number", None) is not None and t.resource is not None:
                        p = mult * number_probability(t.number)
                        if p > best_prob:
                            best_prob, best_tid = p, t.id
        feats.append(1.0 if best_tid == robber_tid else 0.0)

    feats.append(st.num_turns / 100.0)
    return feats


# ---- feature layout (114 raw columns) — keep in sync with encode() above ----
# 0-4 eff me, 5-9 eff opp, 10-14 tot me, 15-19 tot opp,
# 20-24 reach r0 me, 25-29 reach r1 me, 30-34 reach r0 opp, 35-39 reach r1 opp,
# per-player blocks of 22: me 40-61, opp 62-83
#   (hands x5, size, d_settle/d_city/d_dev, vps, played x5, LR, army, road,
#    roads_left, settle_left, cities_left, rolled)
# 84-88 my dev hand x5, 89 opp dev count, 90 my actual vps,
# 91-92 buildable me/opp, 93-104 ports (me 5+1, opp 5+1), 105-109 bank,
# 110 dev deck, 111-112 robber flags, 113 turn.
_RAW = 114


def augment(x):
    """Derived features appended to the raw 114. Shared by training fit and
    deployed agent — never let the two drift."""
    eff_me = x[0:5]; eff_opp = x[5:10]
    r0_me = sum(x[20:25]); r1_me = sum(x[25:30])
    r0_opp = sum(x[30:35]); r1_opp = sum(x[35:40])
    return [
        float(sum(v > 1e-4 for v in eff_me)),
        float(sum(v > 1e-4 for v in eff_opp)),
        sum(eff_me), sum(eff_opp),
        r0_me, r1_me, r0_opp, r1_opp,
        sum(eff_me) - sum(eff_opp),
        x[49] - x[71],                       # public VP differential
        x[45] - x[67],                       # hand size differential
        x[40] - x[62], x[41] - x[63], x[42] - x[64], x[43] - x[65], x[44] - x[66],
        x[90] - x[49],                       # my hidden VP cards
        x[89],                               # opp unplayed dev count
    ]


def features(game, me):
    x = encode(game, me)
    return x + augment(x)

