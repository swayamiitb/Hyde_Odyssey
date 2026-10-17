"""Resource-aware afterstate search. Uses only the public observation API.

Public baseline: bcollazo/catanatron (pinned by the kit). No learned private
seeds, opponent state, external files, network or runtime workers are used.
"""
import math
import random
import time
from collections import deque

from catanatron.models.enums import ActionType as AT, RESOURCES, SETTLEMENT, CITY
from catanatron.models.board import STATIC_GRAPH

RW = (0.85, 0.8, 0.9, 1.45, 1.55)
PROD_SCALE = 1.15
HAND_SCALE = 0.065
DEV_VALUE = 0.45
REACH_SCALE = 0.45
POTENTIAL_SCALE = 0.35
HOLD_KNIGHT = True
FREE_ROADS = False
VP_CURVE = 0.0
DISCARD_STEP = 0.0
DUEL_WEIGHT = 0.6
SEARCH_NODES = 160
SEARCH_WIDTH = 6
SEARCH_DEPTH = 5
NEIGHBORS = {n: tuple(v for v in STATIC_GRAPH.neighbors(n) if v < 54) for n in range(54)}
COSTS = ((0, 0, 0, 2, 3), (1, 1, 1, 1, 0), (0, 0, 1, 1, 1))
DEV_NAMES = ('KNIGHT', 'VICTORY_POINT', 'ROAD_BUILDING', 'YEAR_OF_PLENTY', 'MONOPOLY')


class Agent:
    def __init__(self):
        self.rng = random.Random(1307)

    def prepare(self, g, me):
        self.me = me
        self.k = 'P%d_' % g.state.color_to_index[me]
        self.opp = next(c for c in g.state.colors if c != me)
        self.ek = 'P%d_' % g.state.color_to_index[self.opp]
        self.node_prod = {}
        for node, tiles in g.state.board.map.adjacent_tiles.items():
            p = [0.0] * 5
            for tile in tiles:
                if tile.resource:
                    p[RESOURCES.index(tile.resource)] += (6 - abs(7 - tile.number)) / 36
            self.node_prod[node] = p
        ps = g.state.player_state
        vp = max(ps[self.k+'ACTUAL_VICTORY_POINTS'], ps[self.ek+'VICTORY_POINTS'] + 0.15)
        self.econ = max(0.32, min(1.0, (12 - vp) / 8))
        self.board_cache = {}
        self.nodes = 0

    def board_terms(self, g):
        st = g.state
        b = st.board
        signature = (tuple(sorted(b.buildings.items())), tuple(sorted(b.roads.items())), b.robber_coordinate, st.is_road_building, st.free_roads_available)
        cached = self.board_cache.get(signature)
        if cached is not None:
            return cached
        p = [0.0]*5
        ep = [0.0]*5
        own_nodes = []
        for node, (color, kind) in b.buildings.items():
            dst = p if color == self.me else ep
            mult = 2 if kind == CITY else 1
            for i, amount in enumerate(self.node_prod[node]):
                dst[i] += mult*amount
            if color == self.me:
                own_nodes.append(node)
        # Robber is transient; permanently losing all variety is not a real effect.
        blocked = [0.0]*5
        enemy_blocked = [0.0]*5
        tile = b.map.tiles[b.robber_coordinate]
        if tile.resource:
            ri = RESOURCES.index(tile.resource)
            prob = (6-abs(7-tile.number))/36
            for node in tile.nodes.values():
                if node in b.buildings:
                    c, kind = b.buildings[node]
                    (blocked if c == self.me else enemy_blocked)[ri] += prob*(2 if kind == CITY else 1)
        rates = [4]*5
        owned = set(own_nodes)
        for resource, nodes in b.map.port_nodes.items():
            if owned.intersection(nodes):
                if resource is None:
                    rates = [min(3, r) for r in rates]
                else:
                    rates[RESOURCES.index(resource)] = 2
        utility = sum(RW[i] * math.log1p(p[i]/0.15) for i in range(5))
        utility += sum(p[i]*(4/rates[i]-1)*1.8 for i in range(5))
        value = PROD_SCALE*self.econ*utility
        value -= 2.8*sum(RW[i]*blocked[i] for i in range(5))
        value += 2.4*sum(RW[i]*enemy_blocked[i] for i in range(5))
        # Shortest useful expansion distance, stopping at enemy buildings.
        dist = {}
        q = deque()
        for component in b.connected_components[self.me]:
            for node in component:
                if node not in b.buildings or b.buildings[node][0] == self.me:
                    dist[node] = 0
                    q.append(node)
        while q:
            node = q.popleft()
            d = dist[node]
            if d >= 3:
                continue
            for nxt in NEIGHBORS[node]:
                road = b.roads.get((node,nxt))
                if road is not None and road != self.me:
                    continue
                if nxt in b.buildings and b.buildings[nxt][0] != self.me:
                    continue
                nd = d + (road != self.me)
                if nd < dist.get(nxt,99):
                    dist[nxt] = nd
                    q.append(nxt)
        potential = []
        nearest = 99
        for node in b.board_buildable_ids:
            d = dist.get(node,99)
            if d <= 3:
                gain = sum(RW[i]*math.log1p(self.node_prod[node][i]/(0.15+p[i])) for i in range(5))
                potential.append((1+PROD_SCALE*self.econ*gain)/(1+1.5*d))
                nearest = min(nearest,d)
        potential.sort(reverse=True)
        value += REACH_SCALE*(sum(potential[:1]) + .15*sum(potential[1:3]))
        lr = b.road_lengths[self.me]
        enemy_lr = b.road_lengths[self.opp]
        if b.road_color != self.me:
            need = max(5,enemy_lr+1)
            if lr < need:
                value += .65 * (lr/need)**3
        if st.is_road_building:
            value += .13*st.free_roads_available
        cached = (value,p,rates,nearest)
        self.board_cache[signature] = cached
        return cached

    def evaluate(self,g):
        st = g.state
        ps = st.player_state
        k, ek = self.k,self.ek
        vp = ps[k+'ACTUAL_VICTORY_POINTS']
        if vp >= 10:
            return 100000.0
        value,p,rates,nearest = self.board_terms(g)
        value += vp + VP_CURVE*(vp-2)**2 - .65*ps[ek+'VICTORY_POINTS']
        h = [ps[k+r+'_IN_HAND'] for r in RESOURCES]
        # Reserve useful amounts; a fourth/fifth copy is chiefly trade currency.
        caps = (2,2,2,3,4)
        marginal = [(0.65 + .10/(p[i]+.08))*RW[i] for i in range(5)]
        value += HAND_SCALE*sum(marginal[i]*(min(h[i],caps[i]) + max(0,h[i]-caps[i])/rates[i]) for i in range(5))
        value -= .035*max(0,sum(h)-7) + DISCARD_STEP*(sum(h)>7)
        city_possible = bool(st.buildings_by_color[self.me][SETTLEMENT]) and ps[k+'CITIES_AVAILABLE'] > 0
        sett_possible = nearest <= 1 and ps[k+'SETTLEMENTS_AVAILABLE'] > 0
        potentials = []
        for j,cost in enumerate(COSTS):
            if j == 0 and not city_possible or j == 1 and not sett_possible:
                continue
            deficit = sum(max(0,cost[i]-h[i]) for i in range(5))
            spare = sum(max(0,h[i]-cost[i])/rates[i] for i in range(5))
            effective = max(0,deficit-spare)
            worth = (1.1 if j==0 else 0.9 if j==1 else .42)
            potentials.append(worth*math.exp(-.8*effective))
        value += POTENTIAL_SCALE*max(potentials,default=0)
        played = ps[k+'PLAYED_KNIGHT']
        enemy_played = ps[ek+'PLAYED_KNIGHT']
        held = ps[k+'KNIGHT_IN_HAND']
        threshold = max(3,enemy_played+1)
        if ps[k+'HAS_ARMY']:
            value += .06*min(held,2)
        else:
            value += 1.6*min(1,(played+.65*min(held, max(0,threshold-played)))/threshold)**2
        secure = ps[k+'HAS_ARMY'] and played > enemy_played + 1
        value += (0.1 if secure else DEV_VALUE)*min(held,2) + (0.0 if secure else .08)*max(0,held-2)
        value += 0.15*ps[k+'ROAD_BUILDING_IN_HAND']
        value += 0.22*ps[k+'YEAR_OF_PLENTY_IN_HAND']
        value += 0.3*ps[k+'MONOPOLY_IN_HAND']
        return value

    def successor(self,g,a):
        s = g.copy()
        s.execute(a)
        self.nodes += 1
        return s

    def score_action(self,g,a):
        if a.action_type in (AT.END_TURN,AT.ROLL):
            return self.evaluate(g),None
        if a.action_type == AT.BUY_DEVELOPMENT_CARD:
            # Average over the PUBLIC unseen multiset, never a lucky sampled card.
            ps = g.state.player_state
            unseen = {d:g.state.development_listdeck.count(d)+ps[self.ek+d+'_IN_HAND'] for d in DEV_NAMES}
            n = sum(unseen.values())
            s = self.successor(g,a)
            drawn = s.state.action_records[-1].result
            sk = s.state.player_state
            sk[self.k+drawn+'_IN_HAND'] -= 1
            if drawn == 'VICTORY_POINT':
                sk[self.k+'ACTUAL_VICTORY_POINTS'] -= 1
            score = 0.0
            for d,c in unseen.items():
                if c:
                    sk[self.k+d+'_IN_HAND'] += 1
                    if d == 'VICTORY_POINT':
                        sk[self.k+'ACTUAL_VICTORY_POINTS'] += 1
                    score += c*self.evaluate(s)/n
                    sk[self.k+d+'_IN_HAND'] -= 1
                    if d == 'VICTORY_POINT':
                        sk[self.k+'ACTUAL_VICTORY_POINTS'] -= 1
            return score,None
        s = self.successor(g,a)
        return self.evaluate(s),s

    def opening_value(self, nodes, available, second=None):
        p = [sum(self.node_prod[n][i] for n in nodes) for i in range(5)]
        value = sum(RW[i]*math.log1p(p[i]/.15) for i in range(5))
        ports = self.opening_map.port_nodes
        owned = set(nodes)
        for r, portnodes in ports.items():
            if owned.intersection(portnodes):
                value += .18 if r is None else .5*p[RESOURCES.index(r)]
        # A usable nearby expansion remains useful even for a city/dev economy.
        near = set()
        for n in nodes:
            for a in NEIGHBORS[n]:
                near.update(NEIGHBORS[a])
        options = []
        for n in near.intersection(available):
            options.append(sum(RW[i]*self.node_prod[n][i]/(.15+p[i]) for i in range(5)))
        value += .12*max(options,default=0)
        if second is not None:
            # Starting hand liquidity is useful, but production determines the economy.
            resources = [t.resource for t in self.opening_map.adjacent_tiles[second] if t.resource]
            value += .025*sum(RW[RESOURCES.index(r)] for r in resources)
        return value

    def initial_placement(self,g,actions):
        self.opening_map=g.state.board.map
        own=list(g.state.buildings_by_color[self.me][SETTLEMENT])
        enemy=list(g.state.buildings_by_color[self.opp][SETTLEMENT])
        available=set(g.state.board.board_buildable_ids)
        def remove(places,node):
            return places.difference((node,*NEIGHBORS[node]))
        def quality(prod):
            return sum(w*math.log1p(prod[i]/.15) for i,w in enumerate((1.,1.,.8,1.1,1.2)))
        def greedy(prod,n):
            return sum(prod[j]+self.node_prod[n][j]+.11112*(prod[j]+self.node_prod[n][j]>0) for j in range(5))
        enemy_p=[sum(self.node_prod[n][j] for n in enemy) for j in range(5)]
        ranked=sorted(available,key=lambda n:(-greedy(enemy_p,n),n))
        enemy_quality={n:quality([enemy_p[j]+self.node_prod[n][j] for j in range(5)]) for n in available}
        def response(free):
            return enemy_quality[next(n for n in ranked if n in free)]
        if own:
            best=(-1e99,0)
            for i,a in enumerate(actions):
                free=remove(available,a.value)
                v=self.opening_value(own+[a.value],free,a.value)
                if len(enemy)==1:
                    v-=DUEL_WEIGHT*response(free)
                if v>best[0]:best=(v,i)
            return best[1]
        best=(-1e99,0)
        for i,a in enumerate(actions):
            n=a.value
            free=remove(available,n)
            if not enemy:
                ep=[0.]*5
                for _ in range(2):
                    pick=max(sorted(free),key=lambda q:greedy(ep,q))
                    ep=[ep[j]+self.node_prod[pick][j] for j in range(5)]
                    free=remove(free,pick)
                eq=quality(ep)
            values=[]
            for q in free:
                remainder=remove(free,q)
                v=self.opening_value([n,q],remainder,q)
                v-=DUEL_WEIGHT*(eq if not enemy else response(remainder))
                values.append(v)
            values.sort(reverse=True)
            value=.9*values[0]+.1*sum(values[:3])/min(3,len(values))
            if value>best[0]:best=(value,i)
        return best[1]


    def trade_plans(self,g,acts,best_v,best_i,deadline):
        root_trades = {a:i for i,a in enumerate(acts) if a.action_type == AT.MARITIME_TRADE}
        if not root_trades:
            return best_v,best_i
        _,p,rates,_ = self.board_terms(g)
        start_h = tuple(g.state.player_state[self.k+r+'_IN_HAND'] for r in RESOURCES)
        for target,cost in ((AT.BUILD_CITY,COSTS[0]),(AT.BUILD_SETTLEMENT,COSTS[1]),(AT.BUY_DEVELOPMENT_CARD,COSTS[2]),(AT.BUILD_ROAD,(1,1,0,0,0))):
            if all(start_h[i]>=cost[i] for i in range(5)):
                continue
            if sum(max(0,cost[i]-start_h[i]) for i in range(5))>sum(max(0,start_h[i]-cost[i])//rates[i] for i in range(5)):
                continue
            seen = set()
            agenda = [(g,None,0)]
            while agenda:
                if time.process_time()>deadline or self.nodes>=SEARCH_NODES:
                    return best_v,best_i
                s,first,depth = agenda.pop()
                hand = tuple(s.state.player_state[self.k+r+'_IN_HAND'] for r in RESOURCES)
                if hand in seen:
                    continue
                seen.add(hand)
                missing = [i for i in range(5) if hand[i]<cost[i]]
                if not missing:
                    for a in s.playable_actions:
                        if a.action_type == target:
                            v,_ = self.score_action(s,a)
                            v -= .001*depth
                            if v>best_v:
                                best_v,best_i = v,first
                    continue
                receive = RESOURCES[missing[0]]
                for a in s.playable_actions:
                    if a.action_type == AT.MARITIME_TRADE and a.value[-1]==receive:
                        donor = RESOURCES.index(a.value[0])
                        if hand[donor]-cost[donor]>=rates[donor]:
                            nxt = self.successor(s,a)
                            agenda.append((nxt,root_trades[a] if first is None else first,depth+1))
        return best_v,best_i


    def decide(self,obs,playable_actions):
        if len(playable_actions)==1:
            return 0
        if obs.cpu_left is not None and obs.cpu_left < .04:
            for kind in (AT.BUILD_CITY,AT.BUILD_SETTLEMENT,AT.ROLL,AT.PLAY_KNIGHT_CARD,AT.BUY_DEVELOPMENT_CARD,AT.END_TURN):
                for i,a in enumerate(playable_actions):
                    if a.action_type == kind:
                        return i
            return 0
        g = obs.sample_world(self.rng)
        self.prepare(g,obs.me)
        acts = playable_actions
        if obs.my_actual_vps >= 9 and not g.state.is_initial_build_phase:
            for i,a in enumerate(acts):
                if a.action_type in (AT.BUILD_CITY, AT.BUILD_SETTLEMENT):
                    return i
            value,index=self.trade_plans(g,acts,-1e99,0,time.process_time()+.012)
            if value > 99900:
                return int(index)
        if g.state.is_initial_build_phase and acts[0].action_type == AT.BUILD_SETTLEMENT:
            return int(self.initial_placement(g,acts))
        # Roll before ordinary development-card choices; free a blocked economy first.
        rolls = [i for i,a in enumerate(acts) if a.action_type == AT.ROLL]
        knights = [i for i,a in enumerate(acts) if a.action_type == AT.PLAY_KNIGHT_CARD]
        if rolls:
            if knights:
                t = g.state.board.map.tiles[g.state.board.robber_coordinate]
                if any(g.state.board.buildings.get(n,(None,))[0] == obs.me for n in t.nodes.values()):
                    return knights[0]
            return rolls[0]
        hold = HOLD_KNIGHT and obs.my_devs['KNIGHT']==1 and g.state.player_state[self.k+'HAS_ARMY'] and obs.played_devs['me']['KNIGHT'] > obs.played_devs['opp']['KNIGHT']+1
        if knights and not hold:
            monopolies = [i for i,a in enumerate(acts) if a.action_type==AT.PLAY_MONOPOLY and obs.opp_resources[a.value]>=4]
            if not monopolies:
                return knights[0]
        if FREE_ROADS and max(obs.opp_resources.values()) < 4:
            free = [i for i,a in enumerate(acts) if a.action_type == AT.PLAY_ROAD_BUILDING]
            if free:
                return free[0]
        initial = g.state.is_initial_build_phase
        if initial:
            # Value resources on the second placement, but no imaginary starting hand.
            self.econ = 1.0
        budget = obs.cpu_left if obs.cpu_left is not None else 1.0
        deadline = time.process_time() + max(.002,min(.018,(budget-.05)*.12))
        scored=[]
        for i,a in enumerate(acts):
            v,s = self.score_action(g,a)
            scored.append((v,i,s))
        scored.sort(key=lambda x:(-x[0],x[1]))
        best_v,best_i,_ = scored[0]
        if initial or budget < .09 or acts[0].action_type in (AT.MOVE_ROBBER,AT.DISCARD_RESOURCE):
            return int(best_i)
        frontier = scored[:SEARCH_WIDTH]
        for depth in range(1,SEARCH_DEPTH):
            next_frontier=[]
            for _,root,s in frontier:
                if s is None or s.state.current_color()!=obs.me:
                    continue
                for a in s.playable_actions:
                    if a.action_type in (AT.ROLL,AT.END_TURN,AT.PLAY_KNIGHT_CARD):
                        continue
                    if time.process_time()>deadline or self.nodes>=SEARCH_NODES:
                        return int(best_i)
                    v,ss=self.score_action(s,a)
                    v -= depth*.001
                    if v>best_v:
                        best_v,best_i=v,root
                    next_frontier.append((v,root,ss))
            next_frontier.sort(key=lambda x:(-x[0],x[1]))
            frontier=next_frontier[:SEARCH_WIDTH]
        return int(best_i)
