"""Public-information features shared by offline training and exported agents."""
import math
from catanatron.models.enums import RESOURCES, CITY, SETTLEMENT

def public_features(g, me, node_prod):
    st=g.state
    ps=st.player_state
    opp=next(c for c in st.colors if c!=me)
    k='P%d_'%st.color_to_index[me]
    ek='P%d_'%st.color_to_index[opp]
    b=st.board
    prod={me:[0.]*5,opp:[0.]*5}
    blocked={me:[0.]*5,opp:[0.]*5}
    rates={me:[4]*5,opp:[4]*5}
    nodes={me:set(),opp:set()}
    for n,(c,kind) in b.buildings.items():
        nodes[c].add(n)
        for i,v in enumerate(node_prod[n]):
            prod[c][i]+=(2 if kind==CITY else 1)*v
    tile=b.map.tiles[b.robber_coordinate]
    if tile.resource:
        i=RESOURCES.index(tile.resource)
        for n in tile.nodes.values():
            if n in b.buildings:
                c,kind=b.buildings[n]
                blocked[c][i]+=(2 if kind==CITY else 1)*(6-abs(7-tile.number))/36
    for r,ns in b.map.port_nodes.items():
        for c in (me,opp):
            if nodes[c].intersection(ns):
                if r is None:
                    rates[c]=[min(v,3) for v in rates[c]]
                else:
                    rates[c][RESOURCES.index(r)]=2
    x=[ps[k+'ACTUAL_VICTORY_POINTS'],ps[ek+'VICTORY_POINTS'],
       ps[k+'VICTORY_POINTS'],st.num_turns, int(st.current_color()==me)]
    for c,key in ((me,k),(opp,ek)):
        p=prod[c]
        h=[ps[key+r+'_IN_HAND'] for r in RESOURCES]
        x.extend(p)
        x.extend(math.log1p(v/.15) for v in p)
        x.extend(blocked[c])
        x.extend(h)
        x.extend(rates[c])
        x.extend([sum(p),sum(h),sum(v>0 for v in p),
                  len(st.buildings_by_color[c][CITY]),len(st.buildings_by_color[c][SETTLEMENT]),
                  b.road_lengths[c],ps[key+'PLAYED_KNIGHT'],int(ps[key+'HAS_ARMY']),int(b.road_color==c)])
        for cost in ((0,0,0,2,3),(1,1,1,1,0),(0,0,1,1,1)):
            deficit=sum(max(0,cost[i]-h[i]) for i in range(5))
            spare=sum(max(0,h[i]-cost[i])/rates[c][i] for i in range(5))
            x.append(max(0,deficit-spare))
            x.append(max(max(0,cost[i]-h[i])/(p[i]+.08) for i in range(5)))
    devs=('KNIGHT','VICTORY_POINT','ROAD_BUILDING','YEAR_OF_PLENTY','MONOPOLY')
    x.extend(ps[k+d+'_IN_HAND'] for d in devs)
    x.append(sum(ps[ek+d+'_IN_HAND'] for d in devs))
    return x
