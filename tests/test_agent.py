"""Behavioral regressions for the submitted decision interface and planning."""
import importlib.util
import os
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(os.environ['CATAN_KIT_ROOT']).resolve()
sys.path.insert(0,str(ROOT))
import catan_kit
from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionPrompt, ActionType as AT, RESOURCES

SOURCE=Path(os.environ.get('CATAN_TEST_AGENT',str(Path(__file__).resolve().parents[1]/'submission/agent.py')))
spec=importlib.util.spec_from_file_location('candidate',SOURCE)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def position(hand=(0,0,0,0,0),vp=2):
    g=Game([RandomPlayer(Color.RED),RandomPlayer(Color.BLUE)],seed=4271)
    while g.state.is_initial_build_phase:
        g.execute(g.playable_actions[0])
    st=g.state
    me=st.current_color()
    key='P%d_'%st.color_to_index[me]
    st.player_state[key+'HAS_ROLLED']=True
    st.player_state[key+'VICTORY_POINTS']=vp
    st.player_state[key+'ACTUAL_VICTORY_POINTS']=vp
    st.current_prompt=ActionPrompt.PLAY_TURN
    for i,r in enumerate(RESOURCES):
        for k in ('P0_','P1_'):
            st.player_state[k+r+'_IN_HAND']=hand[i] if k==key else 0
        st.resource_freqdeck[i]=19-hand[i]
    g.playable_actions=generate_playable_actions(st)
    return g,me


class AgentBehavior(unittest.TestCase):
    def test_opening_is_legal_and_does_not_change_observation(self):
        for seed in (3,104,2917):
            g=Game([RandomPlayer(Color.RED),RandomPlayer(Color.BLUE)],seed=seed)
            a=module.Agent()
            for _ in range(8):
                actions=list(g.playable_actions)
                ps=dict(g.state.player_state)
                buildings=dict(g.state.board.buildings)
                index=a.decide(Obs(g,g.state.current_color()),actions)
                self.assertIs(type(index),int)
                self.assertTrue(0<=index<len(actions))
                self.assertEqual(g.state.player_state,ps)
                self.assertEqual(g.state.board.buildings,buildings)
                g.execute(actions[index])
            self.assertFalse(g.state.is_initial_build_phase)

    def test_two_bank_trades_complete_a_winning_city(self):
        g,me=position((4,4,0,2,1),9)
        a=module.Agent()
        chosen=[]
        for _ in range(3):
            actions=g.playable_actions
            obs=Obs(g,me)
            obs.cpu_left=1.0
            index=a.decide(obs,actions)
            chosen.append(actions[index].action_type)
            g.execute(actions[index])
        self.assertEqual(chosen,[AT.MARITIME_TRADE,AT.MARITIME_TRADE,AT.BUILD_CITY])
        self.assertEqual(g.winning_color(),me)

    def test_dev_expectation_does_not_depend_on_sampled_deck_order(self):
        g,me=position((0,0,2,2,3))
        a=module.Agent()
        a.prepare(g,me)
        action=next(x for x in g.playable_actions if x.action_type==AT.BUY_DEVELOPMENT_CARD)
        first=a.score_action(g,action)[0]
        rev=g.copy()
        rev.state.development_listdeck.reverse()
        second=a.score_action(rev,action)[0]
        self.assertAlmostEqual(first,second,places=9)

    def test_free_road_phase_has_a_separate_cache_entry(self):
        g,me=position()
        a=module.Agent()
        a.prepare(g,me)
        first=a.board_terms(g)[0]
        future=g.copy()
        future.state.is_road_building=True
        future.state.free_roads_available=2
        second=a.board_terms(future)[0]
        self.assertAlmostEqual(second-first,.26,places=9)

    def test_clock_reserve_returns_a_legal_build_without_search(self):
        g,me=position((0,0,0,2,3),9)
        obs=Obs(g,me)
        obs.cpu_left=.001
        obs.sample_world=lambda rng:self.fail('Clock reserve must not start search')
        actions=g.playable_actions
        index=module.Agent().decide(obs,actions)
        self.assertIs(type(index),int)
        self.assertTrue(0<=index<len(actions))
        self.assertEqual(actions[index].action_type,AT.BUILD_CITY)


if __name__=='__main__':
    unittest.main(verbosity=2)
