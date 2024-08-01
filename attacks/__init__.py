from attacks.GraD import GraD
from attacks.greedy_rbcd import GreedyRBCD
from attacks.lrbcd import LRBCD
from attacks.pga import PGA
from attacks.prbcd import PRBCD

attacks_map = {
    'prbcd': PRBCD,
    'greedy-rbcd': GreedyRBCD,
    'pga': PGA,
    'lrbcd': LRBCD,
    'grad': GraD,
}