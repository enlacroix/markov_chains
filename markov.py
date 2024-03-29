from collections import Counter
from typing import Self
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from math import gcd
from functools import reduce

from examples import *
import numpy as np

from trp.utils import array_to_tex_matrix

np.set_printoptions(precision=4, suppress=True)


class MarkovChain:
    def __init__(self, matrix: np.array, stochastic_property=False):
        """
        :param matrix:
        :param stochastic_property: Позволит быстро заполнить матрицу, не проверяя корректность введённых данных.
        """
        assert all(map(lambda s: s >= 0, np.nditer(matrix))) is True, 'Элементы матрицы должны быть неотрицательными.'
        assert not stochastic_property or all(map(lambda s: s == 1, np.sum(matrix, axis=1))) is True, 'Сумма чисел во всех строках матрицы должна быть равна 1.'
        assert matrix.shape[0] == matrix.shape[1], 'Передаваемая матрица должна быть квадратной.'
        self.states: list[int] = list(range(1, matrix.shape[0] + 1))
        self.matrix: np.array = matrix
        self.graph = self.createTransitionGraph()

    def createTransitionGraph(self) -> nx.DiGraph:
        graph = nx.DiGraph(directed=True)
        graph.add_nodes_from(self.states)
        for state in self.states:
            for target in self.states:
                if self.matrix[state - 1][target - 1] != 0:
                    graph.add_edge(state, target)
        return graph

    def draw(self) -> None:
        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        plt.show()

    def get_lim_P_t(self, state1: int, state2: int, time):
        return np.linalg.matrix_power(self.matrix, time)[state1 - 1, state2 - 1]

    def _getStrongComponents(self):
        return list(nx.strongly_connected_components(self.graph))

    def orderedRelevantList(self) -> list[list[int]]:
        result = []
        nonrelev = []
        for comp in self._getStrongComponents():
            if self.isRelevant(comp):
                result.append(list(comp))
            else:
                nonrelev += list(comp)
        result.append(nonrelev)
        return result

    def canonicalForm(self) -> np.array:
        """
        [[1, 3, 4, 6, 7] - сущ, [...] - сущ, [2, 5, 8] - несущ]
        [[
         [1, 6],
         [4, 3, 7]
         ],
         [2, 5, 8]
        ]
        C[i][j] = M
        :return:
        """
        canonic = np.zeros((len(self.states), len(self.states)))
        canonic_list = list(itertools.chain(*[self.findCyclicSubclasses(sublist, self.period(sublist[0]), extendFlag=True) for sublist in self.orderedRelevantList()[:-1]])) + \
                       self.orderedRelevantList()[-1]
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                canonic[i][j] = self.matrix[canonic_list[i] - 1][canonic_list[j] - 1]
        return canonic

    def findCyclicSubclasses(self, clsStates: list[int], period: int, extendFlag: bool = False) -> list[list[int]] | list[int]:
        """
        [1 4 5 2]
        [[1 2],
         [4 5]]
        """
        if period < 2:
            return clsStates
        result = []
        for _ in range(period):
            result.append([])
        result: list[list[int]]
        processed = []
        result[0].append(clsStates[0])

        def add_neighbours(node):
            index = None
            for i, sublist in enumerate(result):
                if node in sublist:
                    index = i
                    break
            for n in [k for k in self.graph.neighbors(node) if k not in itertools.chain(*result)]:
                result[(index + 1) % period].append(n)
            processed.append(node)

        while [n for n in itertools.chain(*result) if n not in processed]:
            for n in [n for n in itertools.chain(*result) if n not in processed]:
                add_neighbours(n)

        if extendFlag:
            new_result = []
            for sb in result:
                new_result.extend(sb)
            return new_result

        return result

    def describe(self) -> str:
        """
        Существенность.
        :return:
        """
        result = ''
        for subgroup in self._getStrongComponents():
            if self.isRelevant(subgroup):
                d = self.period(list(subgroup)[0])
                result += f'{subgroup}: существенная, период {d} \n'
                if d > 1:
                    result += 'Циклические подклассы: ' + str(self.findCyclicSubclasses(list(subgroup), d))
            else:
                result += f'{subgroup}: несущественная.'
            result += '\n'
        return result

    def isRelevant(self, subgroup) -> bool:
        fNode = list(subgroup)[0]
        for node in nx.single_source_shortest_path(self.graph, fNode):
            if nx.has_path(self.graph, node, fNode):
                continue
            return False
        return True

    def period(self, node) -> int:
        powers = [i for i in range(1, len(self.states) + 1) if np.linalg.matrix_power(self.matrix, i)[node - 1][node - 1] > 0]
        return reduce(gcd, powers) if powers else 0

    def find_min_len_of_path2self(self, node):
        return min(map(lambda s: len(s), [k for k in list(nx.simple_cycles(self.graph)) if node in k]))

    @classmethod
    def fastinit(cls, data: list[list]) -> Self:
        """
        Быстрая инициализация объекта. Достаточно передать список списков строк, содержащих индексы, где элементы отличны от нуля.
        t = MarkovChain.fastinit([
            [1, 4],
            [2, 3],
            [4],
            [6],
            [0],
            [1, 2],
            [3],
            [1]
            ])
        """
        matrix = np.zeros((len(data), len(data)))
        for i, lst in enumerate(data):
            for k in lst:
                matrix[i][k] = 1
        return cls(matrix, stochastic_property=False)

    def simulate_trajectory(self, num_steps: int, v0: np.array) -> list[int]:
        n_states = len(self.states)
        trajectory = [np.argmax(v0)]
        for _ in range(num_steps):
            current_state = trajectory[-1]
            next_state = np.random.choice(n_states, p=self.matrix[current_state])
            trajectory.append(next_state)
        return trajectory

    def draw_trajectory(self, num_steps: int, initial_vectors: list[np.array]):
        plt.xlabel('Шаги')
        plt.ylabel('Состояния')
        plt.title(f'Траектория цепи Маркова для {num_steps} шагов')
        plt.yticks(range(0, len(self.states)))
        for v0 in initial_vectors:
            plt.plot(self.simulate_trajectory(num_steps, v0))
        plt.show()

    def find_stationary(self):
        """
        vP = v (v - вектор-строка, стационарное распределение)
        v (P - E) = 0
        Сумма всех компонент v = 1, т.к. это стохастический вектор.
        (P - E) ^ T v ^ T = 0
        Выкинем любую строку из (P - E) ^ T, т.к. они линейно зависимые и добавим строку, содержащую информацию о сумме v.
        :return:
        """
        N = len(self.states)
        A = (self.matrix - np.eye(N)).T
        b = np.append(np.zeros(N - 1), 1)
        A = np.delete(A, 0, axis=0)
        A = np.vstack([A, np.ones((1, A.shape[1]))])
        return np.linalg.solve(A, b)

    def construct_vt(self, v0: np.array, t: int):
        vt = np.zeros(len(self.states))
        final_states = [self.simulate_trajectory(t, v0)[-1] for _ in range(len(self.states))]
        for key, value in Counter(final_states).items():
            vt[key] = value / len(self.states)
        return vt


