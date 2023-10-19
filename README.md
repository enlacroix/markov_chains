# markov_chains
Небольшой модуль, позволяющий характеризовать дискретный Марковский процесс по матрице переходов.

# пример использования
task = MarkovChain(matrix)
print(task.describe())
np.set_printoptions(precision=2)
print(task.canonicalForm())
task.draw()
