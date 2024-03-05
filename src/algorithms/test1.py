"""
用于验证opt算法结果
将所有路径信息全部列出

"""
import itertools
import numpy as np
class SubsetsPermutations:
    def __init__(self, N):
        self.N_ = N
        self.subset = []
        self.permutations_ = []
        self.generate_subsets()
        self.generate_permutations()

    def generate_subsets(self):
        for i in range(len(self.N_) + 1):
            self.subset.extend(itertools.combinations(self.N_, i))

    def generate_permutations(self):
        for s in self.subset:
            permutation = list(itertools.permutations(s))
            self.permutations_.extend(permutation)

    def get_permutations(self):
        permutations = [list(i) for i in self.permutations_]
        return permutations

def add_s_d(N, s, d):
    x = [[s] + t + [d] for t in N]
    return x

N_=[2,3,4]
# for test the class named SubsetsPermutations
subsets_permutations = SubsetsPermutations(N_)
N = subsets_permutations.get_permutations()
P= add_s_d(N,1,5)


# 距离数据
distance = np.array([[0., 69.06484223, 75.57585925, 71.65824584, 66.8282913],
                     [69.06484223, 0., 7.68940613, 12.99483813, 2.99111645],
                     [75.57585925, 7.68940613, 0., 17.89846051, 8.96155748],
                     [71.65824584, 12.99483813, 17.89846051, 0., 15.32024517],
                     [66.8282913, 2.99111645, 8.96155748, 15.32024517, 0.]])
# 充电情况数据
charge = np.array([[0., 1., 0., 1., 0.],
                   [1., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0.],
                   [1., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])
# 速度数据
speed = np.array([[11, 42, 25, 32, 41],
    [50, 58, 22, 21, 22],
    [52, 18, 57, 37, 59],
    [39, 53, 54, 60, 25],
    [57, 19, 10, 37, 48]])

timelimit = 4  # Time constraint     Unit: hour
energy = 13.4  # Electric quantity      unit: kw
energy_consume = 0.1344  # 1kilometers of electricity consumption
charge_power = 2.8
C_ = []
S_T = []
S_E = []
path = []
for p in P:
    c_ = 0
    t = 0
    e = energy
    m = len(p)
    for i in range(m - 1):
        l = p[i] - 1
        r = p[i + 1] - 1
        t += distance[l][r] / speed[l][r]
        c_ += energy_consume * distance[l][r]
        e -= energy_consume * distance[l][r]
        if charge[l][r] == 1:
            c_ -= distance[l][r] / speed[l][r] * charge_power
            e += distance[l][r] / speed[l][r] * charge_power
    C_.append(c_)
    S_T.append(t)
    S_E.append(e)
    if t <= timelimit and e>= 0:
        path.append(True)
    else:
        path.append(False)

m = len(P)
for i in range(m):
    if path[i] == True:
        print( "path:", P[i], "  costtime:", S_T[i], "  costenergy:", C_[i], "  otherenergy", S_E[i], "isok:", path[i])
