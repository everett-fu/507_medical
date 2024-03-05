import itertools
import generate_graph
import numpy as np


# 完全子集全排列
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


# for test the class named SubsetsPermutations
"""subsets_permutations = SubsetsPermutations(N_)
all_permutations = subsets_permutations.get_permutations()
print(all_permutations)"""


# Variables with a value of 404 indicate that they are not specified
class Searchbestpath:
    def __init__(self, P, d, c, s):
        self.timelimit = 4  # Time constraint     Unit: hour
        self.energy = 13.4# Electric quantity      unit: kw
        self.energy_consume = 0.1344  # 1kilometers of electricity consumption
        self.charge_power = 2.8
        self.distance = d
        self.charge = c
        self.speed = s
        self.P = P

    # Determine whether the line meets the remaining power constraints and time constraints
    def judge(self, p):
        energy = self.energy
        judge_e = 1
        sum_t = 0
        n = len(p)
        for i in range(n - 1):
            l = p[i]-1
            r = p[i+1]-1
            energy = energy - self.distance[l][r] * self.energy_consume
            sum_t+= self.distance[l][r] / self.speed[l][r]
            if self.charge[l][r] == 1:
                energy += self.charge_power * (self.distance[l][r] / self.speed[l][r])
            judge_e = judge_e * energy
            if judge_e<0 or sum_t >self.timelimit:
                return False
        return True

    # Traverse all paths and find the path that consumes the least amount of power among all paths that satisfy the constraint
    def get_bestpath(self):
        best_p = []
        c_ = float("inf")
        for p in self.P:
            if self.judge(p) is True:
                print(p)
                c = 0
                n = len(p)
                for i in range(n - 1):
                    l = p[i]-1
                    r = p[i+1]-1
                    if self.charge[l][r] > 0:
                        c = c + self.energy_consume * self.distance[l][r] - (self.distance[l][r] / self.speed[l][r]) * self.charge_power
                    else:
                        c = c + self.energy_consume * self.distance[l][r]
                if c < c_:
                    c_ = c
                    best_p = p
        return best_p,c_


# Add a start and end point
def add_s_d(N, s, d):
    x = [[s] + t + [d] for t in N]
    return x


def main():
    N = [x for x in range(1, 6)]
    s = 1
    d = 5
    N_ = [i for i in N if i != s and i != d]

    subsets_permutations = SubsetsPermutations(N_)
    all_permutations = subsets_permutations.get_permutations()
    # print(all_permutations)
    P = add_s_d(all_permutations, s, d)
    # print(P)
    """distance, charge, speed = generate_graph.main() #Call function to generate the matrix"""

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

    print(f"distance\n",distance)
    print(f"charge\n",charge)
    print(f"speed\n",speed)
    find_best_path = Searchbestpath(P, distance, charge, speed)
    best_path ,c_= find_best_path.get_bestpath()
    print(f"best_path:",best_path)
    print(f"c_:",c_)


if __name__ == "__main__":
    main()
