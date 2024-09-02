from typing import Tuple, Dict, Any, List

import gurobipy as gp

# import numpy as np

# input data
model = gp.Model("PPE")
model.Params.OutputFlag = 0

T = 9  # number of periods
N = 21

demand = [20, 22, 24, 15, 13, 16, 15, 16, 18, 21, 18, 21, 23, 20, 19, 23, 20, 22, 25, 23, 24]
periods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]
par_list = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 5, 9, 10, 11, 12, 13, 14, 15, 16, 14, 18, 19]
canOrderA_list = [True, False, False, True, False, False, True, False, False, True, False, False, True, False, False,
                  True, False, False, True, False, False]
c0 = [15, 0, 0, 18, 0, 0, 18, 0, 0, 18, 0, 0, 18, 0, 0, 17, 0, 0, 17, 0, 0]
# c0 = [9, 0, 0, 13, 0, 0, 15, 0, 0, 14, 0, 0, 14, 0, 0, 14, 0, 0, 15, 0, 0]
c1 = [10, 14, 15, 14, 13, 15, 16, 15, 14, 16, 15, 14, 14, 13, 15, 16, 15, 14, 16, 15, 14]
h0 = [10, 9, 10, 9, 9, 11, 10, 9, 10, 10, 9, 10, 9, 9, 11, 10, 9, 10, 10, 9, 10]
h1 = [8, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 9]
beta0 = [60, 0, 0, 70, 0, 0, 80, 0, 0, 80, 0, 0, 80, 0, 0, 70, 0, 0, 80, 0, 0]
beta1 = [80, 60, 70, 70, 80, 90, 80, 70, 70, 80, 70, 70, 70, 80, 90, 80, 70, 70, 80, 70, 70]
u_list = [10, 11, 12, 10, 10, 11, 10, 12, 10, 10, 12, 10, 10, 10, 11, 10, 12, 10, 10, 12, 10]


# leaf_node_list = [7, 10, 16, 19]  # record the leaf node

class policyB:
    def __init__(self, a_interval, total_cost):
        self.a = a_interval
        self.cost = total_cost


class policy:
    def __init__(self, begin, end, a_interval, b_interval, total_cost):
        self.begin = begin
        self.end = end
        self.a = a_interval
        self.b = b_interval
        self.cost = total_cost


class node:
    def __init__(self, num, period, par, canOrderA, d, h_0, h_1, c_0, c_1, beta_0, beta_1, u):
        self.num = num  # node number
        self.period = period  # period
        self.par = par  # parent node, int
        self.canOrderA = canOrderA  # whether you can order product A, bool
        self.d = d  # demand
        self.h0 = h_0  # holding cost for product A
        self.h1 = h_1  # holding cost for product B
        self.c0 = c_0  # ordering cost for product A
        self.c1 = c_1  # ordering cost for product B
        self.beta_0 = beta_0  # setup fee for product A
        self.beta_1 = beta_1  # setup fee for product B
        self.u = u  # penalty cost for product B
        self.x0 = model.addVar(lb=0, vtype=gp.GRB.INTEGER)  # order quantity for product A
        self.x1 = model.addVar(lb=0, vtype=gp.GRB.INTEGER)  # order quantity for product B
        self.q0 = model.addVar(lb=0, vtype=gp.GRB.INTEGER)  # inventory level for product A
        self.q1 = model.addVar(lb=0, vtype=gp.GRB.INTEGER)  # inventory level for product B
        self.y0 = model.addVar(lb=0, vtype=gp.GRB.INTEGER)  # usage for product A
        self.y1 = model.addVar(lb=0, vtype=gp.GRB.INTEGER)  # usage for product B
        self.z0 = model.addVar(vtype=gp.GRB.BINARY)  # whether to order product A
        self.z1 = model.addVar(vtype=gp.GRB.BINARY)  # whether to order product B


# initialization
nodes = []
policys = {}
for i in range(N):
    # print(i, len(demand), len(periods), len(par_list), len(canOrderA_list), len(h0), len(h1), len(c0), len(c1), len(beta0), len(beta1), len(u_list))
    nodes.append(
        node(i, periods[i], par_list[i], canOrderA_list[i], demand[i], h0[i], h1[i], c0[i], c1[i], beta0[i], beta1[i],
             u_list[i]))

demand = {}
holding = {}
for i in range(N):
    for j in range(i, N):
        demand[i, j] = 0
        holding[i, j] = 0
        if i < j:
            for k in range(i, j + 1):
                demand[i, j] += nodes[k].d
                if k > i:
                    holding[i, j] += nodes[k].d
        elif i == j:
            demand[i, j] = nodes[i].d
            holding[i, j] = 0
        else:
            demand[i, j] = 0
            holding[i, j] = 0


def optimalSol(N_list) -> float:
    # objective function
    temp_def = 0
    for temp_i_def in N_list:
        temp_def += nodes[temp_i_def].x0 * nodes[temp_i_def].c0 + nodes[temp_i_def].x1 * nodes[temp_i_def].c1 + nodes[
            temp_i_def].q0 * nodes[temp_i_def].h0 + nodes[temp_i_def].q1 * nodes[temp_i_def].h1 + nodes[temp_i_def].z0 * \
                    nodes[temp_i_def].beta_0 + nodes[temp_i_def].z1 * nodes[temp_i_def].beta_1 + nodes[temp_i_def].u * \
                    nodes[temp_i_def].y1

    model.setObjective(temp_def, gp.GRB.MINIMIZE)

    # constraints
    for temp_i_def in N_list:
        if nodes[temp_i_def].par == -1:  # first node
            model.addConstr(nodes[temp_i_def].x0 == nodes[temp_i_def].y0 + nodes[temp_i_def].q0,
                            name="t{}_a_order_use_left".format(
                                temp_i_def))  # order quantity = usage + inventory level for A
            model.addConstr(nodes[temp_i_def].x1 == nodes[temp_i_def].y1 + nodes[temp_i_def].q1,
                            name="t{}_b_order_use_left".format(
                                temp_i_def))  # order quantity = usage + inventory level for B
        else:
            model.addConstr(
                nodes[temp_i_def].x0 + nodes[nodes[temp_i_def].par].q0 == nodes[temp_i_def].y0 + nodes[temp_i_def].q0,
                name="t{}_a_order_use_left".format(
                    temp_i_def))  # order quantity = usage + inventory level for A
            model.addConstr(
                nodes[temp_i_def].x1 + nodes[nodes[temp_i_def].par].q1 == nodes[temp_i_def].y1 + nodes[temp_i_def].q1,
                name="t{}_b_order_use_left".format(
                    temp_i_def))  # order quantity = usage + inventory level for B
        model.addConstr(nodes[temp_i_def].y0 + nodes[temp_i_def].y1 == nodes[temp_i_def].d,
                        name="t{}_demand".format(temp_i_def))  # usage of two products = demand
        model.addConstr(nodes[temp_i_def].x0 <= 100000 * nodes[temp_i_def].z0,
                        name="z{}_a".format(temp_i_def))  # ordering and setup operation for A
        model.addConstr(nodes[temp_i_def].x1 <= 100000 * nodes[temp_i_def].z1,
                        name="z{}_b".format(temp_i_def))  # ordering and setup operation for B
        if not nodes[temp_i_def].canOrderA:
            model.addConstr(nodes[temp_i_def].x0 == 0, name="t{}_no_order_A".format(temp_i_def))

    model.write("MasterModel.lp")
    model.optimize()

    print("Optimal value:", model.ObjVal)
    print("Optimal solution:")
    for temp_i_def in N_list:
        print("------------------------------------------------------")
        print("In time period {}, node {}, the demand is {}".format(nodes[temp_i_def].period, nodes[temp_i_def].num,
                                                                    nodes[temp_i_def].d))

        if nodes[temp_i_def].z0.x == 1 and nodes[temp_i_def].canOrderA:
            print("-> order from A, and the ordering quantity is {}".format(abs(nodes[temp_i_def].x0.x)))

        if nodes[temp_i_def].z1.x == 1:
            print("-> order from B, and the ordering quantity is {}".format(abs(nodes[temp_i_def].x1.x)))

        print("-> the usage of Product A is {}, and the usage of Product B is {}".format(abs(nodes[temp_i_def].y0.x),
                                                                                         abs(nodes[temp_i_def].y1.x)))

        if temp_i_def == 0:
            print("-> (before) inventory level: {} and {}".format(0, 0))
            print("-> (after) inventory level: {} and {}".format(abs(nodes[temp_i_def].q0.x),
                                                                 abs(nodes[temp_i_def].q1.x)))
        else:
            print("-> (before) inventory level: {} and {}".format(abs(nodes[nodes[temp_i_def].par].q0.x),
                                                                  abs(nodes[nodes[temp_i_def].par].q1.x)))
            print("-> (after) inventory level: {} and {}".format(abs(nodes[temp_i_def].q0.x),
                                                                 abs(nodes[temp_i_def].q1.x)))

    return model.ObjVal


# test the assumptions
def test(temp_scenario) -> None:
    test_assum_2 = True
    test_assum_1 = True
    test_assum_3_a = True
    test_assum_3_b = True
    delta_u = -10000
    min_h_a, min_h_b = -1, -1
    delta_c_a, delta_c_b = -10000, -10000
    delta_beta_a, delta_beta_b = -10000, -10000
    for i_ind in range(len(temp)):
        if nodes[temp[i_ind]].h0 < min_h_a:
            min_h_a = nodes[temp[i_ind]].h0
        if nodes[temp[i_ind]].h1 < min_h_b:
            min_h_b = nodes[temp[i_ind]].h1
        for j_ind in range(i_ind, len(temp)):
            if delta_u <= nodes[temp[j_ind]].u - nodes[temp[i_ind]].u:
                delta_u = nodes[temp[j_ind]].u - nodes[temp[i_ind]].u
            if delta_c_a <= nodes[temp[j_ind]].c0 - nodes[temp[i_ind]].c0 and nodes[temp[j_ind]].c0 != 0 and nodes[
                temp[i_ind]].c0 != 0:
                delta_c_a = nodes[temp[j_ind]].c0 - nodes[temp[i_ind]].c0
            if delta_c_b <= nodes[temp[j_ind]].c1 - nodes[temp[i_ind]].c1:
                delta_c_b = nodes[temp[j_ind]].c1 - nodes[temp[i_ind]].c1
            if delta_beta_a <= nodes[temp[j_ind]].beta_0 - nodes[temp[i_ind]].beta_0 and nodes[
                temp[j_ind]].beta_0 != 0 and nodes[temp[i_ind]].beta_0 != 0:
                delta_beta_a = nodes[temp[j_ind]].beta_0 - nodes[temp[i_ind]].beta_0
            if delta_beta_b <= nodes[temp[j_ind]].beta_1 - nodes[temp[i_ind]].beta_1:
                delta_beta_b = nodes[temp[j_ind]].beta_1 - nodes[temp[i_ind]].beta_1

    for i in temp:
        if nodes[i].c0 != 0 and nodes[i].c0 <= nodes[i].c1:
            test_assum_1 = False

        if nodes[i].h0 - nodes[i].h1 < delta_u:
            test_assum_2 = False

    if delta_beta_a >= 0:
        if min_h_a - delta_c_a < delta_beta_a:
            test_assum_3_a = False
    else:
        if min_h_a - delta_c_a < 0:
            test_assum_3_a = False

    if delta_beta_b >= 0:
        if min_h_b - delta_c_b < delta_beta_b:
            test_assum_3_b = False
    else:
        if min_h_b - delta_c_b < 0:
            test_assum_3_b = False

    print("The first assumption is {}".format(test_assum_1))
    print("The second assumption is {}".format(test_assum_2))
    print("The third assumption for A is {}".format(test_assum_3_a))
    print("The third assumption for B is {}".format(test_assum_3_b))
    print("\n\n========================================================")


temp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# test(temp)
# gurobi_sol = optimalSol(temp)

def cal_cost(b_start, b_end, a_start, a_end) -> float:
    temp_cost = 0
    if a_start is None and a_end is None:
        temp_cost += nodes[b_start].c1 * demand[b_start, b_end] + nodes[b_start].beta_1
        for i in range(b_start, b_end + 1):
            temp_cost += nodes[i].h1 * holding[i, b_end] + nodes[i].u * nodes[i].d
        return temp_cost
    else:
        demand_a = demand[a_start, a_end]  # product A need to order
        demand_b = demand[b_start, b_end] - demand[a_start, a_end]  # product B need to order

        if b_start == a_start:  # order A first
            temp_cost += nodes[a_start].c0 * demand_a + nodes[
                a_start].beta_0  # setup and ordering cost for A at a_start
            temp_cost += nodes[a_end + 1].c1 * demand_b + nodes[
                a_end + 1].beta_1  # setup and ordering cost for B at a_end + 1

            for i in range(a_start, a_end + 1):
                temp_cost += nodes[i].h0 * holding[i, a_end]

            for i in range(a_end + 1, b_end + 1):
                temp_cost += nodes[i].h1 * holding[i, b_end] + nodes[i].u * nodes[i].d
            return temp_cost
        else:
            temp_cost += nodes[b_start].c1 * demand_b + nodes[
                b_start].beta_1  # setup and ordering cost for B at b_start
            temp_cost += nodes[a_start].c0 * demand_a + nodes[
                a_start].beta_0  # setup and ordering cost for A at a_start

            for i in range(b_start, a_start):
                demand_b -= nodes[i].d
                temp_cost += nodes[i].h1 * demand_b + nodes[i].u * nodes[i].d  # holding and penalty cost for B

            for i in range(a_start, a_end + 1):
                temp_cost += nodes[i].h0 * holding[i, a_end]  # holding cost for A
                temp_cost += nodes[i].h1 * demand_b  # holding cost for B

            for i in range(a_end + 1, b_end + 1):
                demand_b -= nodes[i].d
                temp_cost += nodes[i].h1 * demand_b + nodes[i].u * nodes[i].d  # holding and penalty cost for B

            return temp_cost


def cal_cost_A(a_s, a_e) -> float:
    temp_cost = nodes[a_s].c0 * demand[a_s, a_e] + nodes[a_s].beta_0
    for temp_i in range(a_s, a_e + 1):
        temp_cost += nodes[temp_i].h0 * holding[temp_i, a_e]
    return temp_cost


def cal_del_cost(b_start, b_end, a_start, a_end) -> float:
    temp_cost = nodes[a_start].beta_0 + (nodes[a_start].c0 - nodes[b_start].c1) * demand[a_start, a_end]
    for temp_i in range(a_start, a_end + 1):
        temp_cost += (nodes[temp_i].h0 - nodes[temp_i].h1) * holding[temp_i, a_end]
        temp_cost -= nodes[temp_i].u * nodes[temp_i].d
    for temp_i in range(b_start, a_start):
        temp_cost -= nodes[temp_i].h1 * demand[a_start, a_end]
    return temp_cost


def FindNextA(cur, A_List) -> int | None:
    res = []
    for temp_i in A_List:
        if temp_i > cur:
            res.append(temp_i)
    if not res:
        return None
    return min(res)


def bestA(temp_i, temp_j) -> float:
    base = cal_cost(temp_i, temp_j, None, None)
    temp_bestA = base
    policyBs = {}
    A_List = []
    for temp_k in range(temp_i, temp_j + 1):
        if nodes[temp_k].canOrderA and temp_k != temp_i:
            A_List.append(temp_k)
            for temp_p in range(temp_k, temp_j + 1):
                temp_temp = cal_del_cost(temp_i, temp_j, temp_k, temp_p)
                policyBs[temp_k, temp_p] = policyB([temp_k, temp_p], temp_temp)

    cur_best = {}
    cur_best_cost = {}
    for temp_k in reversed(A_List):
        temp_min = 10000000
        for temp_p in range(temp_k, temp_j + 1):
            if FindNextA(temp_p, A_List) is None:
                temp_cost = policyBs[temp_k, temp_p].cost
                # print(temp_k, temp_p, temp_cost)
                if temp_min > temp_cost:
                    temp_min = temp_cost
                    cur_best[temp_k] = policyBs[temp_k, temp_p]
                    cur_best_cost[temp_k] = temp_cost
            else:
                next_a = FindNextA(temp_p, A_List)
                temp_cost = policyBs[temp_k, temp_p].cost + cur_best_cost[next_a]
                # print(temp_k, temp_p, next_a, policyBs[temp_k, temp_p].cost, cur_best_cost[next_a], temp_cost)
                if temp_min > temp_cost:
                    temp_min = temp_cost
                    cur_best[temp_k] = policyBs[temp_k, temp_p]
                    cur_best_cost[temp_k] = temp_cost

    # print(A_List)
    # for temp_i in A_List:
    #     print(cur_best[temp_i].a, cur_best_cost[temp_i])
    if A_List:
        return (min(cur_best_cost[A_List[0]] + base, temp_bestA))  # min(mixed A, pure B)
    else:
        return temp_bestA


# print(bestA(1, 2))

def getOpt() -> float:
    BestBPolicywithA = {}
    BestAPolicy = {}
    for temp_i in range(N):
        for temp_j in range(temp_i, N):
            if nodes[temp_i].canOrderA is False:
                BestBPolicywithA[temp_i, temp_j] = bestA(temp_i, temp_j)
            else:
                BestAPolicy[temp_i, temp_j] = cal_cost_A(temp_i, temp_j)

    #print(BestBPolicywithA[20, 20])
    #exit(0)
    BestPolicy = {}

    for temp_i in range(N - 1, -1, -1):
        min_cur = 1000000
        for temp_j in range(temp_i, N):
            if temp_j + 1 == N:
                if nodes[temp_i].canOrderA is False:
                    if min_cur > BestBPolicywithA[temp_i, temp_j]:
                        min_cur = BestBPolicywithA[temp_i, temp_j]
                        BestPolicy[temp_i] = BestBPolicywithA[temp_i, temp_j]
                    # print(temp_i, temp_j, BestBPolicywithA[temp_i, temp_j])
                else:
                    if min_cur > BestAPolicy[temp_i, temp_j]:
                        min_cur = BestAPolicy[temp_i, temp_j]
                        BestPolicy[temp_i] = BestAPolicy[temp_i, temp_j]
                    # print(temp_i, temp_j, BestAPolicy[temp_i, temp_j])
            else:
                if nodes[temp_i].canOrderA is False:
                    cur_cost_B = BestBPolicywithA[temp_i, temp_j] + BestPolicy[temp_j + 1]
                    # print(temp_i, temp_j, BestBPolicywithA[temp_i, temp_j], BestPolicy[temp_j + 1], cur_cost_B)
                    if min_cur > cur_cost_B:
                        min_cur = cur_cost_B
                        BestPolicy[temp_i] = cur_cost_B
                else:
                    cur_cost_A = BestAPolicy[temp_i, temp_j] + BestPolicy[temp_j + 1]
                    # print(temp_i, temp_j, BestAPolicy[temp_i, temp_j], BestPolicy[temp_j + 1], cur_cost_A)
                    if min_cur > cur_cost_A:
                        min_cur = cur_cost_A
                        BestPolicy[temp_i] = cur_cost_A
    # print(BestPolicy)
    return (BestPolicy[0])


# print("The best policy cost is ", best_policy_cost)
# print("=> The optimal policy is ordering B from {} to {} and ordering A from {} to {}".format(best_policy[0],
#                                                                                               best_policy[1],
#                                                                                               best_policy[2],
#                                                                                               best_policy[3]))


# compare the delta cost
def compareDelta(temp_i, temp_j, temp_a, temp_b) -> None:
    temp_del = cal_del_cost(temp_i, temp_j, temp_a, temp_b)
    temp_base = cal_cost(temp_i, temp_j, None, None)
    temp_final = cal_cost(temp_i, temp_j, temp_a, temp_b)
    if temp_final == temp_base + temp_del:
        print("The delta cost is correct")
    else:
        print("The delta cost is wrong")

# compareDelta(0, 8, 3, 4)
gurobi_sol = optimalSol(temp)
sol = getOpt()
print("The optimal cost (Gurobi) is ", gurobi_sol)
print("The optimal cost is ", sol)