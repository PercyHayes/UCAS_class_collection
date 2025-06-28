import numpy as np
import random
import time
import math
from tqdm import tqdm
from generate_queens import gen_single_map


def load_seq(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    file.close()
    data_list = []
    for line in lines:
        # 使用空格分割每一行的字符串，并将结果转换为整数列表
        row_list = [int(num) for num in (line.split())]
        data_list.append(row_list)
    return data_list


def attacked_queens_pairs(seqs):
    map = np.array([0] * 81)  # 创建一个有81个0的一维数组
    map = map.reshape(9, 9)  # 改为9*9二维数组。为方便后面使用，只用后八行和后八列的8*8部分，作为一个空白棋盘
    attacked_nums = 0  # 互相攻击的皇后对数初始化为0
    # 初始化
    for i in range(1, 9):
        map[seqs[i - 1]][i] = 1

    for i in range(1, 9):
        for j in list(range(1, i)) + list(range(i + 1, 9)):
            # 除当前列棋盘所在的行外的其他行
            if map[seqs[i - 1]][j] == 1:
                # 行攻击检查（列不可能攻击
                attacked_nums += 1

        row_index1 = row_index2 = seqs[i - 1]  # 第i列棋子所在的行
        # 检查左半对角线
        for j in range(i - 1, 0, -1):
            # 左上的检查
            if row_index1 != 1:
                row_index1 -= 1
                if map[row_index1][j] == 1:
                    attacked_nums += 1
            # 左下的检查
            if row_index2 != 8:
                row_index2 += 1
                if map[row_index2][j] == 1:
                    attacked_nums += 1

        # 检查右半对角线
        row_index1 = row_index2 = seqs[i - 1]  # 第i列棋子所在的行
        for j in range(i + 1, 9):
            if row_index1 != 1:
                row_index1 -= 1
                if map[row_index1][j] == 1:
                    attacked_nums += 1

            if row_index2 != 8:
                row_index2 += 1
                if map[row_index2][j] == 1:
                    attacked_nums += 1
    return int(attacked_nums / 2)


def successor_map(cur_seq):
    # 生成但钱输入序列的所有后继状态
    successor = []
    count = 0
    for item in cur_seq:
        for row_index in list(range(1, item)) + list(range(item + 1, 9)):
            tmp_seq = list(cur_seq)
            tmp_seq[count] = row_index
            successor.append(tmp_seq)
        count = count + 1
        if count == 8:
            break
    return successor


def Steepest_mount_climbing(input_seq):
    cur_seq = input_seq
    cur_attack_pairs = attacked_queens_pairs(cur_seq)
    fail_flag = False
    search_steps = 0
    while (fail_flag == False) and (cur_attack_pairs != 0):
        successor = successor_map(cur_seq)  # 56个后继
        cost = []
        dicts = []
        search_steps += 1
        for s in successor:
            tmp_attack_pairs = attacked_queens_pairs(s)
            dicts.append({'seqs': s, 'attacked_queens_pairs': tmp_attack_pairs})
            cost.append(tmp_attack_pairs)
        min_cost = min(cost)  # 找到最小攻击对数的移动
        if min_cost >= cur_attack_pairs:  # 找到局部极小值，结束爬山法
            fail_flag = True
            break
        else:
            tmp = []
            for d in dicts:
                if d['attacked_queens_pairs'] == min_cost:
                    tmp.append(d['seqs'])

            cur_seq = random.choice(tmp)
            cur_attack_pairs = attacked_queens_pairs(cur_seq)

    answer = cur_seq
    return fail_flag, answer, cur_attack_pairs, search_steps


def first_choice_mount_climbing(input_seq):
    cur_seq = input_seq
    cur_attack_pairs = attacked_queens_pairs(cur_seq)
    fail_flag = False
    search_steps = 0
    while (fail_flag == False) and (cur_attack_pairs != 0):
        successor = successor_map(cur_seq)  # 56个后继
        count = 0
        search_steps += 1
        for s in successor:
            tmp_attack_pairs = attacked_queens_pairs(s)
            count += 1
            if tmp_attack_pairs < cur_attack_pairs:  # 找到第一个，首选
                cur_seq = s  # 状态更新：找到优于当前的结果即更新，并结束当前探索
                break
            else:
                continue

        if count >= 55:  # 找不到比当前更优的后继
            fail_flag = True
            break
        cur_attack_pairs = attacked_queens_pairs(cur_seq)

    answer = cur_seq
    return fail_flag, answer, cur_attack_pairs, search_steps


def random_restart_climbing(input_seq, max_restart_time=5, method='first'):
    restart_cnt = 0
    seq = input_seq
    fail_flag = True
    steps = 0
    answer = seq
    cur_attack_pairs = 100000
    search_steps = 0
    while restart_cnt < max_restart_time and fail_flag == True:
        # print(restart_cnt)
        if method == 'first':  # 首步爬山法
            fail_flag, answer, cur_attack_pairs, search_steps = first_choice_mount_climbing(seq)
        else:
            fail_flag, answer, cur_attack_pairs, search_steps = Steepest_mount_climbing(seq)

        seq = gen_single_map()
        restart_cnt += 1
        steps += search_steps

    return fail_flag, answer, cur_attack_pairs, steps


def perturb(solution):
    # 随机扰动一列
    new_solution = list(solution)
    col_to_change = random.randint(0, 7)
    perturb_list = list(range(1, new_solution[col_to_change])) + list(range(new_solution[col_to_change] + 1, 9))
    new_solution[col_to_change] = random.choice(perturb_list)
    # print(col_to_change)
    # random.randint(1, 8)
    # print("input solution: ",solution)
    # print("new solution: ",new_solution)
    return new_solution


def Simulated_Annealing(input_seq, max_search_cycle=50, initial_temp=5, cooling_rate=0.99, min_temp=0.0001):
    current_temp = initial_temp
    current_solution = input_seq
    search_steps = 0
    current_attack_pairs = attacked_queens_pairs(current_solution)
    while current_temp > min_temp and current_attack_pairs > 0:
        for i in range(max_search_cycle):
            new_solution = perturb(current_solution)
            new_attack_pairs = attacked_queens_pairs(new_solution)
            if new_attack_pairs < current_attack_pairs or random.random() < math.exp(
                    (current_attack_pairs - new_attack_pairs) / current_temp):
                current_solution = new_solution
                current_attack_pairs = new_attack_pairs
                if current_attack_pairs == 0:
                    break

            search_steps += 1
        current_temp *= cooling_rate

    fail_flag = 0 if current_attack_pairs == 0 else 1
    return fail_flag, current_solution, current_attack_pairs, search_steps


if __name__ == '__main__':
    start = time.time()
    data_list = load_seq("eight_queen_seqs.txt")
    success = [0, 0, 0, 0, 0]  # 最陡爬山、首选爬上、随机重启(最陡、随机)、模拟退火
    fail = [0, 0, 0, 0, 0]
    steps = [0, 0, 0, 0, 0]
    total = len(data_list)
    
    print("testing steepest mount climbing...")
    for seq in data_list:
        fail_flag, answer, pairs, search_step = Steepest_mount_climbing(seq)
        if fail_flag == True:
            fail[0] += 1
        else:
            success[0] += 1
            steps[0] += search_step
    end = time.time()
    print(f"total samples: {total}, success: {success[0]}, fial: {fail[0]}, avg steps: {steps[0] / (success[0] + 1e-5)} \ntotal cost time: {end - start}s\n")
    start = time.time()

    print("testing first choice mount climbing...")
    for seq in data_list:
        fail_flag, answer, pairs, search_step = first_choice_mount_climbing(seq)
        if fail_flag == True:
            fail[1] += 1
        else:
            success[1] += 1
            steps[1] += search_step
    end = time.time()
    print(f"total samples: {total}, success: {success[1]}, fial: {fail[1]}, avg steps: {steps[1] / (success[1] + 1e-5)} \ntotal cost time: {end - start}s\n")
    start = time.time()
    
    print("testing random restart mount climbing(steepest)...")
    for seq in tqdm(data_list):
        fail_flag, answer, pairs, search_step = random_restart_climbing(seq,5,'steepest')
        if fail_flag == True:
            fail[2] += 1
        else:
            success[2] += 1
            steps[2] += search_step
    end = time.time()
    print(
        f"total samples: {total}, success: {success[2]}, fial: {fail[2]}, avg steps: {steps[2] / (success[2] + 1e-5)} \ntotal cost time: {end - start}s\n")
    start = time.time()

    print("testing random restart mount climbing(first)...")
    for seq in data_list:
        fail_flag, answer, pairs, search_step = random_restart_climbing(seq, 5,'first')
        if fail_flag == True:
            fail[3] += 1
        else:
            success[3] += 1
            steps[3] += search_step
    end = time.time()
    print(
        f"total samples: {total}, success: {success[3]}, fial: {fail[3]}, avg steps: {steps[3] / (success[3] + 1e-5)} \ntotal cost time: {end - start}s\n")
    start = time.time()
    
    print("testing Simulated_Annealing...")
    for seq in tqdm(data_list):
        fail_flag, answer, pairs, search_step = Simulated_Annealing(seq,max_search_cycle=1, initial_temp=5, cooling_rate=0.999, min_temp=0.001)
        # max_search_cycle=1, initial_temp=5, cooling_rate=0.99, min_temp=0.001
        if fail_flag == True:
            fail[4] += 1
        else:
            success[4] += 1
            steps[4] += search_step
    end = time.time()
    print(
        f"total samples: {total}, success: {success[4]}, fial: {fail[4]}, avg steps: {steps[4] / (success[4] + 1e-5)} \ntotal cost time: {end - start}s\n")

