import numpy as np
from tqdm import tqdm
import random
import time
import math
import copy


def load_seq(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data_list = []
    for line in lines:
        # 使用空格分割每一行的字符串，并将结果转换为整数列表
        row_list = [int(num) for num in (line.split())]
        data_list.append(row_list)
    return data_list


def mismatched_nums(input_seq):
    # 计算与标准结果不一致的错位数
    # goal_seq = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mismatched = 0
    for i in range(len(input_seq)):
        if input_seq[i] == 0:
            continue
        else:
            if input_seq[i] != i:
                mismatched += 1

    return mismatched



def manhattan_distance(input_seq):
    # 计算与标准结果不一致的曼哈顿距离
    distance = 0
    for i in range(len(input_seq)):
        if input_seq[i] == 0:
            continue
        distance += abs(i // 3 - input_seq[i] // 3) + abs(i % 3 - input_seq[i] % 3)
    return distance


def successor_code(input_seq):
    # 找到当前状态的所有后继状态
    blank_index = 0
    for i in range(len(input_seq)):
        if input_seq[i] == 0:
            blank_index = i
            break
    successor_list = []
    # 向上移动
    if blank_index // 3 <= 1:
        tmp = blank_index + 3
        tmp_seq = copy.deepcopy(input_seq)
        tmp_seq[blank_index] = tmp_seq[tmp]
        tmp_seq[tmp] = 0
        successor_list.append(tmp_seq)
        # print("down: ",tmp_seq)
    if blank_index // 3 >= 1:
        tmp = blank_index - 3
        tmp_seq = copy.deepcopy(input_seq)
        tmp_seq[blank_index] = tmp_seq[tmp]
        tmp_seq[tmp] = 0
        successor_list.append(tmp_seq)
        # print("up: ", tmp_seq)
    if blank_index % 3 <= 1:
        tmp = blank_index + 1
        tmp_seq = copy.deepcopy(input_seq)
        tmp_seq[blank_index] = tmp_seq[tmp]
        tmp_seq[tmp] = 0
        successor_list.append(tmp_seq)
        # print("right: ", tmp_seq)
    if blank_index % 3 >= 1:
        tmp = blank_index - 1
        tmp_seq = copy.deepcopy(input_seq)
        tmp_seq[blank_index] = tmp_seq[tmp]
        tmp_seq[tmp] = 0
        successor_list.append(tmp_seq)
        # print("left: ", tmp_seq)
    return successor_list


def steepest_mount_climbing(input_seq, method='mismatch'):
    cur_seq = input_seq
    if method == 'mismatch':
        cur_distance = mismatched_nums(input_seq)
    else:
        cur_distance = manhattan_distance(input_seq)
    fail_flag = False
    search_steps = 0
    while (fail_flag == False) and (cur_distance != 0):
        successor = successor_code(cur_seq)
        cost = []
        dicts = []
        search_steps += 1
        for s in successor:
            if method == 'mismatch':
                tmp_distance = mismatched_nums(s)
            else:
                tmp_distance = manhattan_distance(s)
            dicts.append({'seqs': s, 'distance': tmp_distance})
            cost.append(tmp_distance)
        min_cost = min(cost)
        if min_cost >= cur_distance:  # 找到局部极小值，结束爬山法
            fail_flag = True
            break
        else:
            tmp = []
            for d in dicts:
                if d['distance'] == min_cost:
                    tmp.append(d['seqs'])
            cur_seq = random.choice(tmp)
            if method == 'mismatch':
                cur_distance = mismatched_nums(cur_seq)
            else:
                cur_distance = manhattan_distance(cur_seq)

    answer = cur_seq
    return fail_flag, answer, cur_distance, search_steps


def first_choice_mount_climbing(input_seq, method='mismatch'):
    cur_seq = input_seq
    if method == 'mismatch':
        cur_distance = mismatched_nums(input_seq)
    else:
        cur_distance = manhattan_distance(input_seq)
    fail_flag = False
    search_steps = 0
    while (fail_flag == False) and (cur_distance != 0):
        successor = successor_code(cur_seq)
        count = 0
        search_steps += 1
        for s in successor:
            if method == 'mismatch':
                tmp_distance = mismatched_nums(s)
            else:
                tmp_distance = manhattan_distance(s)
            count += 1
            if tmp_distance < cur_distance:  # 找到一个更优的解
                cur_seq = s  # 状态更新：找到优于当前的结果即更新，并结束当前探索
                break
            else:
                continue
        if count >= len(successor):  # 没找到
            fail_flag = True
            break
        if method == 'mismatch':
            cur_distance = mismatched_nums(cur_seq)
        else:
            cur_distance = manhattan_distance(cur_seq)
    answer = cur_seq
    return fail_flag, answer, cur_distance, search_steps


def perturb(input_seq):
    # 随机选择一个扰动
    successor = successor_code(input_seq)
    return random.choice(successor)


def Simulated_Annealing(input_seq, max_search_cycle=4, initial_temp=5, cooling_rate=0.9999, min_temp=0.0001,
                        method='mismatch'):
    current_temp = initial_temp
    current_solution = input_seq
    search_steps = 0
    if method == 'mismatch':
        current_distance = mismatched_nums(current_solution)
    else:
        current_distance = manhattan_distance(current_solution)
    while current_temp > min_temp and current_distance > 0:
        for i in range(max_search_cycle):
            new_solution = perturb(current_solution)
            if method == 'mismatch':
                new_distance = mismatched_nums(new_solution)
            else:
                new_distance = manhattan_distance(new_solution)
            if new_distance < current_distance or random.random() < math.exp(
                    (current_distance - new_distance) / current_temp):
                current_solution = new_solution
                current_distance = new_distance
                if current_distance == 0:
                    break

            search_steps += 1

        current_temp *= cooling_rate
    fail_flag = 0 if current_distance == 0 else 1
    return fail_flag, current_solution, current_distance, search_steps


if __name__ == '__main__':
    print("testing")
    data_list = load_seq("eight_code_seqs_10000.txt")
    start = time.time()
    success = [0, 0, 0]  # 最陡爬山、首选爬山、随机重启(最陡、随机)、模拟退火
    fail = [0, 0, 0]
    steps = [0, 0, 0]
    total = len(data_list)

    print("testing steepest mount climbing...")
    time.sleep(0.5)
    for seq in tqdm(data_list):
        fail_flag, answer, cur_distance, search_steps = steepest_mount_climbing(seq)
        if fail_flag:
            fail[0] += 1
        else:
            success[0] += 1
            steps[0] += search_steps

    end = time.time()
    time.sleep(0.5)
    print(
        f"total samples: {total}, success: {success[0]}, fail: {fail[0]}, avg steps: {steps[0] / (success[0] + 1e-5)} \ntotal cost time: {end - start-0.5}s\n")


    start = time.time()
    print("testing first choice mount climbing...")
    time.sleep(0.5)
    for seq in tqdm(data_list):
        fail_flag, answer, cur_distance, search_steps = first_choice_mount_climbing(seq)
        if fail_flag:
            fail[1] += 1
        else:
            success[1] += 1
            steps[1] += search_steps
    end = time.time()
    time.sleep(0.5)
    print(
        f"total samples: {total}, success: {success[1]}, fail: {fail[1]}, avg steps: {steps[1] / (success[1] + 1e-5)} \ntotal cost time: {end - start-0.5}s\n")
    start = time.time()

    print("testing Simulated_Annealing...")
    for seq in tqdm(data_list):
        fail_flag, answer, cur_distance, search_steps = Simulated_Annealing(seq, max_search_cycle=100, initial_temp=500,cooling_rate=0.9, min_temp=0.1)
        # max_search_cycle=30, initial_temp=5,cooling_rate=0.999, min_temp=0.001
        if fail_flag:
            fail[2] += 1
        else:
            success[2] += 1
            steps[2] += search_steps
    end = time.time()
    print(
        f"total samples: {total}, success: {success[2]}, fail: {fail[2]}, avg steps: {steps[2] / (success[2] + 1e-5)} \ntotal cost time: {end - start}s\n")

