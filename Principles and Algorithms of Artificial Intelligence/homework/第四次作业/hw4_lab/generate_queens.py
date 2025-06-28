import numpy as np


def gen_single_map():
    map_seq = np.zeros(8,dtype=int)
    for i in range(8):
        map_seq[i] = np.random.randint(1, 9)

    return map_seq


def gen_queen_list(max_length):
    queens_list = set()
    while len(queens_list) < max_length:
        tmp_list = gen_single_map()
        queens_list.add(tuple(tmp_list))

    return list(queens_list)

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        for row in data:
            # 将每个数组转换为字符串，并用空格分隔元素
            row_str = ' '.join(map(str, row))
            file.write(row_str + '\n')  # 写入文件，并在每行后添加换行符

#print(gen_single_map())


if __name__ == '__main__':
    data = gen_queen_list(1000)
    save_to_file(data,'eight_queen_seqs.txt')
    print(f"successfully generate {len(data)} samples!")