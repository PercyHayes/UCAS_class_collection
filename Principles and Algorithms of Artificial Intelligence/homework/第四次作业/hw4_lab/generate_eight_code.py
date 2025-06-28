import numpy as np

def gen_single_map():
    map_seq = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.random.shuffle(map_seq)
    return map_seq

def gen_code_list(max_length):
    code_list = set()
    while len(code_list)<max_length:
        tmp_list = gen_single_map()
        code_list.add(tuple(tmp_list))

    return list(code_list)

def save_to_file(data,filename):
    with open(filename, 'w') as file:
        for row in data:
            # 将每个数组转换为字符串，并用空格分隔元素
            row_str = ' '.join(map(str, row))
            file.write(row_str + '\n')  # 写入文件，并在每行后添加换行符

if __name__ == '__main__':
    data = gen_code_list(10000)
    save_to_file(data,'eight_code_seqs_10000.txt')
    print(f"successfully generate {len(data)} samples!")