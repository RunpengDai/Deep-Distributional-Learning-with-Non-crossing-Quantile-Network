


def split_list(input_list, n):
    length = len(input_list)
    return [input_list[i*length // n: (i+1)*length // n] for i in range(n)]
