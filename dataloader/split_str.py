import re

def extract_numbers_from_string(s):
    # 按'_'分隔字符串
    substrings = s.split('_')

    # 用正则表达式提取每个子字符串中的数字
    numbers = [re.findall(r'\d+', substring) for substring in substrings]

    # 将数字从嵌套列表中提取出来并转为整数类型
    numbers = [int(num) for sublist in numbers for num in sublist]

    return numbers