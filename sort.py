import math

def lengthOfLongestSubstring(s: str) -> int:
    max_len = 1
    lenth = len(s)
    current = s[0]
    for i in range(1, lenth):
        single = s[i]
        current += single
        for j in range(1, len(current)):
            select_string = current[-j:]
            if select_string in current[:-1]:
                max_len = max(max_len, len(current)-1)
                current = single
                continue
    return max_len      

print(lengthOfLongestSubstring("abcabcbb"))