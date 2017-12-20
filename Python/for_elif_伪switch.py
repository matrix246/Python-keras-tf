# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:41:50 2017

@author: Administrator
"""

# 定义函数。 根据分数判断 grade
def fun_grade(score):
    if(score > 100 or score < 0):
        print("value of score is not allowed.")    
    elif(score >= 90):
        print('grade is A')
    elif(score >= 80 and score < 90):
        print('grade is B+')
    elif(score >= 70 and score < 80):
        print('grade is B')
    elif(score >= 60 and score < 70):
        print('grade is C')       
    else:
        print('grade is D')
        
    
# 调用函数
fun_grade(45)
fun_grade(145)
fun_grade(95)
fun_grade(-45)
print('\n')

# 定义函数，求从 2 到 n 的所有 偶数 的和
def fun_sum_even(n):
    sum_even = 0
    for x in range(n//2 +1):
        sum_even += x 
    return 2 * sum_even 
        
# 调用函数
n = 5
print(fun_sum_even(n))
n = 8
print(fun_sum_even(n))
print('\n')
    

# 定义函数，使用字典实现 switch case 语句
# ref:http://python.jobbole.com/82008/
def numbers_to_strings(argument):
    switcher = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four"
    }
    return switcher.get(argument, "nothing")

# 调用函数
print(numbers_to_strings(0))
print(numbers_to_strings(9))
print('\n')
