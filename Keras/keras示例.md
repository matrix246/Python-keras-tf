# python 中函数的使用

## 定义简单的函数
```
# 定义函数
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

```
### 注意


| -   | 含义 | python中的表达 |
| --- | ---- | -------------- |
|     | 且   | and            |
|     | 或   | or             |
|     | 非   | not               |
|     | else if | elif           |
|     |         |                |
|     |         |                |

## 定义复杂点的函数
