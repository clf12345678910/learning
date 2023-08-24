# import random
#
# xs = [x for x in range(100)]
# ys = [x*3+2+random.random()/10 for x in xs]
#
# lr = 0.0001
#
# def GD():
#     w = random.random()
#     b = random.random()
#     for i in range(1000000):
#         for x, y in zip(xs, ys):
#             h = w * x + b
#             o = h - y
#
#             # 损失函数使用均方差损失函数，其可以准确预测预测值与真实值的误差
#             loss = o ** 2
#
#             dw = 2 * o * x
#             db = 2 * o
#
#             w = w - dw * lr
#             b = b - db * lr
#
#             if i % 100000 == 0:
#                 print(w,b,loss)
#
# if __name__ == '__main__':
#     GD()


import random

lr = 0.00001
xs = [x for x in range(100)]
ys = [3 * x + random.random() / 10 for x in xs]


def GD():
    # 随机初始化权重参数
    w = random.random()
    b = random.random()

    for i in range(100000):
        for x, y in zip(xs, ys):
            h = w * x + b
            o = h - y

            # 使用均方差损失函数
            loss = o ** 2

            dw = 2 * o * x
            db = 2 * o

            w = w - lr * dw
            b = b - lr * db

            if i % 100 == 0:
                print(w, b, loss)


if __name__ == '__main__':
    GD()
