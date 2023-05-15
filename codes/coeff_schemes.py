#!/usr/bin/python3
import numpy as np


# This is a program to calculate the coefficients of the differential format
def printinfo():
    print("The Program can the coeefficient")
    print("The scheme is:f'(j)=a(1)*f(j-k)+...+a(m)*f(j-k+m-1)")
    print("or f'(j)=F(j+1/2)-F(j-1/2)")
    print("Where:F(j+1/2)=b(1)*f(j-k+1)+...+b(m-1)*f(j-k+m-1)")
    print("------------------------------------------------------------")


def get_params():
    m = input("Please enter the number of base frame points 'm':")
    k = input("Please enter the position of leftmost point 'k':")
    order = input("Please enter the order of the derivative:")
    while order >= m:
        print("m points can only approximate the m-1 order derivate！")
        order = input("Please enter the order of the derivative again:")
    return (int(m), int(k), int(order))


# if calculate the 1 order derivate, b = [[0, 1, 0, ..., 0]]
def set_order(m, order):
    order_matrix = np.zeros(shape=(1, m))
    order_matrix[0, order] = 1
    return order_matrix


# calculate the coefficients of taylort expand, get the m order, then we can get the error
# 计算泰勒展开系数矩阵，多计算一阶用于算差分格式系数的误差项系数
# [[1, f'(0), f''(0), ..., ]
# [...]]
def get_taylor_coef(m, k):
    taylor_matrix = np.zeros(shape=(m, m + 1))
    for i in range(m):
        for j in range(m + 1):
            taylor_matrix[i, j] = np.power(-k + i, j) / np.math.factorial(j)
    return taylor_matrix


# a X = b -> a
def cal_matrixa(order_matrix, taylor_matrix):
    taylor_matrix_inv = np.linalg.inv(taylor_matrix[:, 0:-1])
    matrix_a = order_matrix.dot(taylor_matrix_inv)
    error_coef = matrix_a.dot(taylor_matrix[:, -1].reshape(-1, 1))
    return matrix_a, error_coef


# 转化为守恒型格式
def convert_to_const(matrix_a):
    _, n = matrix_a.shape
    matrix_b = np.zeros(shape=(1, n - 1))
    matrix_b[0, 0] = -matrix_a[0, 0]
    for i in range(1, n - 1):
        matrix_b[0, i] = matrix_b[0, i - 1] - matrix_a[0, i]
    return matrix_b


def calc(m, k, order):
    order_matrix = set_order(m, order)
    taylor_matrix = get_taylor_coef(m, k)
    matrix_a, error_coef = cal_matrixa(order_matrix, taylor_matrix)
    matrix_b = convert_to_const(matrix_a)
    return (matrix_a, error_coef, matrix_b)


if __name__ == '__main__':
    printinfo()
    m, k, order = get_params()
    matrix_a, error_coef, matrix_b = calc(m, k, order)
    matrix_a, error_coef, matrix_b = calc(m, k, order)
    order_matrix = set_order(m, 1)
    taylor_matrix = get_taylor_coef(m, k)
    matrix_a, error_coef = cal_matrixa(order_matrix, taylor_matrix)
    matrix_b = convert_to_const(matrix_a)
    print(matrix_a)
    print(error_coef)
    print(matrix_b)
