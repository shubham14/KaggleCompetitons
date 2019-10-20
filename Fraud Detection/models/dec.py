# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:46:12 2019

@author: Shubham
"""

def our_decorator(func):
    def func_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return func_wrapper

@our_decorator
def foo(x):
    print("Hi howdy {}".format(x))


def arg_test_num(f):
    def helper(x):
        if type(x) == int and x > 0:
            return f(x)
        else:
            raise Exception("Argument is not an integer")
    return helper

@argument_test_natural_number
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)
    