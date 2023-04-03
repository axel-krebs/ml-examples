# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:25:28 2023

@author: akrebs5
"""
from brain import *


def print_menu():
    print("0. Leave the program.")
    print("1. Run tests")
    print("2. Run sample for AND (")


def menu():

    print("Perceptron")

    while True:

        print_menu()

        user_inp = input(">")
        if user_inp == "0":
            break

        elif user_inp == "1":
            test()

        elif user_inp == "2":
            print("Running sample..")


menu()
