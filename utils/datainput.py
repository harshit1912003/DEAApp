import numpy as np
import pandas as pd
import pickle
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


def arr2matrix(x, y):
    input_data = np.array(x).T
    output_data = np.array(y).T
    return input_data, output_data


def load_results():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    if file_path:  
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        print(f"Results loaded from {file_path}")
        return results
    else:
        print("Load operation canceled.")
        return None
    

def xlsx2matrix(file_path, x, y):
    df = pd.read_excel(file_path)

    input_data = df[x].to_numpy()
    output_data = df[y].to_numpy()

    input_data[np.isnan(input_data)] = 0
    output_data[np.isnan(output_data)] = 0

    return input_data, output_data


def csv2matrix(file_path, x, y):
    df = pd.read_csv(file_path)

    input_data = df[x].to_numpy()
    output_data = df[y].to_numpy()

    input_data[np.isnan(input_data)] = 0
    output_data[np.isnan(output_data)] = 0

    return input_data, output_data



def initializeUnif(N, M, S, a, b):    
    input_array = np.random.uniform(a, b, size=(N, M))
    output_array = np.random.uniform(a, b, size=(N, S))
    return input_array, output_array


def initialize_sparse(N, M, S, a, b, density):
    if not (0 < density <= 1):
        raise ValueError("Density must be between 0 and 1.")

    input_array = np.random.uniform(a, b, size=(N, M))

    output_array = np.random.uniform(a, b, size=(N, S))
    mask = np.random.rand(N, S) < density
    output_array = output_array * mask
    plt.figure(figsize=(8, 6))
    plt.imshow(output_array == 0, cmap="Grays", interpolation="none", aspect='auto')
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()
    
    return input_array, output_array

