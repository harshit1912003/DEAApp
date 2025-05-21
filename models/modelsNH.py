from gurobipy import Model, GRB, quicksum, LinExpr
import pandas as pd
import numpy as np
from utils.is_efficient import is_efficient
from .modelsDEA import DEA
from utils.groupDMUs import groupDMUs

class Non_Homo:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.n, self.m = input_data.shape
        self.s = output_data.shape[1]
        self.results = []
        self.N, self.R, self.dmu_outputs, self.correspondingR, self.L, self.dmu_to_subgroup, self.M = groupDMUs(self.output_data)
        self.alphas = []

    def no2mat(self, a, b):
        A = np.full((self.m, len(self.R), len(self.N)), a)
        B = np.full((self.m, len(self.R), len(self.N)), b)
        return A, B

    def compute_alphas(self, a, b):
        if self.alphas:
            return self.alphas

        alphas = []

        for knot in range(self.n):
            model = Model("non_homo_crs")
            gamma = {}

            for i in range(self.m):
                for k in range(len(self.R)):  
                    for p in range(len(self.N)):  
                        gamma[(i, k, p)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)

            for i in range(self.m):
                for np_j in range(len(self.N)):
                    L_np = self.L[np_j]
                    for k in range(len(self.R)):
                        if k not in L_np:
                            model.addConstr(gamma[(i, k, np_j)] == 0)

            for i in range(self.m):
                for np_j in range(len(self.N)):
                    L_np = self.L[np_j]
                    for k in range(len(self.R)):
                        if k in L_np:
                            model.addConstr(gamma[(i, k, np_j)] >= 1e-6)

            model.setParam(GRB.Param.OutputFlag, 0)

            mu = [model.addVar(lb=1e-6, name=f"mu_{r}") for r in range(self.s)]
            v = [model.addVar(lb=1e-6, name=f"v_{i}") for i in range(self.m)]

            npo = self.dmu_to_subgroup[knot]
            L_npo = self.L[npo]

            eo = LinExpr()
            for k in L_npo:
                for r in list(self.R[k]):
                    eo += mu[r] * self.output_data[knot, r]
            model.setObjective(eo, GRB.MAXIMIZE)

            norm_expr = LinExpr()
            for k in L_npo:
                for i in range(self.m):
                    norm_expr += gamma[(i, k, npo)] * self.input_data[knot, i]
            model.addConstr(norm_expr == 1, name="Normalization")

            for j in range(self.n):
                np_j = self.dmu_to_subgroup[j]
                L_np = list(self.L[np_j])

                for k in L_np:
                    output_expr = LinExpr()
                    input_expr = LinExpr()

                    for r in list(self.R[k]):
                        output_expr += mu[r] * self.output_data[j, r]

                    for i in range(self.m):
                        input_expr += gamma[(i, k, np_j)] * self.input_data[j, i]

                    model.addConstr(output_expr - input_expr <= 0, name=f"Constraint_Rk_{k}")

            for i in range(self.m):
                for np_j in range(len(self.N)):
                    L_np = list(self.L[np_j])
                    bound_inp = LinExpr()
                    for k in L_np:
                        bound_inp += gamma[(i, k, np_j)]
                    model.addConstr(bound_inp == v[i])

            for i in range(self.m):
                for np_j in range(len(self.N)):
                    for k in list(self.L[np_j]):
                        model.addConstr(a[i, k, np_j] * v[i] <= gamma[(i, k, np_j)])
                        model.addConstr(gamma[(i, k, np_j)] <= v[i] * b[i, k, np_j])

            model.optimize()

            if model.status == GRB.OPTIMAL:
                print("Optimal solution found.")
            elif model.status == GRB.INFEASIBLE:
                print("Model is infeasible.")
            elif model.status == GRB.UNBOUNDED:
                print("Model is unbounded.")
            elif model.status == GRB.TIME_LIMIT:
                print("Time limit reached.")
            elif model.status == GRB.INTERRUPTED:
                print("Optimization interrupted.")
            elif model.status == GRB.SUBOPTIMAL:
                print("Suboptimal solution found.")
            else:
                print("Optimization ended with an unknown status.")

            alphas.append({
                "DMU": knot,
                "alphas": {key: (gamma[key].x) / (v[key[0]].x) for key in gamma if key[2] == self.dmu_to_subgroup[knot]}
            })

            self.alphas = alphas
        return alphas

    def new_input_data(self):
        if self.alphas:
            alphas = self.alphas
        else:
            raise Exception("compute alphas first")

        new_input_data = np.zeros((self.n, self.m, len(self.R)))
        for j in range(self.n):
            alpha_values = alphas[j]['alphas'].items()

            for (i, k, p) , alpha in alpha_values:
                new_input_data[j, i, k] = alpha * self.input_data[j, i]
        return new_input_data

    def average_efficiency(self, new_input_data):
        if self.results:
            return self.results[0]
        
        efficiencyvals = {i:0 for i in range(self.input_data.shape[0])}
        rep = {i : 0 for i in range(self.input_data.shape[0])}

        for k in range(len(self.R)):
            for dmu in list(self.M[k]):
                rep[dmu] += 1

            x_dash = new_input_data[list(self.M[k]) , : , k]
            y_dash = self.output_data[list(self.M[k]), :] 
            y_dash = y_dash[:, list(self.R[k])]

            dea = DEA(x_dash, y_dash)

            ccr_res = dea.ccr_input()[["DMU", 'efficiency']]

            reindexed_dmus = {idx: dmu for idx, dmu in enumerate(self.M[k])}

            for idx in ccr_res["DMU"]:
                original_dmu = reindexed_dmus[idx]
                efficiency_score = ccr_res.loc[ccr_res["DMU"] == idx, "efficiency"]
                if not efficiency_score.empty:
                    efficiencyvals[original_dmu] += efficiency_score.iloc[0]

        average_efficiencies = {dmu: efficiencyvals[dmu] / rep[dmu] for dmu in efficiencyvals}
        self.results.append(average_efficiencies)
        return average_efficiencies
    