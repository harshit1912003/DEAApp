from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
from utils.is_efficient import is_efficient 
import pickle
import matplotlib.pyplot as plt

class FDH:
    def __init__(self, input_data, output_data, dmu_names=None, input_feature_names=None, output_feature_names=None):
        self.input_data = np.asarray(input_data)
        self.output_data = np.asarray(output_data)

        if self.input_data.ndim == 1:
            self.input_data = self.input_data.reshape(-1, 1)
        if self.output_data.ndim == 1:
            self.output_data = self.output_data.reshape(-1, 1)

        self.n, self.m = self.input_data.shape
        if self.output_data.shape[0] != self.n:
            raise ValueError("Input and output data must have the same number of DMUs (rows).")
        self.s = self.output_data.shape[1]
        self.results = {}

        self.set_dmu_names(dmu_names)
        self.set_input_feature_names(input_feature_names)
        self.set_output_feature_names(output_feature_names)

    def set_dmu_names(self, names_list):
        if names_list is None:
            self.dmu_names = [f"DMU_{i+1}" for i in range(self.n)] 
        elif len(names_list) == self.n:
            self.dmu_names = list(names_list)
        else:
            raise ValueError(f"DMU names list length must be {self.n}, but got {len(names_list)}")

    def _get_dmu_name(self, index):
        return self.dmu_names[index]

    def set_input_feature_names(self, names_list):
        if names_list is None:
            self.input_feature_names = [f"Input_{i+1}" for i in range(self.m)]
        elif len(names_list) == self.m:
            self.input_feature_names = list(names_list)
        else:
            raise ValueError(f"Input feature names list length must be {self.m}, but got {len(names_list)}")

    def set_output_feature_names(self, names_list):
        if names_list is None:
            self.output_feature_names = [f"Output_{i+1}" for i in range(self.s)]
        elif len(names_list) == self.s:
            self.output_feature_names = list(names_list)
        else:
            raise ValueError(f"Output feature names list length must be {self.s}, but got {len(names_list)}")

    def save_results(self, file_path):
        if not file_path:
            print("Save operation canceled: No file path provided.")
            return
        if not file_path.lower().endswith(".pkl"):
            file_path += ".pkl"

        with open(file_path, "wb") as file:
            pickle.dump(self.results, file)
        print(f"Results saved to {file_path}")

    def fdh_input_crs(self):
        model_name = "fdh_input_crs"
        if model_name in self.results:
            return self.results[model_name]

        results_list = []

        base_model_gurobi = Model("FDH_Input_CRS_Base")
        base_model_gurobi.setParam(GRB.Param.OutputFlag, 0)

        for o in range(self.n):
            model = base_model_gurobi.copy() 

            z = model.addVars(self.n, vtype=GRB.BINARY, name="z")
            theta = model.addVar(lb=0, ub=1.0, name="theta") 
            delta = model.addVar(lb=0, name='delta')

            model.setObjective(theta, GRB.MINIMIZE)

            for i in range(self.m):
                model.addConstr(
                    delta * quicksum(z[j] * self.input_data[j, i] for j in range(self.n))
                    <= theta * self.input_data[o, i]
                )

            for r in range(self.s):
                model.addConstr(
                    quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n))
                    >= self.output_data[o, r]
                )

            model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": model.objVal
                })
            else:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": np.nan
                })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, model_name) 

        self.results[model_name] = results_df
        return results_df

    def fdh_output_crs(self):
        model_name = "fdh_output_crs"
        if model_name in self.results:
            return self.results[model_name]

        results_list = []

        for o in range(self.n):
            model = Model(f"FDH_Output_CRS_DMU_{o}")
            model.setParam(GRB.Param.OutputFlag, 0)

            z = [model.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(self.n)]
            eta = model.addVar(lb=1, name="eta") 
            delta = model.addVar(lb=0, name='delta')

            model.setObjective(eta, GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(delta * z[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(delta * z[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])

            model.addConstr(quicksum(z[j] for j in range(self.n)) == 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": model.objVal
                })
            else:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": np.nan
                })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, model_name)
        self.results[model_name] = results_df
        return results_df

    def fdh_input_vrs(self):
        model_name = "fdh_input_vrs"
        if model_name in self.results:
            return self.results[model_name]

        results_list = []
        for o in range(self.n):
            model = Model(f"FDH_Input_VRS_DMU_{o}")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
            theta = model.addVar(lb=0, ub=1.0, name="theta") 
            model.setObjective(theta, GRB.MINIMIZE)

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])

            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": model.objVal
                })
            else:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": np.nan
                })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, model_name)
        self.results[model_name] = results_df
        return results_df

    def fdh_output_vrs(self):
        model_name = "fdh_output_vrs"
        if model_name in self.results:
            return self.results[model_name]

        results_list = []
        for o in range(self.n):
            model = Model(f"FDH_Output_VRS_DMU_{o}")
            model.setParam(GRB.Param.OutputFlag, 0)

            lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
            eta = model.addVar(lb=1, name="eta") 

            model.setObjective(eta, GRB.MAXIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": model.objVal
                })
            else:
                results_list.append({
                    "DMU": o,
                    "DMU_Name": self._get_dmu_name(o),
                    "efficiency": np.nan
                })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, model_name)
        self.results[model_name] = results_df
        return results_df

    def plot_fdh(self, typ):  

        if self.input_data.shape[1] != 1 or self.output_data.shape[1] != 1:
            raise ValueError("Unsupported combination of M and S")

        x = self.input_data.flatten()
        y = self.output_data.flatten()
        dmu_names = [self._get_dmu_name(i) for i in range(len(x))]

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.set_title('Input vs Output (FDH)')

        ax.scatter(x, y, c='blue', edgecolor='k', alpha=0.7)

        typ2func = {
            'fdh_input_vrs': self.fdh_input_vrs,
            'fdh_output_vrs': self.fdh_output_vrs,
        }
        if typ not in typ2func:
            raise ValueError(f"Unsupported type: {typ}")
        if typ not in self.results:
            self.results[typ] = typ2func[typ]()
        results = self.results[typ]

        efficient_idx = np.where(results['is_efficient'])[0]
        for i in efficient_idx:

            circ = plt.Circle((x[i], y[i]), radius=0.01 * max(x.max(), y.max()),
                              fill=False, edgecolor='red', linewidth=2)
            ax.add_artist(circ)

        for i, (xi, yi, name) in enumerate(zip(x, y, dmu_names)):
            ax.text(xi, yi, name, fontsize=8, color='gray',
                    ha='right', va='bottom')

        eff_points = list(zip(x[efficient_idx], y[efficient_idx]))
        if len(eff_points) > 1:

            eff_points = sorted(eff_points, key=lambda p: (p[0], -p[1]))
            xs, ys = zip(*eff_points)

            for j in range(len(xs) - 1):
                ax.hlines(ys[j], xs[j], xs[j+1], colors='green', linewidth=1)
                ax.vlines(xs[j+1], ys[j+1], ys[j], colors='green', linewidth=1)

            ax.hlines(ys[-1], xs[-1], x.max() * 1.1, colors='green', linewidth=1)
            ax.vlines(xs[0], 0, ys[0], colors='green', linewidth=1)

        ax.set_xlim(0, x.max() * 1.1)
        ax.set_ylim(0, y.max() * 1.1)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        # plt.show()
        return fig  # Return the figure instead of showing it


    def rdm_fdh(self):
        model_name = "rdm_fdh"
        if model_name in self.results:
            return self.results[model_name]

        results_list = []

        min_inputs_overall = np.min(self.input_data, axis=0)
        max_outputs_overall = np.max(self.output_data, axis=0)

        for o in range(self.n):
            model = Model(f"RDM_FDH_DMU_{o}")
            model.setParam(GRB.Param.OutputFlag, 0)

            R_minus_o = self.input_data[o, :] - min_inputs_overall

            R_plus_o = max_outputs_overall - self.output_data[o, :]

            lambdas = [model.addVar(vtype=GRB.BINARY, name=f"lambda_{j}") for j in range(self.n)]
            beta = model.addVar(lb=0, ub=1, name="beta") 

            model.setObjective(1 - beta, GRB.MINIMIZE) 

            for i in range(self.m):

                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) 
                                <= self.input_data[o, i] - beta * R_minus_o[i])

            for r in range(self.s):

                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) 
                                >= self.output_data[o, r] + beta * R_plus_o[r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            efficiency_val = np.nan
            if model.status == GRB.OPTIMAL:
                efficiency_val = model.ObjVal 

                if abs(efficiency_val) < 1e-9: 
                    efficiency_val = 0.0
                elif abs(efficiency_val - 1.0) < 1e-9: 
                    efficiency_val = 1.0

            results_list.append({
                "DMU": o,
                "DMU_Name": self._get_dmu_name(o),
                "efficiency": efficiency_val 
            })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, model_name) 
        self.results[model_name] = results_df
        return results_df

    def predict(self, x, model_name):
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} has not been computed yet.")
        x = np.array(x)
        model_results = self.results[model_name]

        valid_dmus = []
        for i in range(self.n):
            if model_results.loc[i, 'is_efficient'] and np.all(self.input_data[i] <= x):
                valid_dmus.append(i)

        if not valid_dmus:
            return 0

        max_output = np.max(self.output_data[valid_dmus], axis=0)

        return max_output.item()