from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np

from utils.is_efficient import is_efficient 
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

class DEA:
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

    def ccr_input_p1(self):
        efficiencies = []
        for o in range(self.n):
            model = Model("InputEfficiency_CCR_P1")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            theta = model.addVar(lb=0, name="theta")
            model.setObjective(theta, GRB.MINIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])
            model.optimize()
            efficiencies.append(model.objVal if model.status == GRB.OPTIMAL else np.nan)
        return efficiencies

    def ccr_input_p2(self, input_efficiencies):
        results_list = [] 
        for o in range(self.n):
            theta_star = input_efficiencies[o]
            if np.isnan(theta_star):
                results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o), 'efficiency': np.nan,
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })
                continue

            model = Model("Phase2_CCR_Input")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])
            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    'DMU': o,
                    'DMU_Name': self._get_dmu_name(o), 
                    'efficiency': theta_star,
                    'slacks_minus': [S_minus[i].x for i in range(self.m)],
                    'slacks_plus': [S_plus[r].x for r in range(self.s)],
                    'Lambda': [lambdas[j].x for j in range(self.n)]
                })
            else: 
                results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o), 'efficiency': theta_star, 
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })
        return pd.DataFrame(results_list)

    def ccr_input(self):
        if "ccr_input" in self.results:
            return self.results['ccr_input']

        results_df = self.ccr_input_p2(self.ccr_input_p1()) 
        results_df = is_efficient(results_df, 'ccr_input')
        self.results['ccr_input'] = results_df
        return results_df

    def ccr_output(self):
        if "ccr_output" in self.results:
            return self.results['ccr_output']

        if "ccr_input" not in self.results:
            self.ccr_input() 

        base_results_df = self.results['ccr_input']
        if base_results_df is None or base_results_df.empty: 

            print("Error: CCR_Input results are not available or invalid for CCR_Output computation.")
            return pd.DataFrame() 

        results_df = base_results_df.copy() 

        results_df['n_val'] = np.nan
        results_df['slacks_minus_transformed'] = None
        results_df['slacks_plus_transformed'] = None
        results_df['Lambda_transformed'] = None

        for i in range(self.n):
            if pd.isna(results_df.loc[i, 'efficiency']) or results_df.loc[i, 'efficiency'] == 0:
                n_val = np.inf 
            else:
                n_val = 1 / results_df.loc[i, 'efficiency']
            results_df.loc[i, 'n_val'] = n_val

            if not pd.isna(n_val):
                results_df.at[i, 'slacks_minus_transformed'] = [val * n_val for val in results_df.loc[i, 'slacks_minus']]
                results_df.at[i, 'slacks_plus_transformed'] = [val * n_val for val in results_df.loc[i, 'slacks_plus']]
                results_df.at[i, 'Lambda_transformed'] = [val * n_val for val in results_df.loc[i, 'Lambda']]
            else: 
                results_df.at[i, 'slacks_minus_transformed'] = [np.nan] * self.m
                results_df.at[i, 'slacks_plus_transformed'] = [np.nan] * self.s
                results_df.at[i, 'Lambda_transformed'] = [np.nan] * self.n

        output_columns = ['DMU', 'DMU_Name']
        results_df_transformed = results_df[output_columns].copy()
        results_df_transformed['n'] = results_df['n_val']
        results_df_transformed['t_minus'] = results_df['slacks_minus_transformed']
        results_df_transformed['t_plus'] = results_df['slacks_plus_transformed']
        results_df_transformed['u'] = results_df['Lambda_transformed']

        results_df_transformed = is_efficient(results_df_transformed, 'ccr_output')
        self.results['ccr_output'] = results_df_transformed
        return results_df_transformed

    def bcc_input_p1(self):
        efficiencies = []
        for o in range(self.n):
            model = Model("InputEfficiency_BCC_P1")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            theta = model.addVar(lb=0, name="theta")
            model.setObjective(theta, GRB.MINIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= theta * self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r])
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()
            efficiencies.append(model.objVal if model.status == GRB.OPTIMAL else np.nan)
        return efficiencies

    def bcc_input_p2(self, input_efficiencies):
        results_list = []
        for o in range(self.n):
            theta_star = input_efficiencies[o]
            if np.isnan(theta_star):
                results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o), 'efficiency': np.nan,
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })
                continue

            model = Model("Phase2_BCC_Input")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == theta_star * self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= self.output_data[o, r])
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    'DMU': o,
                    'DMU_Name': self._get_dmu_name(o), 
                    'efficiency': theta_star,
                    'slacks_minus': [S_minus[i].x for i in range(self.m)],
                    'slacks_plus': [S_plus[r].x for r in range(self.s)],
                    'Lambda': [lambdas[j].x for j in range(self.n)]
                })
            else:
                 results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o), 'efficiency': theta_star,
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })
        return pd.DataFrame(results_list)

    def bcc_input(self):
        if "bcc_input" in self.results:
            return self.results['bcc_input']
        results_df = self.bcc_input_p2(self.bcc_input_p1())
        results_df = is_efficient(results_df, 'bcc_input')
        self.results['bcc_input'] = results_df
        return results_df

    def bcc_output_p1(self):
        efficiencies = []
        for o in range(self.n):
            model = Model("OutputEfficiency_BCC_P1")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambdas_{j}") for j in range(self.n)]
            eta = model.addVar(lb=1, name="eta") 
            model.setObjective(eta, GRB.MAXIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= eta * self.output_data[o, r])
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()
            efficiencies.append(model.objVal if model.status == GRB.OPTIMAL else np.nan)
        return efficiencies

    def bcc_output_p2(self, output_efficiencies):
        results_list = []
        for o in range(self.n):
            eta_star = output_efficiencies[o]
            if np.isnan(eta_star):
                results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o), 'efficiency': np.nan,
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })
                continue

            model = Model("Phase2_BCC_Output")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] >= eta_star * self.output_data[o, r])
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    'DMU': o,
                    'DMU_Name': self._get_dmu_name(o), 
                    'efficiency': eta_star, 
                    'slacks_minus': [S_minus[i].x for i in range(self.m)], 
                    'slacks_plus': [S_plus[r].x for r in range(self.s)], 
                    'Lambda': [lambdas[j].x for j in range(self.n)] 
                })
            else:
                results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o), 'efficiency': eta_star,
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })
        return pd.DataFrame(results_list)

    def bcc_output(self):
        if "bcc_output" in self.results:
            return self.results['bcc_output']
        results_df = self.bcc_output_p2(self.bcc_output_p1())
        results_df.rename(columns={
            'efficiency': 'n', 
            'slacks_minus': 't_minus',
            'slacks_plus':'t_plus',
            'Lambda':'u'}, inplace=True)
        results_df = is_efficient(results_df, 'bcc_output')
        self.results['bcc_output'] = results_df
        return results_df

    def add(self):
        if "add" in self.results:
            return self.results['add']
        results_list = []
        for o in range(self.n):
            model = Model("AdditiveModel")
            model.setParam(GRB.Param.OutputFlag, 0)
            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            model.setObjective(quicksum(S_minus) + quicksum(S_plus), GRB.MAXIMIZE)
            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == self.output_data[o, r])
            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                results_list.append({
                    'DMU': o,
                    'DMU_Name': self._get_dmu_name(o), 
                    'slacks_minus': [S_minus[i].x for i in range(self.m)],
                    'slacks_plus': [S_plus[r].x for r in range(self.s)],
                    'Lambda': [lambdas[j].x for j in range(self.n)]
                })
            else:
                results_list.append({
                    'DMU': o, 'DMU_Name': self._get_dmu_name(o),
                    'slacks_minus': [np.nan]*self.m, 'slacks_plus': [np.nan]*self.s, 'Lambda': [np.nan]*self.n
                })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, 'add')
        self.results['add'] = results_df
        return results_df

    def sbm_non_oriented(self, crs=True): 
        model_type_str = f"sbm_non_oriented_{'crs' if crs else 'vrs'}"
        if model_type_str in self.results:
            return self.results[model_type_str]

        results_list = [] 
        for o in range(self.n):
            model = Model(f"SBM_NonOriented_DMU{o}") 
            model.setParam(GRB.Param.OutputFlag, 0)

            Lambda_vars = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)] 
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            t_var = model.addVar(lb=1e-9, name="t") 

            obj_terms_s_minus = []
            active_inputs_count = 0
            for i in range(self.m):
                if self.input_data[o, i] > 1e-9: 
                    obj_terms_s_minus.append(S_minus[i] / self.input_data[o, i])
                    active_inputs_count += 1
                else: 
                    model.addConstr(S_minus[i] == 0)

            constr_terms_s_plus = []
            active_outputs_count = 0
            for r in range(self.s):
                if self.output_data[o, r] > 1e-9: 
                    constr_terms_s_plus.append(S_plus[r] / self.output_data[o, r])
                    active_outputs_count += 1
                else: 
                     model.addConstr(S_plus[r] == 0)

            if active_inputs_count > 0:
                model.setObjective(t_var - (1/active_inputs_count) * quicksum(obj_terms_s_minus), GRB.MINIMIZE)
            else: 
                model.setObjective(t_var, GRB.MINIMIZE) 

            if active_outputs_count > 0:
                model.addConstr(t_var + (1/active_outputs_count) * quicksum(constr_terms_s_plus) == 1)
            else: 
                model.addConstr(t_var == 1)

            for i in range(self.m):
                model.addConstr(quicksum(Lambda_vars[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == t_var * self.input_data[o, i] )
            for r in range(self.s):
                model.addConstr(quicksum(Lambda_vars[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == t_var * self.output_data[o, r] )

            if not crs:
                model.addConstr(quicksum(Lambda_vars) == t_var)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                rho_star = model.objVal
                t_star_val = t_var.x
                if abs(t_star_val) < 1e-9 : t_star_val = 1e-9 

                lambda_star = [Lambda_vars[j].x / t_star_val for j in range(self.n)]
                s_minus_star_orig = [S_minus[i].x / t_star_val for i in range(self.m)] 
                s_plus_star_orig = [S_plus[r].x / t_star_val for r in range(self.s)]   
            else:
                rho_star, t_star_val = np.nan, np.nan
                lambda_star = [np.nan]*self.n
                s_minus_star_orig = [np.nan]*self.m
                s_plus_star_orig = [np.nan]*self.s

            results_list.append({
                'DMU': o, 'DMU_Name': self._get_dmu_name(o), 
                'rho': rho_star, 'lambda': lambda_star, 
                's_minus': s_minus_star_orig, 's_plus': s_plus_star_orig, 
                'S_minus_transformed': [S_minus[i].x if model.status == GRB.OPTIMAL else np.nan for i in range(self.m)], 
                'S_plus_transformed': [S_plus[r].x if model.status == GRB.OPTIMAL else np.nan for r in range(self.s)],   
                't_star': t_star_val
            })
        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, 'sbm_non_oriented') 
        self.results[model_type_str] = results_df
        return results_df

    def sbm_input(self, crs=True):
        model_type_str = f"sbm_input_{'crs' if crs else 'vrs'}"
        if model_type_str in self.results:
            return self.results[model_type_str]

        results_list = []
        for o in range(self.n):
            model = Model(f"SBM_Input_DMU{o}")
            model.setParam(GRB.Param.OutputFlag, 0)
            Lambda_vars = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]

            S_plus = [model.addVar(lb=0, ub=0, name=f"S_plus_{r}") for r in range(self.s)] 
            t_var = model.addVar(lb=1e-9, name="t")

            obj_terms_s_minus = []
            active_inputs_count = 0
            for i in range(self.m):
                if self.input_data[o, i] > 1e-9:
                    obj_terms_s_minus.append(S_minus[i] / self.input_data[o, i])
                    active_inputs_count += 1
                else:
                    model.addConstr(S_minus[i] == 0) 

            if active_inputs_count > 0:
                model.setObjective(t_var - (1 / active_inputs_count) * quicksum(obj_terms_s_minus), GRB.MINIMIZE)
            else: 
                model.setObjective(t_var, GRB.MINIMIZE) 

            model.addConstr(t_var == 1)

            for i in range(self.m):
                model.addConstr(quicksum(Lambda_vars[j] * self.input_data[j, i] for j in range(self.n)) + S_minus[i] == t_var * self.input_data[o, i])
            for r in range(self.s):
                model.addConstr(quicksum(Lambda_vars[j] * self.output_data[j, r] for j in range(self.n)) >= t_var * self.output_data[o, r])

            if not crs: 
                model.addConstr(quicksum(Lambda_vars) == t_var)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                rho_star = model.objVal 
                t_star_val = t_var.x 
                lambda_star = [Lambda_vars[j].x / t_star_val for j in range(self.n)]
                s_minus_star_orig = [S_minus[i].x / t_star_val for i in range(self.m)]
                s_plus_star_orig = [0.0 for _ in range(self.s)] 
            else:
                rho_star, t_star_val = np.nan, np.nan
                lambda_star = [np.nan] * self.n
                s_minus_star_orig = [np.nan] * self.m
                s_plus_star_orig = [np.nan] * self.s

            results_list.append({
                'DMU': o, 'DMU_Name': self._get_dmu_name(o),
                'rho': rho_star, 'lambda': lambda_star,
                's_minus': s_minus_star_orig, 's_plus': s_plus_star_orig,
                'S_minus_transformed': [S_minus[i].x if model.status == GRB.OPTIMAL else np.nan for i in range(self.m)],
                'S_plus_transformed': [S_plus[r].x if model.status == GRB.OPTIMAL else np.nan for r in range(self.s)], 
                't_star': t_star_val
            })
        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, 'sbm_input')
        self.results[model_type_str] = results_df
        return results_df

    def sbm_output(self, crs=True):
        model_type_str = f"sbm_output_{'crs' if crs else 'vrs'}"
        if model_type_str in self.results:
            return self.results[model_type_str]

        results_list = []
        for o in range(self.n):
            model = Model(f"SBM_Output_DMU{o}")
            model.setParam(GRB.Param.OutputFlag, 0)
            Lambda_vars = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)]

            S_minus = [model.addVar(lb=0, ub=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            t_var = model.addVar(lb=1e-9, name="t")

            model.setObjective(t_var, GRB.MINIMIZE)

            constr_terms_s_plus = []
            active_outputs_count = 0
            for r in range(self.s):
                if self.output_data[o, r] > 1e-9:
                    constr_terms_s_plus.append(S_plus[r] / self.output_data[o, r])
                    active_outputs_count += 1
                else: 
                    model.addConstr(S_plus[r] == 0)

            if active_outputs_count > 0:
                model.addConstr(t_var + (1 / active_outputs_count) * quicksum(constr_terms_s_plus) == 1)
            else:
                model.addConstr(t_var == 1)

            for i in range(self.m):

                model.addConstr(quicksum(Lambda_vars[j] * self.input_data[j, i] for j in range(self.n)) <= t_var * self.input_data[o, i])

            for r in range(self.s):
                model.addConstr(quicksum(Lambda_vars[j] * self.output_data[j, r] for j in range(self.n)) - S_plus[r] == t_var * self.output_data[o, r])

            if not crs: 
                model.addConstr(quicksum(Lambda_vars) == t_var)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                rho_star = model.objVal 
                t_star_val = t_var.x 

                lambda_star = [Lambda_vars[j].x / t_star_val for j in range(self.n)]
                s_minus_star_orig = [0.0 for _ in range(self.m)] 
                s_plus_star_orig = [S_plus[r].x / t_star_val for r in range(self.s)]
            else:
                rho_star, t_star_val = np.nan, np.nan
                lambda_star = [np.nan] * self.n
                s_minus_star_orig = [np.nan] * self.m
                s_plus_star_orig = [np.nan] * self.s

            results_list.append({
                'DMU': o, 'DMU_Name': self._get_dmu_name(o),
                'rho': rho_star, 'lambda': lambda_star,
                's_minus': s_minus_star_orig, 's_plus': s_plus_star_orig,
                'S_minus_transformed': [S_minus[i].x if model.status == GRB.OPTIMAL else np.nan for i in range(self.m)], 
                'S_plus_transformed': [S_plus[r].x if model.status == GRB.OPTIMAL else np.nan for r in range(self.s)],
                't_star': t_star_val
            })
        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, 'sbm_output')
        self.results[model_type_str] = results_df
        return results_df

    def modified_sbm(self):
        if "modified_sbm" in self.results: 
            return self.results['modified_sbm']

        results_list = []

        min_inputs = np.min(self.input_data, axis=0)
        max_outputs = np.max(self.output_data, axis=0)

        P_minus_all = self.input_data - min_inputs
        P_plus_all = max_outputs - self.output_data

        for o in range(self.n): 
            model = Model("ModifiedSBM_Model")
            model.setParam(GRB.Param.OutputFlag, 0)

            Lambda_vars = [model.addVar(lb=0, name=f"Lambda_{j}") for j in range(self.n)]
            S_minus = [model.addVar(lb=0, name=f"S_minus_{i}") for i in range(self.m)]
            S_plus = [model.addVar(lb=0, name=f"S_plus_{r}") for r in range(self.s)]
            t_var = model.addVar(lb=1e-6, name="t") 

            P_minus_o = P_minus_all[o, :]
            P_plus_o = P_plus_all[o, :]

            objective_terms = []
            num_active_input_ranges = 0
            for i in range(self.m):
                if P_minus_o[i] > 1e-9: 
                    objective_terms.append(S_minus[i] / P_minus_o[i])
                    num_active_input_ranges +=1
                else: 
                    model.addConstr(S_minus[i] == 0)

            if num_active_input_ranges > 0:
                 model.setObjective(t_var - quicksum(objective_terms) / num_active_input_ranges, GRB.MINIMIZE)
            else: 
                 model.setObjective(t_var, GRB.MINIMIZE)

            constraint_terms = []
            num_active_output_ranges = 0
            for r in range(self.s):
                if P_plus_o[r] > 1e-9: 
                    constraint_terms.append(S_plus[r] / P_plus_o[r])
                    num_active_output_ranges +=1
                else: 
                    model.addConstr(S_plus[r] == 0)

            if num_active_output_ranges > 0:
                model.addConstr(t_var + quicksum(constraint_terms) / num_active_output_ranges == 1)
            else: 
                model.addConstr(t_var == 1)

            for i in range(self.m):

                model.addConstr(quicksum(Lambda_vars[j] * P_minus_all[j, i] for j in range(self.n)) == \
                                t_var * P_minus_o[i] - S_minus[i])

            for r in range(self.s):

                model.addConstr(quicksum(Lambda_vars[j] * P_plus_all[j, r] for j in range(self.n)) == \
                                t_var * P_plus_o[r] + S_plus[r])

            model.addConstr(quicksum(Lambda_vars[j] for j in range(self.n)) == t_var)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                rho_star = model.objVal
                t_star = t_var.x

                if abs(t_star) < 1e-9 : t_star = 1e-9 

                lambda_star = [Lambda_vars[j].x / t_star for j in range(self.n)]
                s_minus_star = [S_minus[i].x / t_star for i in range(self.m)]
                s_plus_star = [S_plus[r].x / t_star for r in range(self.s)]
            else:
                rho_star, t_star = np.nan, np.nan
                lambda_star, s_minus_star, s_plus_star = [np.nan]*self.n, [np.nan]*self.m, [np.nan]*self.s

            results_list.append({
                'DMU': o, 'DMU_Name': self._get_dmu_name(o),
                'rho': rho_star, 'lambda': lambda_star,
                's_minus': s_minus_star, 's_plus': s_plus_star,
                't_star': t_star 
            })

        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, 'modified_sbm') 
        self.results['modified_sbm'] = results_df
        return results_df

    def rdm(self):
        if "rdm" in self.results:
            return self.results['rdm']
        results_list = [] 

        min_inputs_overall = np.min(self.input_data, axis=0)
        max_outputs_overall = np.max(self.output_data, axis=0)

        for o in range(self.n):
            model = Model("RDM_Model")
            model.setParam(GRB.Param.OutputFlag, 0)

            R_minus_o = self.input_data[o, :] - min_inputs_overall
            R_plus_o = max_outputs_overall - self.output_data[o, :]

            lambdas = [model.addVar(lb=0, name=f"lambda_{j}") for j in range(self.n)]
            beta = model.addVar(lb=0, ub=1, name="beta") 

            model.setObjective(1 - beta, GRB.MINIMIZE)

            for i in range(self.m):
                model.addConstr(quicksum(lambdas[j] * self.input_data[j, i] for j in range(self.n)) <= self.input_data[o, i] - beta * R_minus_o[i])
            for r in range(self.s):
                model.addConstr(quicksum(lambdas[j] * self.output_data[j, r] for j in range(self.n)) >= self.output_data[o, r] + beta * R_plus_o[r])

            model.addConstr(quicksum(lambdas[j] for j in range(self.n)) == 1)
            model.optimize()

            efficiency_val = np.nan
            if model.status == GRB.OPTIMAL:
                efficiency_val = model.ObjVal 
                if abs(efficiency_val) < 1e-9: efficiency_val = 0.0
                if abs(efficiency_val - 1.0) < 1e-9: efficiency_val = 1.0

            results_list.append({
                "DMU": o, "DMU_Name": self._get_dmu_name(o), 
                "efficiency": efficiency_val
            })
        results_df = pd.DataFrame(results_list)
        results_df = is_efficient(results_df, 'rdm')
        self.results['rdm'] = results_df
        return results_df

    def plot2d(self, typ):
        if typ in self.results:
            print(f"Results for model '{typ}' are already computed. Skipping plot.")

        if self.input_data.shape[1] != 1 or self.output_data.shape[1] != 1:
            raise ValueError("Unsupported combination of M and S")

        fig, ax = plt.subplots(figsize=(8, 6))
        x = self.input_data.flatten()
        y = self.output_data.flatten()
        ax.scatter(x, y, c='blue', label='DMUs')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        title = 'Input vs Output'
        fake_origin = (np.min(x), np.max(y))
        max_x, max_y = np.max(x), np.max(y)

        typ2func = {
            'ccr_input': self.ccr_input,  
            'ccr_output': self.ccr_output,
            'sbm_non_oriented': self.sbm_non_oriented,
            'sbm_non_oriented_vrs': self.sbm_non_oriented(crs=False),
            'sbm_non_oriented_crs': self.sbm_non_oriented(crs=True),
            'add': self.add,
            'bcc_input': self.bcc_input,
            'bcc_output': self.bcc_output,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            results = typ2func[typ]()  
        else:
            print(f"Using cached results for {typ}...")
            results = self.results[typ]  

        efficient_points = []

        for i, efficient in enumerate(results['is_efficient']):
            dmu_name = self._get_dmu_name(i)
            ax.text(x[i], y[i], dmu_name, color='gray', fontsize=9, ha='right', va='bottom')
            if efficient:
                efficient_points.append((x[i], y[i]))
                circle = plt.Circle((x[i], y[i]), radius=0.05, fill=False, ec='red', lw=1.5)
                ax.add_patch(circle)

        if len(efficient_points) > 2:
            hull = ConvexHull(efficient_points)
            edges = []
            for simplex in hull.simplices:
                edge = (efficient_points[simplex[0]], efficient_points[simplex[1]])
                dist = self.perpendicular_distance(fake_origin, edge[0], edge[1])
                edges.append((edge, dist))
            edges.sort(key=lambda x: x[1])
            selected_edges = edges[:len(efficient_points) - 1]
            for edge, _ in selected_edges:
                ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'r-', linewidth=2)
        elif len(efficient_points) == 2:
            ax.plot([efficient_points[0][0], efficient_points[1][0]],
                    [efficient_points[0][1], efficient_points[1][1]], 'r-', linewidth=2)
        elif len(efficient_points) == 1:
            extended_point = self.extend_line((0, 0), efficient_points[0], max_x)
            ax.plot([0, extended_point[0]], [0, extended_point[1]], 'r-', linewidth=2)

        ax.set_xlim(0, max_x * 1.1)
        ax.set_ylim(0, max_y * 1.1)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        # plt.show()
        return fig  # Return the figure instead of showing it

    def plot3d(self, typ):
        fig = plt.figure()

        N, M = self.input_data.shape
        N, S = self.output_data.shape

        if M == 1 and S == 1:
            raise ValueError("Exactally 3 axes required")

        elif M == 2 and S == 1:
            ax = fig.add_subplot(111, projection='3d')
            x = self.input_data[:, 0]
            y = self.input_data[:, 1]
            z = self.output_data.flatten()
            ax.scatter(x, y, z, c='green')
            ax.set_xlabel('Input1')
            ax.set_ylabel('Input2')
            ax.set_zlabel('Output')
            title = 'Input1 vs Input2 vs Output'

        elif M == 1 and S == 2:
            ax = fig.add_subplot(111, projection='3d')
            x = self.input_data.flatten()
            y = self.output_data[:, 0]
            z = self.output_data[:, 1]
            ax.scatter(x, y, z, c='red')
            ax.set_xlabel('Input')
            ax.set_ylabel('Output1')
            ax.set_zlabel('Output2')
            title = 'Input vs Output1 vs Output2'

        else:
            raise ValueError("Unsupported combination of M and S")

        typ2func = {
            'ccr_input': self.ccr_input,  
            'ccr_output': self.ccr_output,
            'sbm_non_oriented_vrs': self.sbm_non_oriented(crs=False),
            'sbm_non_oriented_crs': self.sbm_non_oriented(crs=True),
            'add': self.add,
            'bcc_input': self.bcc_input,
            'bcc_output': self.bcc_output,
        }

        if typ not in typ2func:
            print(f"Unsupported type: {typ}")
            return

        if typ not in self.results:
            print(f"Calculating results for {typ}...")
            res = typ2func[typ]()  
            self.results[typ] = res  
        else:
            print(f"Using cached results for {typ}...")
            res = self.results[typ]  

        efficiency_data = res  
        if 'is_efficient' in efficiency_data.columns:
            is_efficient = efficiency_data['is_efficient'].values
        else:
            print(f"Warning: 'is_efficient' column missing in results for {typ}")
            return

        for i, efficient in enumerate(is_efficient):
            if efficient:
                ax.scatter(x[i], y[i], z[i], s=200, facecolors='none', edgecolors='r', linewidths=2)
                dmu_name = self._get_dmu_name(i)
                ax.text(x[i], y[i], z[i] + 0.02, dmu_name, color='gray', fontsize=10, ha='center')

        # for i, efficient in enumerate(is_efficient):
        #     if efficient:
        #         ax.scatter(x[i], y[i], z[i], s=200, facecolors='none', edgecolors='r', linewidths=2)
        #         ax.text(x[i], y[i], z[i] + 0.02, typ, color='black', fontsize=10, ha='center')

        ax.set_title(title)
        ax.grid(True)
        # plt.show()
        return fig  # Return the figure instead of showing it




#####################################################################################


    def perpendicular_distance(self, point, line_point1, line_point2):
        x0, y0 = point
        x1, y1 = line_point1
        x2, y2 = line_point2
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    def extend_line(self, p1, p2, max_x):
        if p2[0] == p1[0]:  
            return (p2[0], max_x * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1])
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            return (max_x, slope * (max_x - p1[0]) + p1[1])
        
