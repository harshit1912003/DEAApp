import streamlit as st
import pandas as pd
import numpy as np
from models.modelsDEA import DEA
from models.modelsFDH import FDH
from utils.datainput import xlsx2matrix, csv2matrix
import matplotlib.pyplot as plt
import io

import eat 
import graphviz 

@st.cache_data
def load_data(uploaded_file, input_cols, output_cols):
    """Loads data from uploaded file using selected columns."""
    try:
        file_buffer = io.BytesIO(uploaded_file.getvalue())
        file_name = uploaded_file.name
        df_full = None

        if file_name.endswith('.xlsx'):
            input_data, output_data = xlsx2matrix(file_buffer, input_cols, output_cols)
            file_buffer.seek(0)
            df_full = pd.read_excel(file_buffer)
        elif file_name.endswith('.csv'):
            input_data, output_data = csv2matrix(file_buffer, input_cols, output_cols)
            file_buffer.seek(0)
            df_full = pd.read_csv(file_buffer)
        else:
            return None, None, None, "Unsupported file type. Please upload .xlsx or .csv"

        return input_data, output_data, df_full, None
    except Exception as e:
        return None, None, None, f"Error processing file: {e}"

@st.cache_resource
def get_model_instance(model_type, _input_data, _output_data):
    """Gets or creates a model instance for DEA/FDH."""

    input_copy = np.copy(_input_data)
    output_copy = np.copy(_output_data)
    if model_type == "DEA":
        return DEA(input_copy, output_copy)
    elif model_type == "FDH":
        return FDH(input_copy, output_copy)
    elif model_type == "EAT":
        return None 
    else:
        return None

st.set_page_config(layout="wide")
st.title("📊 Efficiency Analysis Toolkit (DEA/FDH/EAT)")

default_session_state = {
    'data_loaded': False, 'input_data': None, 'output_data': None, 'df_full': None,
    'input_columns': [], 'output_columns': [], 'model_results': None,
    'model_type': "DEA", 'model_function_name': None, 'model_instance': None,
    'sbm_crs_checkbox': True, 
    'eat_model_instance': None,
    'eat_dot_data': None,
    'eat_predictions': None,
    'eat_num_stop': 5,
    'eat_fold': 5
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Excel File (.xlsx) or CSV File (.csv)",
        type=["xlsx", "csv"]
    )

    if uploaded_file:
        try:
            file_buffer_header = io.BytesIO(uploaded_file.getvalue())
            file_name = uploaded_file.name
            if file_name.endswith('.xlsx'):
                df_sample = pd.read_excel(file_buffer_header, nrows=0)
            elif file_name.endswith('.csv'):
                df_sample = pd.read_csv(file_buffer_header, nrows=0)
            else:
                 st.error("Unsupported file type for header reading.")
                 st.stop()

            column_headers = df_sample.columns.tolist()

            st.header("2. Select Columns")
            input_columns = st.multiselect(
                "Input Columns (X)",
                options=column_headers,
                default=st.session_state.get('input_columns', []),
                key="sel_input_cols"
            )
            output_columns = st.multiselect(
                "Output Columns (Y)",
                options=column_headers,
                default=st.session_state.get('output_columns', []),
                key="sel_output_cols"
            )

            if st.button("Confirm Columns & Load Data"):
                error = False
                if not input_columns:
                    st.error("Please select at least one Input Column.")
                    error = True
                if not output_columns:
                    st.error("Please select at least one Output Column.")
                    error = True
                if set(input_columns) & set(output_columns):
                    st.error("Input and Output columns must be different.")
                    error = True

                if not error:
                    input_data, output_data, df_full, error_msg = load_data(uploaded_file, input_columns, output_columns)
                    if error_msg:
                        st.error(error_msg)
                        st.session_state['data_loaded'] = False
                    else:
                        st.session_state['input_data'] = input_data
                        st.session_state['output_data'] = output_data
                        st.session_state['df_full'] = df_full
                        st.session_state['input_columns'] = input_columns
                        st.session_state['output_columns'] = output_columns
                        st.session_state['data_loaded'] = True
                        st.session_state['model_results'] = None
                        st.session_state['model_instance'] = None
                        st.session_state['eat_model_instance'] = None
                        st.session_state['eat_dot_data'] = None
                        st.session_state['eat_predictions'] = None
                        st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error reading file structure: {e}")
            st.session_state['data_loaded'] = False

    if st.session_state['data_loaded']:
        st.header("3. Select Model")
        model_type = st.selectbox(
            "Model Type",
            options=["DEA", "FDH", "EAT"],
            key="sel_model_type",
            index=["DEA", "FDH", "EAT"].index(st.session_state.get('model_type', "DEA"))
        )
        if st.session_state['model_type'] != model_type: 
            st.session_state['model_results'] = None
            st.session_state['model_instance'] = None
            st.session_state['eat_model_instance'] = None
            st.session_state['eat_dot_data'] = None
            st.session_state['eat_predictions'] = None
            st.session_state['model_function_name'] = None
        st.session_state['model_type'] = model_type

        model_functions = {
            "DEA": ["ccr_input", "ccr_output", "bcc_input", "bcc_output",
                    "sbm_non_oriented", "sbm_input", "sbm_output",
                    "add", "rdm", "modified_sbm"],
            "FDH": ["fdh_input_crs", "fdh_input_vrs", "fdh_output_crs", "fdh_output_vrs", "rdm_fdh"],
            "EAT": [] 
        }
        sbm_models = ["sbm_non_oriented", "sbm_input", "sbm_output"]

        if model_type == "EAT":
            st.session_state['model_function_name'] = None 
            st.subheader("EAT Parameters")
            num_stop_val = st.number_input(
                "Stopping Criterion (numStop)",
                min_value=1,
                value=st.session_state.get('eat_num_stop', 5),
                step=1,
                key="eat_num_stop_input"
            )
            st.session_state.eat_num_stop = num_stop_val

            fold_val = st.number_input(
                "Cross-validation Folds (fold)",
                min_value=1, 
                value=st.session_state.get('eat_fold', 5),
                step=1,
                key="eat_fold_input"
            )
            st.session_state.eat_fold = fold_val

        elif model_type in model_functions: 
            available_funcs = model_functions.get(model_type, [])
            if not available_funcs: 
                 st.warning(f"No specific functions currently available for model type: {model_type}")
                 st.session_state['model_function_name'] = None
            else:
                current_func_name = st.session_state.get('model_function_name')
                if current_func_name not in available_funcs:
                    st.session_state['model_function_name'] = available_funcs[0] if available_funcs else None

                model_function_name = st.selectbox(
                    f"Select {model_type} Function",
                    options=available_funcs,
                    key="sel_model_func",
                    index=available_funcs.index(st.session_state.get('model_function_name')) if st.session_state.get('model_function_name') in available_funcs else 0
                )
                st.session_state['model_function_name'] = model_function_name

                if model_type == "DEA" and model_function_name in sbm_models:
                    st.session_state['sbm_crs_checkbox'] = st.checkbox(
                        "CRS (Constant Returns to Scale)",
                        value=st.session_state.get('sbm_crs_checkbox', True),
                        key="sbm_crs_tickbox"
                    )
        else: 
             st.warning(f"Model type {model_type} configuration not recognized.")
             st.session_state['model_function_name'] = None

        if st.button("🚀 Run Analysis"):
            if not st.session_state['data_loaded']:
                st.error("Please load data first.")
                st.stop()

            st.session_state['model_results'] = None
            st.session_state['model_instance'] = None 
            st.session_state['eat_model_instance'] = None 
            st.session_state['eat_dot_data'] = None      
            st.session_state['eat_predictions'] = None   

            current_model_type_run = st.session_state['model_type']

            if current_model_type_run == "EAT":
                try:
                    df_full = st.session_state.get('df_full')
                    if df_full is None:
                        st.error("Full dataset not found. Please reload data.")
                        st.stop()

                    x_cols = st.session_state.input_columns
                    y_cols = st.session_state.output_columns

                    if not x_cols or not y_cols:
                        st.error("Input and/or Output columns must be selected for EAT model.")
                        st.stop()

                    required_cols = x_cols + y_cols
                    if not all(col in df_full.columns for col in required_cols):
                        missing = [col for col in required_cols if col not in df_full.columns]
                        st.error(f"One or more selected columns not found in the dataset: {missing}")
                        st.stop()

                    dataset_for_eat = df_full[required_cols].copy() 

                    num_stop = st.session_state.eat_num_stop
                    fold = st.session_state.eat_fold

                    eat_model_obj = eat.EAT(dataset_for_eat, x_cols, y_cols, numStop=num_stop, fold=fold)
                    eat_model_obj.fit()

                    st.session_state.eat_model_instance = eat_model_obj
                    dot_data = eat_model_obj.export_graphviz('EAT Tree') 
                    st.session_state.eat_dot_data = dot_data
                    st.session_state.model_results = f"EAT model (numStop={num_stop}, fold={fold}) fitted successfully."
                    st.success(st.session_state.model_results)

                except Exception as e:
                    st.error(f"Error running EAT analysis: {e}")
                    st.session_state.model_results = f"EAT analysis failed: {e}"

            elif current_model_type_run in ["DEA", "FDH"]:
                if st.session_state['model_function_name']:
                    try:
                        model_instance = get_model_instance( 
                            current_model_type_run,
                            st.session_state['input_data'],
                            st.session_state['output_data']
                        )
                        st.session_state['model_instance'] = model_instance

                        if model_instance:
                            model_func_to_call_name = st.session_state['model_function_name']
                            if hasattr(model_instance, model_func_to_call_name):
                                model_func = getattr(model_instance, model_func_to_call_name)
                                results_df = None 
                                if current_model_type_run == "DEA" and model_func_to_call_name in sbm_models:
                                    crs_value = st.session_state.get('sbm_crs_checkbox', True)
                                    results_df = model_func(crs=crs_value)
                                else:
                                    results_df = model_func()

                                st.session_state['model_results'] = results_df 
                                st.success(f"{model_func_to_call_name} completed!")
                            else:
                                 st.error(f"Function '{model_func_to_call_name}' not found in {current_model_type_run} model.")
                        else:
                            st.error(f"Could not create model instance for {current_model_type_run}.")
                    except Exception as e:
                        st.error(f"Error running {current_model_type_run} analysis: {e}")
                else:
                    st.warning(f"Please select a model function for {current_model_type_run}.")
            else: 
                st.error(f"Model type {current_model_type_run} not recognized for analysis.")

if st.session_state['data_loaded']:
    st.header("Results")
    current_model_type_disp = st.session_state.get('model_type')

    if current_model_type_disp == "EAT":
        if st.session_state.get('eat_dot_data'):
            st.write(f"Displaying results for: **EAT** (Parameters: numStop={st.session_state.eat_num_stop}, fold={st.session_state.eat_fold})")
            st.subheader("📊 EAT Tree Visualization")
            st.graphviz_chart(st.session_state.eat_dot_data)
        elif st.session_state.get('model_results'): 
            st.info(st.session_state.model_results) 
        else:
            st.info("Run EAT analysis to view the decision tree.")

    elif current_model_type_disp in ["DEA", "FDH"]:
        if st.session_state.get('model_results') is not None and isinstance(st.session_state.model_results, pd.DataFrame):
            st.write(f"Displaying results for: **{current_model_type_disp} - {st.session_state.get('model_function_name','N/A')}**")
            st.dataframe(st.session_state['model_results'])

            st.subheader("📈 Visualization (DEA/FDH)")
            plottable_dea_models = ["ccr_input", "ccr_output", "bcc_input", "bcc_output", "add", "sbm_non_oriented"]
            plottable_fdh_models = ["fdh_input_vrs", "fdh_output_vrs"]

            try:
                if st.session_state['input_data'] is not None and st.session_state['output_data'] is not None:
                    m = st.session_state['input_data'].shape[1]
                    s = st.session_state['output_data'].shape[1]
                else: m, s = 0, 0

                model_instance_plot = st.session_state.get('model_instance') 
                model_func_name_plot = st.session_state.get('model_function_name')
                fig = None
                can_plot = False

                if current_model_type_disp == "DEA" and model_func_name_plot in plottable_dea_models: can_plot = True
                elif current_model_type_disp == "FDH" and model_func_name_plot in plottable_fdh_models: can_plot = True

                if model_instance_plot and model_func_name_plot and can_plot:
                    if model_func_name_plot not in model_instance_plot.results:
                         st.warning(f"Results for '{model_func_name_plot}' not computed or available. Run the analysis first.")
                    elif m == 1 and s == 1:
                        if isinstance(model_instance_plot, DEA): fig = model_instance_plot.plot2d(model_func_name_plot)
                        elif isinstance(model_instance_plot, FDH): fig = model_instance_plot.plot_fdh(model_func_name_plot)
                    elif (m == 2 and s == 1) or (m == 1 and s == 2):
                         if isinstance(model_instance_plot, DEA): fig = model_instance_plot.plot3d(model_func_name_plot)
                         else: st.info("3D plot primarily for DEA models with 2 inputs/1 output or 1 input/2 outputs.")
                    else:
                        if m > 0 and s > 0: st.info(f"Plotting for {m} inputs, {s} outputs not supported (1x1, or DEA 2x1/1x2).")
                    if fig: st.pyplot(fig)

                elif model_func_name_plot and not can_plot:
                     st.info(f"Plotting not available for: {current_model_type_disp} - {model_func_name_plot}.")
            except Exception as e:
                 st.error(f"An error occurred during DEA/FDH plot rendering: {e}")

        elif st.session_state.get('model_results'): 
            st.info(st.session_state.model_results)
        else:
            st.info(f"Run {current_model_type_disp} analysis to view results and plots.")

    elif st.session_state.get('data_loaded'): 
         st.info("Select a model and run analysis to view results.")

    if current_model_type_disp == "EAT" and st.session_state.get('eat_model_instance'):
        st.header("🌳 EAT Model Prediction")
        st.markdown("""
        Predict efficiency scores using the trained EAT model.
        Enter the input values for a new DMU (or observation) below.
        """)

        eat_model_to_predict = st.session_state.eat_model_instance
        x_cols_for_eat_pred = st.session_state.input_columns 

        if not x_cols_for_eat_pred:
            st.warning("Input columns for EAT prediction are not defined. This usually means data hasn't been fully processed or input columns were not selected.")
        else:
            st.subheader("Enter New Input Values for EAT Prediction:")
            new_eat_input_values_dict = {}

            default_input_values_eat = {}
            df_full_for_defaults = st.session_state.get('df_full')
            if df_full_for_defaults is not None:
                for col_name in x_cols_for_eat_pred:
                    if col_name in df_full_for_defaults.columns and pd.api.types.is_numeric_dtype(df_full_for_defaults[col_name]):
                        try:
                            default_input_values_eat[col_name] = float(df_full_for_defaults[col_name].mean())
                        except: 
                            default_input_values_eat[col_name] = 0.0
                    else:
                        default_input_values_eat[col_name] = 0.0 
            else: 
                 for col_name in x_cols_for_eat_pred:
                    default_input_values_eat[col_name] = 0.0

            for col_name in x_cols_for_eat_pred:

                clean_col_name_key = "".join(c if c.isalnum() else "_" for c in col_name)
                default_val = default_input_values_eat.get(col_name, 0.0)
                val = st.number_input(
                    f"Input: {col_name}",
                    value=default_val,
                    step=0.1 if default_val == 0.0 or abs(default_val) < 10 else abs(default_val)/100 , 
                    format="%.4f", 
                    key=f"pred_input_eat_{clean_col_name_key}" 
                )
                new_eat_input_values_dict[col_name] = val

            if st.button("Run EAT Prediction"):
                try:

                    data_to_predict_on_eat = pd.DataFrame([new_eat_input_values_dict])

                    data_to_predict_on_eat = data_to_predict_on_eat[x_cols_for_eat_pred]

                    eat_predictions_df = eat_model_to_predict.predict(data_to_predict_on_eat, x_cols_for_eat_pred)
                    st.session_state.eat_predictions = eat_predictions_df
                    st.subheader("EAT Predictions Results:")
                    st.dataframe(eat_predictions_df)
                except Exception as e:
                    st.error(f"Error during EAT prediction: {e}")
                    st.session_state.eat_predictions = None

            elif st.session_state.get('eat_predictions') is not None: 
                st.subheader("Last EAT Predictions Results:")
                st.dataframe(st.session_state.eat_predictions)

    if st.session_state.get('model_instance') and isinstance(st.session_state['model_instance'], FDH):
        st.header("⚙️ Prediction (Based on FDH Frontier)")
        st.markdown("""
        This section estimates the *maximum potential output* for a new set of inputs,
        based on the performance of *efficient* DMUs in the dataset that use *no more*
        input than specified (FDH concept). This is a common way to benchmark potential.
        *Note: This prediction requires running an FDH model first.*
        """)
        fdh_instance_pred = st.session_state['model_instance']
        available_fdh_results_in_instance = [k for k in fdh_instance_pred.results if k.startswith('fdh_') or k == 'rdm_fdh']

        if available_fdh_results_in_instance:
            chosen_fdh_basis = st.selectbox("Base FDH prediction on efficiency from:", available_fdh_results_in_instance, key="fdh_pred_basis")
            st.subheader("Enter New Input Values for FDH Prediction:")
            new_input_values_fdh = []
            if st.session_state['input_data'] is not None:
                for i in range(st.session_state['input_data'].shape[1]):

                    clean_col_name_key_fdh = "".join(c if c.isalnum() else "_" for c in st.session_state['input_columns'][i])
                    default_val_fdh = float(np.mean(st.session_state['input_data'][:, i])) if st.session_state['input_data'].size > 0 else 0.0
                    val = st.number_input(
                        f"Input {i+1} ({st.session_state['input_columns'][i]})",
                        value=default_val_fdh,
                        step=0.1 if default_val_fdh == 0.0 or abs(default_val_fdh) < 10 else abs(default_val_fdh)/100, 
                        format="%.4f", 
                        key=f"pred_input_fdh_{clean_col_name_key_fdh}_{i}" 
                    )
                    new_input_values_fdh.append(val)

                if st.button("Predict Max Potential Output (FDH)"):
                    try:
                        predicted_output_fdh = fdh_instance_pred.predict(new_input_values_fdh, chosen_fdh_basis)
                        st.subheader("Predicted Maximum Output (FDH):")
                        if isinstance(predicted_output_fdh, (int, float, np.number)):
                             st.success(f"Estimated Max Output: **{predicted_output_fdh:.4f}** (based on {st.session_state['output_columns'][0] if st.session_state['output_columns'] else 'the output'})")
                        elif isinstance(predicted_output_fdh, np.ndarray):
                            out_str_parts = [f"{st.session_state['output_columns'][j] if j < len(st.session_state['output_columns']) else f'Output {j+1}'}: **{val_out:.4f}**" for j, val_out in enumerate(predicted_output_fdh)]
                            st.success(f"Estimated Max Outputs: {', '.join(out_str_parts)}")
                        else: st.warning("FDH Prediction returned an unexpected format.")
                    except ValueError as ve: st.warning(f"FDH Prediction warning: {ve}")
                    except Exception as e: st.error(f"Error during FDH prediction: {e}")
            else: st.warning("Input data not available for FDH prediction setup.")
        else: st.warning("Please run an FDH model analysis first to enable FDH prediction.")
    elif current_model_type_disp == "FDH" and not (st.session_state.get('model_instance') and isinstance(st.session_state['model_instance'], FDH)):
        st.info("Run an FDH analysis to enable the FDH prediction section.")

else: 
    st.info("👈 Upload a data file and select columns to begin.")