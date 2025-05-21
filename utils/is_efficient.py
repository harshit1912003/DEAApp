import numpy as np

def is_efficient(results, efficiency_type):
    if efficiency_type == 'ccr_input':
        slacks_minus_zero = np.array([np.all(np.array(s) == 0) for s in results['slacks_minus']])
        slacks_plus_zero = np.array([np.all(np.array(s) == 0) for s in results['slacks_plus']])

        results['is_efficient'] = ((results['efficiency'] == 1) &
                                   np.logical_and(slacks_minus_zero, slacks_plus_zero))
    elif efficiency_type == 'ccr_output':
        t_minus_zero = np.array([np.all(np.array(t) == 0) for t in results['t_minus']])
        t_plus_zero = np.array([np.all(np.array(t) == 0) for t in results['t_plus']])

        results['is_efficient'] = ((results['n'] == 1) &
                                   np.logical_and(t_minus_zero, t_plus_zero))
    elif efficiency_type == 'bcc_input':
        slacks_minus_zero = np.array([np.all(np.array(s) == 0) for s in results['slacks_minus']])
        slacks_plus_zero = np.array([np.all(np.array(s) == 0) for s in results['slacks_plus']])

        results['is_efficient'] = ((results['efficiency'] == 1) &
                                   np.logical_and(slacks_minus_zero, slacks_plus_zero))
    elif efficiency_type == 'bcc_output':
        t_minus_zero = np.array([np.all(np.array(t) == 0) for t in results['t_minus']])
        t_plus_zero = np.array([np.all(np.array(t) == 0) for t in results['t_plus']])

        results['is_efficient'] = ((results['n'] == 1) &
                                   np.logical_and(t_minus_zero, t_plus_zero))

    elif efficiency_type == 'add':
        slacks_minus_zero = np.array([np.all(np.array(s) == 0) for s in results['slacks_minus']])
        slacks_plus_zero = np.array([np.all(np.array(s) == 0) for s in results['slacks_plus']])

        results['is_efficient'] = np.logical_and(slacks_minus_zero, slacks_plus_zero)

    elif efficiency_type == 'sbm':
        results['is_efficient'] = (results['rho'] == 1)

    elif efficiency_type == 'sbm_non_oriented':
        results['is_efficient'] = (results['rho'] == 1)
    elif efficiency_type == 'sbm_input':
        results['is_efficient'] = (results['rho'] == 1)
    elif efficiency_type == 'sbm_output':
        results['is_efficient'] = (results['rho'] == 1)
    elif efficiency_type == 'modified_sbm':
        results['is_efficient'] = (results['rho'] == 1)


    elif efficiency_type == 'fdh_input_crs':
        results['is_efficient'] = (results['efficiency'] == 1)

    elif efficiency_type == 'fdh_output_crs':
        results['is_efficient'] = (results['efficiency'] == 1)
        
    elif efficiency_type == 'fdh_input_vrs':
        results['is_efficient'] = (results['efficiency'] == 1)

    elif efficiency_type == 'fdh_output_vrs':
        results['is_efficient'] = (results['efficiency'] == 1)
    
    elif efficiency_type == 'rdm':
        results['is_efficient'] = (results['efficiency'] == 1)
        
    elif efficiency_type == 'rdm_fdh':
        results['is_efficient'] = (results['efficiency'] == 1)
    else:
        raise ValueError(f"Unknown efficiency type: {efficiency_type}")

    return results
