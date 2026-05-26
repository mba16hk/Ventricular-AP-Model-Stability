#functions to calculate action potential duration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import plotly.express as px
import plotly.graph_objects as go

def APD_df(simulated_df, cycle_length):
    Voltage_time = simulated_df.loc[:,['time', 'V']]
    Voltage_time = Voltage_time.apply(pd.to_numeric, errors="coerce")
    #ionic_current_time = simulated_df.loc[:,['time', 'dNai','dCai','dKi']]
    chunks = []
    state_chunks = []
    #intermediate_variable_chunks = []
    start_APD_voltage = []
    end_APD_voltage = []
    max_APD_voltage = []
    
    # Identify the time range
    start_time = max(Voltage_time['time'].min(),0)
    end_time = Voltage_time['time'].max()
    
    # Split the dataframe into chunks based on the time intervals
    for start in np.arange(start_time, end_time, cycle_length):
        # Select the rows that fall within the current interval
        chunk = Voltage_time[(Voltage_time['time'] >= start) & (Voltage_time['time'] < start + cycle_length)]
        state_chunk = simulated_df[(simulated_df['time'] >= start) & (simulated_df['time'] < start + cycle_length)]
        #intermediate_variable_chunk = currents_df[(currents_df['time'] >= start) & (currents_df['time'] < start + cycle_length)]
        start_times_idx = chunk.index[chunk['V'] == chunk['V'].max()].values[0]
        end_times_idx = chunk.index[chunk['V'] == chunk['V'].min()].values[0]
        target_voltage = chunk.loc[end_times_idx,['V']].values[0]
        start_times = chunk.loc[start_times_idx,['time']].values[0]
        max_volts = chunk.loc[start_times_idx,['V']].values[0]
        chunks.append(chunk)
        state_chunks.append(state_chunk)
        #intermediate_variable_chunks.append(intermediate_variable_chunk)
        start_APD_voltage.append(start_times)
        end_APD_voltage.append(target_voltage)
        max_APD_voltage.append(max_volts)
    
    return chunks,start_APD_voltage,end_APD_voltage,max_APD_voltage, state_chunks#, intermediate_variable_chunks

def Calculate_APD(chunks, state_chunks, start_time, target_voltage, max_volts, percentage, stimulus_duration):

    APD = []
    times = []
    cycles = []
    resting_mem_potential = []
    AP_max = []
    AP_min = []
    AP_amp = []
    dV_dtmax_steepest = []
    Catmax_list = []
    Catmin_list = []
    Catamp_list = []

    for i in range(0,len(chunks)):

        total_V_trace = abs(max_volts[i]-target_voltage[i])
        percentage_drop = (percentage/100)*total_V_trace
        target_voltage_value = max_volts[i]-percentage_drop
        chunk = chunks[i]
        resting_voltage = chunk['V'].iloc[-1]
        df = chunk[(chunk['time'] >= start_time[i])]
        sorted_max_voltage = df.iloc[(df['V']-target_voltage_value).abs().argsort()[:1]]
        calculated_APD = sorted_max_voltage['time'].values[0] - start_time[i]

        max_V = chunk["V"].max()
        min_V = chunk["V"].iloc[0]
        AP_amplitude = max_V - min_V
        AP_max.append(max_V)
        AP_min.append(min_V)
        AP_amp.append(AP_amplitude)

        min_time = chunk['time'].min()
        df_filtered = chunk[chunk['time'] > (min_time + stimulus_duration)]
        dv_dt = np.gradient(df_filtered['V'].to_numpy(), df_filtered['time'].to_numpy())
        max_dv_dt = np.max(dv_dt)
        dV_dtmax_steepest.append(max_dv_dt)

        # Ca transient metrics from in-memory state chunk (logic ported from Ca_Transient.py)
        state_chunk = state_chunks[i]
        Catmax = state_chunk['Cai'].max()
        Catmin = state_chunk['Cai'].min()
        Catmax_list.append(Catmax)
        Catmin_list.append(Catmin)
        Catamp_list.append(Catmax - Catmin)

        APD.append(calculated_APD)
        times.append(start_time[i])
        cycles.append(i)
        resting_mem_potential.append(resting_voltage)

    APD_df = pd.DataFrame({'time': times, 'Cycle':cycles,'APD': APD})
    AP_amp_prop = pd.DataFrame({'time': times, 'Cycle':cycles,'AP_amp': AP_amp})
    AP_min_prop = pd.DataFrame({'time': times, 'Cycle':cycles,'AP_min': AP_min})
    AP_max_prop = pd.DataFrame({'time': times, 'Cycle':cycles,'AP_max': AP_max})
    resting_mem_potential_df = pd.DataFrame({'time': times, 'Cycle':cycles,'Resting Membrane Potential': resting_mem_potential})
    dV_dtmax_steepest_df = pd.DataFrame({'time': times, 'Cycle':cycles,'dV/dtmax steepest': dV_dtmax_steepest})
    Ca_transient_df = pd.DataFrame({'time': times, 'Cycle':cycles, 'Catmax': Catmax_list, 'Catmin': Catmin_list, 'Catamp': Catamp_list})
    return APD_df, resting_mem_potential_df, dV_dtmax_steepest_df, AP_amp_prop, AP_min_prop, AP_max_prop, Ca_transient_df

def plot_APD(chunks, state_chunks, start_time, target_voltage, max_volts, stimulus_duration, plot_fig):
    """
    Plots APD over time.

    Parameters:
    APD_df (pd.DataFrame): DataFrame with 'time' as the x-axis and 'APD' as the y-axis.
    """
    print("I have entered this function")
    APD30, resting_mem_potential_df, dV_dtmax_steepest_df, AP_amp, AP_min, AP_max, Ca_transient_df = Calculate_APD(chunks, state_chunks, start_time, target_voltage, max_volts, 30, stimulus_duration)
    APD50, resting_mem_potential_df, dV_dtmax_steepest_df, AP_amp, AP_min, AP_max, Ca_transient_df = Calculate_APD(chunks, state_chunks, start_time, target_voltage, max_volts, 50, stimulus_duration)
    APD60, resting_mem_potential_df, dV_dtmax_steepest_df, AP_amp, AP_min, AP_max, Ca_transient_df = Calculate_APD(chunks, state_chunks, start_time, target_voltage, max_volts, 60, stimulus_duration)
    APD90, resting_mem_potential_df, dV_dtmax_steepest_df, AP_amp, AP_min, AP_max, Ca_transient_df = Calculate_APD(chunks, state_chunks, start_time, target_voltage, max_volts, 90, stimulus_duration)

    Triangulation = APD90.copy()
    Triangulation['Triangulation 30'] = APD90['APD']-APD30['APD']
    Triangulation['Triangulation 60'] = APD90['APD']-APD60['APD']
    Triangulation = Triangulation.drop(columns=['APD'])

    APD_df_30 = APD30.rename(columns={'APD': 'APD30'})
    APD_df_50 = APD50.rename(columns={'APD': 'APD50'})
    APD_df_60 = APD60.rename(columns={'APD': 'APD60'})
    APD_df_90 = APD90.rename(columns={'APD': 'APD90'})

    # Merge the dataframes on 'time' and 'Cycle' to avoid duplication
    merged_df = APD_df_30.merge(APD_df_50, on=['time', 'Cycle'], how='outer')\
                        .merge(APD_df_60, on=['time', 'Cycle'], how='outer')\
                        .merge(APD_df_90, on=['time', 'Cycle'], how='outer')\
                        .merge(AP_amp, on=['time', 'Cycle'], how='outer')\
                        .merge(AP_min, on=['time', 'Cycle'], how='outer')\
                        .merge(AP_max, on=['time', 'Cycle'], how='outer')\
                        .merge(resting_mem_potential_df, on=['time', 'Cycle'], how='outer')\
                        .merge(dV_dtmax_steepest_df, on=['time', 'Cycle'], how='outer')\
                        .merge(Triangulation, on=['time', 'Cycle'], how='outer')\
                        .merge(Ca_transient_df, on=['time', 'Cycle'], how='outer')
                        
    #Calculate the rate of change of APD
    # Calculate time difference
    dt = merged_df['time'].diff()

    # Calculate rate of change for each APD
    differential_df = pd.DataFrame({
        'Cycle' : merged_df['Cycle'],
        'dAPD30_dt' : merged_df['APD30'].diff() / dt,
        'dAPD50_dt' : merged_df['APD50'].diff() / dt,
        'dAPD60_dt' : merged_df['APD60'].diff() / dt,
        'dAPD90_dt' : merged_df['APD90'].diff() / dt,
        'dAP_amp_dt' : merged_df['AP_amp'].diff() / dt,
        'dAP_min_dt' : merged_df['AP_min'].diff() / dt,
        'dAP_max_dt' : merged_df['AP_max'].diff() / dt,
        'd(dV/dtmax steepest)_dt' : merged_df['dV/dtmax steepest'].diff() / dt
    })
    
    if plot_fig == True:                   
        # Use a clean seaborn style
        sns.set_context("paper", font_scale=1.4)
        sns.set_style("whitegrid", {
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.2,
            'grid.color': 'lightgrey',
            'grid.linestyle': '--'
        })

        fig, axs = plt.subplots(4, 2, figsize=(14, 16), dpi=150)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        marker_size = 4
        line_width = 1.5
        color_palette = sns.color_palette("deep", 6)  # Professional palette

        # Plot 1: APD
        labels = ["APD30", "APD50", "APD60", "APD90"]
        APD_dfs = [APD30, APD50, APD60, APD90]
        ax = axs[0, 0]
        for i, df in enumerate(APD_dfs):
            ax.plot(df['Cycle'], df['APD'], marker='o',markersize = marker_size,linewidth = line_width, linestyle='-', color=color_palette[i], label=labels[i])
        ax.set_title("APD (ms)")
        ax.legend(frameon=False, loc='upper center', ncol = 4)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)
            
        ax = axs[0, 1]
        ax.plot(differential_df['Cycle'], differential_df['dAPD30_dt'], label='dAPD30/dt', color=color_palette[1])
        ax.plot(differential_df['Cycle'], differential_df['dAPD50_dt'], label='dAPD50/dt', color=color_palette[2])
        ax.plot(differential_df['Cycle'], differential_df['dAPD60_dt'], label='dAPD60/dt', color=color_palette[3])
        ax.plot(differential_df['Cycle'], differential_df['dAPD90_dt'], label='dAPD90/dt', color=color_palette[4])
        ax.set_title("Rate of chnage of APD")
        ax.legend(frameon=False, loc='upper center', ncol = 4)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)
            
        # Plot 3: dV/dtmax
        ax = axs[1, 0]
        ax.plot(merged_df['Cycle'], merged_df['AP_amp'], label='AP amplitude', color=color_palette[1])
        ax.plot(merged_df['Cycle'], merged_df['AP_min'], label='AP minimum', color=color_palette[2])
        ax.plot(merged_df['Cycle'], merged_df['AP_max'], label='AP maximum', color=color_palette[3])
        ax.set_title("AP Upstroke")
        ax.legend(frameon=False, loc='upper center', ncol = 3)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)
            
        ax = axs[1, 1]
        ax.plot(differential_df['Cycle'], differential_df['dAP_amp_dt'], label='dAP amplitude/dt', color=color_palette[1])
        ax.plot(differential_df['Cycle'], differential_df['dAP_min_dt'], label='dAP minimum/dt', color=color_palette[2])
        ax.plot(differential_df['Cycle'], differential_df['dAP_max_dt'], label='dAP maximum/dt', color=color_palette[3])
        ax.set_title("Rate of AP Upstroke")
        ax.legend(frameon=False, loc='upper center', ncol = 3)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)
            
        # Plot 4: dV/dtmax steepest
        ax = axs[2, 0]
        ax.plot(dV_dtmax_steepest_df['Cycle'], dV_dtmax_steepest_df['dV/dtmax steepest'],
                markersize = marker_size,linewidth = line_width,
                marker='o', linestyle='-', color='b')
        ax.set_title("dV/dtmax steepest point (mV/ms)")
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)
            
        ax = axs[2, 1]
        ax.plot(differential_df['Cycle'], differential_df['d(dV/dtmax steepest)_dt'],
                markersize = marker_size,linewidth = line_width,
                marker='o', linestyle='-', color='b')
        ax.set_title("Rate of Change of dV/dtmax at steepest point")
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)

        # Plot 5: Resting Membrane Potential
        ax = axs[3, 0]
        ax.plot(resting_mem_potential_df['Cycle'],
                resting_mem_potential_df['Resting Membrane Potential'],
                markersize = marker_size,linewidth = line_width,
                marker='o', linestyle='-', color='y')
        ax.set_title("Resting Membrane Potential (mV)")
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)

        # Plot 6: Triangulation
        ax = axs[3, 1]
        ax.plot(Triangulation['Cycle'], Triangulation['Triangulation 30'],
                markersize = marker_size,linewidth = line_width,
                marker='o', linestyle='-', color=color_palette[1], label="Triangulation 30")
        ax.plot(Triangulation['Cycle'], Triangulation['Triangulation 60'],
                markersize = marker_size,linewidth = line_width,
                marker='o', linestyle='-', color=color_palette[2], label="Triangulation 60")
        ax.set_title("Triangulation (ms)")
        ax.legend(frameon=False, loc='right')
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('dimgray')
            ax.spines[spine].set_linewidth(1.2)

        fig.text(0.5, 0.0, 'Cycle', ha='center', va='center', fontsize=24)
        # Clean up whitespace
        fig.tight_layout()
    
    if plot_fig == False:
        return merged_df
    else:
        return merged_df, fig

def plot_voltage(chunks,num_chunks,bs_df):
    
    """
    Combines the last 'num_chunks' chunks from a list into a single DataFrame.

    Parameters:
    chunks (list of pd.DataFrame): A list of DataFrames (chunks).
    num_chunks (int): Number of last chunks to combine.

    Returns:
    pd.DataFrame: A concatenated DataFrame containing the last 'num_chunks'.
    """
    if num_chunks > len(chunks):
        num_chunks = len(chunks)  # Avoid out-of-range issues
    
    selected_chunks = chunks[-num_chunks:]  # Select the last num_chunks
    plottable_df = pd.concat(selected_chunks, ignore_index=True)  # Combine into one DataFrame
    
    if (bs_df is not None):
        # Shift bs_df time so it starts at the same time as chunk
        time_offset = plottable_df['time'].iloc[0] - bs_df['time'].iloc[0]
        bs_df['time'] += time_offset
        bs_df = bs_df[bs_df['time'] <= plottable_df['time'].max()]
        # Create figure
        fig = go.Figure()
        
        # Add second scatter plot
        fig.add_trace(go.Scatter(
            x=bs_df['time'],
            y=bs_df['V'],
            mode='lines',
            name='Baseline',  # Legend entry
            marker=dict(color='black')
        ))

        # Add first scatter plot
        fig.add_trace(go.Scatter(
            x=plottable_df['time'],
            y=plottable_df['V'],
            mode='lines',
            name='Adjusted conductance(s)',  # Legend entry
            marker=dict(color='red')
        ))

        # Layout options
        fig.update_layout(
            title='dV/dt',
            xaxis_title='time (ms)',
            yaxis_title='V (mV)',
            legend_title='AP'
        )
        
    else:
        fig = px.line(
        plottable_df, x='time', y='V',
        title="dV/dt"
        )
    
    return fig

