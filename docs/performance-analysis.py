import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import numpy at the top level
from scipy import stats

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['grid_size'] = df['width'] * df['height']
    df['implementation'] = df['target'].str.extract(r'(std|openmp|cuda)')
    return df

def add_trend_line_equation(ax, x, y, color, label=None, position=None):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    equation = f'{label if label else ""}: y = {slope:.10f}x + {intercept:.6f}'
    r_squared = f'R² = {r_value**2:.3f}'
    
    if position is None:
        position = (0.05, 0.95)
        
    ax.text(position[0], position[1], equation + '\n' + r_squared,
            transform=ax.transAxes,
            verticalalignment='top',
            color=color,
            bbox=dict(facecolor='white', alpha=0.8))

def create_performance_plots(df):
    metrics = {
        'time_init': 'Initialization Time',
        'time_simulation': 'Simulation Time',
        'time_per_round': 'Time per Round',
        'time_total': 'Total Time'
    }
    
    implementations = df['target'].unique()
    
    for implementation in implementations:
        impl_data = df[df['target'] == implementation]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Performance Analysis - {implementation}', fontsize=16)
        axes = axes.ravel()
        
        for idx, (metric, title) in enumerate(metrics.items()):
            sns.scatterplot(
                data=impl_data,
                x='grid_size',
                y=metric,
                ax=axes[idx],
                alpha=0.7,
                color='blue',
                label='Measured data'
            )
            
            x = impl_data['grid_size']
            y = impl_data[metric]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = pd.Series([x.min(), x.max()])
            line_y = slope * line_x + intercept
            axes[idx].plot(line_x, line_y, color='red', label='Linear regression')
            
            add_trend_line_equation(axes[idx], x, y, 'black')
            
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Grid Size (width × height)')
            axes[idx].set_ylabel('Time (seconds)')
            axes[idx].legend()
            axes[idx].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'performance_analysis_{implementation}.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.close()

def create_combined_performance_plots(df):
    metrics = {
        'time_init': 'Initialization Time',
        'time_simulation': 'Simulation Time',
        'time_per_round': 'Time per Round',
        'time_total': 'Total Time'
    }
    
    colors = {
        'std': 'blue',
        'openmp': 'green',
        'cuda': 'red'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Combined Performance Analysis - All Implementations', fontsize=16)
    axes = axes.ravel()
    
    for idx, (metric, title) in enumerate(metrics.items()):
        y_position = 0.95
        
        for implementation, color in colors.items():
            impl_data = df[df['target'] == implementation]
            
            if impl_data.empty:
                continue
                
            sns.scatterplot(
                data=impl_data,
                x='grid_size',
                y=metric,
                ax=axes[idx],
                alpha=0.7,
                color=color,
                label=f'{implementation} data'
            )
            
            x = impl_data['grid_size']
            y = impl_data[metric]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = pd.Series([x.min(), x.max()])
            line_y = slope * line_x + intercept
            axes[idx].plot(line_x, line_y, color=color, linestyle='--', 
                         label=f'{implementation} regression')
            
            add_trend_line_equation(axes[idx], x, y, color, implementation, 
                                 (0.05, y_position))
            y_position -= 0.15
        
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Grid Size (width × height)')
        axes[idx].set_ylabel('Time (seconds)')
        axes[idx].legend()
        axes[idx].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('performance_analysis_combined.svg', dpi=300, bbox_inches='tight', format='svg')
    plt.close()

def create_zoomed_performance_plots(df):
    metrics = {
        'time_init': 'Initialization Time',
        'time_simulation': 'Simulation Time',
        'time_per_round': 'Time per Round',
        'time_total': 'Total Time'
    }
    
    colors = {
        'std': 'blue',
        'openmp': 'green',
        'cuda': 'red'
    }
    
    # Define grid size range
    min_grid_size = 0
    max_grid_size = 8192 * 8192# 4096 x 4096
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Zoomed Performance Analysis 8192 x 8192 on range 0 to 2500 x 2500 - All Implementations', fontsize=16)
    axes = axes.ravel()
    
    for idx, (metric, title) in enumerate(metrics.items()):
        y_position = 0.95
        
        # Calculate y-axis limits for this metric
        filtered_df = df[df['grid_size'].between(min_grid_size, max_grid_size)]
        y_min = filtered_df[metric].min()
        y_max = filtered_df[metric].max()
        y_range = y_max - y_min
        
        # Add 10% padding to y-axis
        y_min = y_min - (y_range * 0.1)
        y_max = y_max + (y_range * 0.1)
        
        for implementation, color in colors.items():
            impl_data = df[df['target'] == implementation]
            impl_data = impl_data[
                (impl_data['grid_size'] >= min_grid_size) & 
                (impl_data['grid_size'] <= max_grid_size)
            ]
            
            if impl_data.empty:
                continue
                
            sns.scatterplot(
                data=impl_data,
                x='grid_size',
                y=metric,
                ax=axes[idx],
                alpha=0.7,
                color=color,
                label=f'{implementation} data'
            )
            
            x = impl_data['grid_size']
            y = impl_data[metric]
            
            if len(x) > 1:  # Only calculate regression if we have enough data points
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line_x = pd.Series([min_grid_size, max_grid_size])
                line_y = slope * line_x + intercept
                axes[idx].plot(line_x, line_y, color=color, linestyle='--', 
                             label=f'{implementation} regression')
                
                add_trend_line_equation(axes[idx], x, y, color, implementation, 
                                     (0.05, y_position))
            y_position -= 0.15
        
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Grid Size (width × height)')
        axes[idx].set_ylabel('Time (seconds)')
        axes[idx].legend()
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
        def format_func(value, tick_number):
            if value <= 0:
                return '0x0'
            if pd.isna(value):
                return '0x0'
            side = int(np.sqrt(value))
            return f'{side}x{side}'
        
        axes[idx].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axes[idx].xaxis.set_major_locator(plt.MaxNLocator(5))
        
        # Set axes limits
        axes[idx].set_xlim(min_grid_size, 2500 * 2500)
        div = 15
        axes[idx].set_ylim(y_min / div, y_max / div)
    
    plt.tight_layout()
    plt.savefig('performance_analysis_zoomed.svg', dpi=300, bbox_inches='tight', format='svg')
    plt.close()

def main():
    df = load_and_prepare_data('benchmark_result.csv')
    # create_performance_plots(df)
    # create_combined_performance_plots(df)
    create_zoomed_performance_plots(df)

if __name__ == "__main__":
    main()