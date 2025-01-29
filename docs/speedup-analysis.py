import pandas as pd
import numpy as np

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['grid_size'] = df['width'] * df['height']
    df['implementation'] = df['target'].str.extract(r'(std|openmp|cuda)')
    return df

def format_grid_size(size):
    side = int(np.sqrt(size))
    return f"{side}x{side}"

def df_to_markdown(df, title, description=None):
    markdown = f"### {title}\n\n"
    if description:
        markdown += f"{description}\n\n"
    
    # Convert DataFrame to markdown
    markdown += df.to_markdown(index=True, floatfmt=".3f")
    markdown += "\n\n"
    return markdown

def create_comparison_tables(df):
    # Metrics we want to analyze
    metrics = ['time_init', 'time_simulation', 'time_per_round', 'time_total']
    metrics_names = {
        'time_init': 'Initialization Time',
        'time_simulation': 'Simulation Time',
        'time_per_round': 'Time per Round',
        'time_total': 'Total Time'
    }
    
    markdown_output = "# Performance Comparison Analysis\n\n"
    
    for metric in metrics:
        markdown_output += f"## {metrics_names[metric]} Analysis\n\n"
        
        # Create pivot table
        pivot = df.pivot(index='grid_size', columns='implementation', values=metric)
        
        # Format grid size index
        pivot.index = pivot.index.map(format_grid_size)
        
        # Absolute time differences
        diff_table = pd.DataFrame(index=pivot.index)
        diff_table['OpenMP - STD'] = pivot['openmp'] - pivot['std']
        diff_table['CUDA - STD'] = pivot['cuda'] - pivot['std']
        diff_table['CUDA - OpenMP'] = pivot['cuda'] - pivot['openmp']
        
        markdown_output += df_to_markdown(
            diff_table,
            "Absolute Time Differences (seconds)",
            "Negative values indicate faster execution"
        )
        
        # Speedup ratios
        speedup_table = pd.DataFrame(index=pivot.index)
        speedup_table['OpenMP vs STD'] = pivot['std'] / pivot['openmp']
        speedup_table['CUDA vs STD'] = pivot['std'] / pivot['cuda']
        speedup_table['CUDA vs OpenMP'] = pivot['openmp'] / pivot['cuda']
        
        # Format speedup values
        for col in speedup_table.columns:
            speedup_table[col] = speedup_table[col].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
        
        markdown_output += df_to_markdown(
            speedup_table,
            "Speedup Ratios",
            "Values > 1 indicate faster execution (e.g., 2.00x means twice as fast)"
        )
        
    return markdown_output

def calculate_summary_statistics(df):
    metrics = ['time_init', 'time_simulation', 'time_per_round', 'time_total']
    metrics_names = {
        'time_init': 'Initialization Time',
        'time_simulation': 'Simulation Time',
        'time_per_round': 'Time per Round',
        'time_total': 'Total Time'
    }
    
    markdown_output = "# Summary Statistics\n\n"
    
    for metric in metrics:
        pivot = df.pivot(index='grid_size', columns='implementation', values=metric)
        
        # Calculate speedup ratios
        speedup = pd.DataFrame(index=pivot.index)
        speedup['openmp_vs_std'] = pivot['std'] / pivot['openmp']
        speedup['cuda_vs_std'] = pivot['std'] / pivot['cuda']
        speedup['cuda_vs_openmp'] = pivot['openmp'] / pivot['cuda']
        
        # Calculate summary statistics
        summary = pd.DataFrame({
            'OpenMP vs STD': {
                'Mean Speedup': speedup['openmp_vs_std'].mean(),
                'Max Speedup': speedup['openmp_vs_std'].max(),
                'Min Speedup': speedup['openmp_vs_std'].min(),
                'Median Speedup': speedup['openmp_vs_std'].median()
            },
            'CUDA vs STD': {
                'Mean Speedup': speedup['cuda_vs_std'].mean(),
                'Max Speedup': speedup['cuda_vs_std'].max(),
                'Min Speedup': speedup['cuda_vs_std'].min(),
                'Median Speedup': speedup['cuda_vs_std'].median()
            },
            'CUDA vs OpenMP': {
                'Mean Speedup': speedup['cuda_vs_openmp'].mean(),
                'Max Speedup': speedup['cuda_vs_openmp'].max(),
                'Min Speedup': speedup['cuda_vs_openmp'].min(),
                'Median Speedup': speedup['cuda_vs_openmp'].median()
            }
        })
        
        # Format the summary values
        formatted_summary = summary.round(2)
        formatted_summary = formatted_summary.applymap(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
        
        markdown_output += df_to_markdown(
            formatted_summary,
            f"{metrics_names[metric]} - Summary Statistics"
        )
    
    return markdown_output

def main():
    # Load and process data
    df = load_and_prepare_data('benchmark_result.csv')
    
    # Generate markdown output
    comparison_md = create_comparison_tables(df)
    summary_md = calculate_summary_statistics(df)
    
    # Save markdown to files
    with open('performance_comparison.md', 'w') as f:
        f.write(comparison_md)
    
    with open('performance_summary.md', 'w') as f:
        f.write(summary_md)

if __name__ == "__main__":
    main()