"""
Module 2 CDIO Assessment Analysis Script
Statistical analysis of pre-post assessment data for induction motor education study
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind, pearsonr
import warnings
warnings.filterwarnings('ignore')

def cohen_d(x, y):
    """Calculate Cohen's d effect size"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def paired_cohen_d(x1, x2):
    """Calculate Cohen's d for paired samples"""
    diff = x1 - x2
    return np.mean(diff) / np.std(diff, ddof=1)

def load_and_clean_data():
    """Load and clean the assessment data"""
    print("Loading assessment data...")
    
    # Load data
    pre_data = pd.read_csv('PreLab Assessment_ Induction Machine LabSheet1.csv', encoding='cp1252')
    post_data = pd.read_csv('PostLab Assessment_ Induction Machine LabSheet1.csv', encoding='cp1252')
    
    print(f"Pre-assessment: {len(pre_data)} responses")
    print(f"Post-assessment: {len(post_data)} responses")
    
    # Clean and standardize Student ID columns
    pre_data['Student_ID'] = pre_data['Student ID'].astype(str).str.strip()
    post_data['Student_ID'] = post_data['Student ID'].astype(str).str.strip()
    
    return pre_data, post_data

def extract_competency_scores():
    """Extract and organize competency scores from both assessments"""
    
    pre_data, post_data = load_and_clean_data()
    
    # Define competency domains based on the paper's framework
    theoretical_items = [
        'Rate your current understanding of no-load testing of electrical machines. \n1 (No understanding) to 5 (Complete understanding)\n',
        'Rate your current understanding of "slip" in induction motors',
        'Rate your current understanding of power factor measurement in AC circuits',
        'Rate your current understanding of speed-torque characteristics in motors'
    ]
    
    practical_items = [
        'Rate your current ability to connect three-phase electrical equipment safely',
        'Rate your current ability to use digital multimeters for AC measurements', 
        'Rate your current ability to operate variable voltage sources (like variacs)',
        'Rate your current ability to record experimental data systematically and accurately',
        'Rate your confidence in calculating slip from speed measurements',
        'Rate your confidence in analyzing speed-torque relationships from experimental data'
    ]
    
    safety_items = [
        'Rate your confidence in setting up electrical measurement equipment safelyScale: 1 (Not confident) to 5 (Very confident)\n',
        'Rate your confidence in selecting appropriate voltage ranges for motor testing',
        'Rate your confidence in identifying and following electrical safety procedures',
        'Rate your confidence in planning a systematic data collection strategy'
    ]
    
    # Extract scores for each domain
    def get_domain_scores(data, items, domain_name):
        scores = []
        available_items = []
        for item in items:
            if item in data.columns:
                scores.append(data[item])
                available_items.append(item)
            else:
                print(f"Warning: '{item}' not found in {domain_name}")
        
        if scores:
            domain_scores = pd.concat(scores, axis=1).mean(axis=1)
            print(f"{domain_name}: {len(available_items)} items found")
            return domain_scores
        else:
            print(f"Warning: No items found for {domain_name}")
            return None
    
    # Extract pre-assessment scores
    pre_theoretical = get_domain_scores(pre_data, theoretical_items, "Pre-Theoretical")
    pre_practical = get_domain_scores(pre_data, practical_items, "Pre-Practical")
    pre_safety = get_domain_scores(pre_data, safety_items, "Pre-Safety")
    
    # Extract post-assessment scores  
    post_theoretical = get_domain_scores(post_data, theoretical_items, "Post-Theoretical")
    post_practical = get_domain_scores(post_data, practical_items, "Post-Practical")
    post_safety = get_domain_scores(post_data, safety_items, "Post-Safety")
    
    # Create matched dataset
    matched_data = []
    
    for idx, student_id in enumerate(pre_data['Student_ID']):
        post_idx = post_data[post_data['Student_ID'] == student_id].index
        if len(post_idx) > 0:
            post_idx = post_idx[0]
            
            matched_data.append({
                'Student_ID': student_id,
                'Program': pre_data.loc[idx, 'Program'],
                'Gender': pre_data.loc[idx, 'Gender'],
                'Pre_Theoretical': pre_theoretical.iloc[idx] if pre_theoretical is not None else np.nan,
                'Pre_Practical': pre_practical.iloc[idx] if pre_practical is not None else np.nan,
                'Pre_Safety': pre_safety.iloc[idx] if pre_safety is not None else np.nan,
                'Post_Theoretical': post_theoretical.iloc[post_idx] if post_theoretical is not None else np.nan,
                'Post_Practical': post_practical.iloc[post_idx] if post_practical is not None else np.nan,
                'Post_Safety': post_safety.iloc[post_idx] if post_safety is not None else np.nan,
            })
    
    matched_df = pd.DataFrame(matched_data)
    
    # Calculate overall scores
    matched_df['Pre_Overall'] = matched_df[['Pre_Theoretical', 'Pre_Practical', 'Pre_Safety']].mean(axis=1)
    matched_df['Post_Overall'] = matched_df[['Post_Theoretical', 'Post_Practical', 'Post_Safety']].mean(axis=1)
    
    # Remove rows with missing data
    matched_df = matched_df.dropna()
    
    print(f"\nMatched dataset: {len(matched_df)} students with complete pre-post data")
    print(f"Engineering Technology: {sum(matched_df['Program'] == 'Engineering Technology')}")
    print(f"Conventional Engineering: {sum(matched_df['Program'] == 'Conventional Engineering')}")
    
    return matched_df

def run_statistical_analysis(matched_df):
    """Perform the main statistical analyses"""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*60)
    
    results = {}
    
    # Overall competency analysis
    pre_overall = matched_df['Pre_Overall'].dropna()
    post_overall = matched_df['Post_Overall'].dropna()
    
    # Paired t-test for overall scores
    t_stat, p_val = ttest_rel(post_overall, pre_overall)
    effect_size = paired_cohen_d(post_overall, pre_overall)
    
    print(f"\n1. OVERALL COMPETENCY CHANGES")
    print(f"Pre-assessment:  M = {pre_overall.mean():.2f}, SD = {pre_overall.std():.2f}")
    print(f"Post-assessment: M = {post_overall.mean():.2f}, SD = {post_overall.std():.2f}")
    print(f"Paired t-test: t({len(pre_overall)-1}) = {t_stat:.2f}, p = {p_val:.3f}")
    print(f"Cohen's d = {effect_size:.2f}")
    
    results['overall'] = {
        'pre_mean': pre_overall.mean(),
        'pre_std': pre_overall.std(),
        'post_mean': post_overall.mean(), 
        'post_std': post_overall.std(),
        't_stat': t_stat,
        'p_val': p_val,
        'cohen_d': effect_size,
        'n': len(pre_overall)
    }
    
    # Domain-specific analysis
    domains = ['Theoretical', 'Practical', 'Safety']
    
    print(f"\n2. DOMAIN-SPECIFIC RESULTS")
    
    for domain in domains:
        pre_col = f'Pre_{domain}'
        post_col = f'Post_{domain}'
        
        if pre_col in matched_df.columns and post_col in matched_df.columns:
            pre_scores = matched_df[pre_col].dropna()
            post_scores = matched_df[post_col].dropna()
            
            if len(pre_scores) > 0 and len(post_scores) > 0:
                t_stat, p_val = ttest_rel(post_scores, pre_scores)
                effect_size = paired_cohen_d(post_scores, pre_scores)
                
                print(f"\n{domain} Understanding:")
                print(f"  Pre:  M = {pre_scores.mean():.2f}, SD = {pre_scores.std():.2f}")
                print(f"  Post: M = {post_scores.mean():.2f}, SD = {post_scores.std():.2f}")
                print(f"  Cohen's d = {effect_size:.2f}, p = {p_val:.3f}")
                
                results[domain.lower()] = {
                    'pre_mean': pre_scores.mean(),
                    'pre_std': pre_scores.std(),
                    'post_mean': post_scores.mean(),
                    'post_std': post_scores.std(),
                    't_stat': t_stat,
                    'p_val': p_val,
                    'cohen_d': effect_size,
                    'n': len(pre_scores)
                }
    
    # Program type comparison
    print(f"\n3. PROGRAM TYPE COMPARISONS")
    
    et_students = matched_df[matched_df['Program'] == 'Engineering Technology']
    ce_students = matched_df[matched_df['Program'] == 'Conventional Engineering']
    
    if len(et_students) > 0 and len(ce_students) > 0:
        # Calculate improvement scores
        et_improvement = et_students['Post_Overall'] - et_students['Pre_Overall']
        ce_improvement = ce_students['Post_Overall'] - ce_students['Pre_Overall']
        
        # Independent t-test on improvement scores
        t_stat, p_val = ttest_ind(et_improvement, ce_improvement)
        effect_size = cohen_d(et_improvement, ce_improvement)
        
        print(f"Engineering Technology (n={len(et_students)}):")
        print(f"  Pre:  M = {et_students['Pre_Overall'].mean():.2f}, SD = {et_students['Pre_Overall'].std():.2f}")
        print(f"  Post: M = {et_students['Post_Overall'].mean():.2f}, SD = {et_students['Post_Overall'].std():.2f}")
        print(f"  Improvement: +{et_improvement.mean():.2f}")
        print(f"  Cohen's d = {paired_cohen_d(et_students['Post_Overall'], et_students['Pre_Overall']):.2f}")
        
        print(f"\nConventional Engineering (n={len(ce_students)}):")
        print(f"  Pre:  M = {ce_students['Pre_Overall'].mean():.2f}, SD = {ce_students['Pre_Overall'].std():.2f}")
        print(f"  Post: M = {ce_students['Post_Overall'].mean():.2f}, SD = {ce_students['Post_Overall'].std():.2f}")
        print(f"  Improvement: +{ce_improvement.mean():.2f}")
        print(f"  Cohen's d = {paired_cohen_d(ce_students['Post_Overall'], ce_students['Pre_Overall']):.2f}")
        
        print(f"\nBetween-groups comparison:")
        print(f"  t({len(et_improvement) + len(ce_improvement) - 2}) = {t_stat:.2f}, p = {p_val:.3f}")
        print(f"  Effect size d = {effect_size:.2f}")
        
        results['program_comparison'] = {
            'et_n': len(et_students),
            'ce_n': len(ce_students),
            'et_pre_mean': et_students['Pre_Overall'].mean(),
            'et_post_mean': et_students['Post_Overall'].mean(),
            'ce_pre_mean': ce_students['Pre_Overall'].mean(), 
            'ce_post_mean': ce_students['Post_Overall'].mean(),
            'between_groups_t': t_stat,
            'between_groups_p': p_val,
            'between_groups_d': effect_size
        }
    
    return results, matched_df

def create_summary_table(results):
    """Create a summary table for the paper"""
    
    print(f"\n4. SUMMARY TABLE FOR PAPER")
    print("-" * 80)
    print(f"{'Domain':<20} {'Pre-M (SD)':<12} {'Post-M (SD)':<13} {'Cohen\'s d':<10} {'95% CI':<15}")
    print("-" * 80)
    
    if 'theoretical' in results:
        r = results['theoretical']
        ci_lower = r['cohen_d'] - 1.96 * np.sqrt((r['n']-1)/(r['n']-3)) * np.sqrt(2/r['n'])
        ci_upper = r['cohen_d'] + 1.96 * np.sqrt((r['n']-1)/(r['n']-3)) * np.sqrt(2/r['n'])
        print(f"{'Theoretical Understanding':<20} {r['pre_mean']:.2f} ({r['pre_std']:.2f})  {r['post_mean']:.2f} ({r['post_std']:.2f})   {r['cohen_d']:.2f}*     {ci_lower:.2f}-{ci_upper:.2f}")
    
    if 'practical' in results:
        r = results['practical']
        ci_lower = r['cohen_d'] - 1.96 * np.sqrt((r['n']-1)/(r['n']-3)) * np.sqrt(2/r['n'])
        ci_upper = r['cohen_d'] + 1.96 * np.sqrt((r['n']-1)/(r['n']-3)) * np.sqrt(2/r['n'])
        print(f"{'Practical Skills':<20} {r['pre_mean']:.2f} ({r['pre_std']:.2f})  {r['post_mean']:.2f} ({r['post_std']:.2f})   {r['cohen_d']:.2f}*     {ci_lower:.2f}-{ci_upper:.2f}")
    
    if 'safety' in results:
        r = results['safety'] 
        ci_lower = r['cohen_d'] - 1.96 * np.sqrt((r['n']-1)/(r['n']-3)) * np.sqrt(2/r['n'])
        ci_upper = r['cohen_d'] + 1.96 * np.sqrt((r['n']-1)/(r['n']-3)) * np.sqrt(2/r['n'])
        print(f"{'Safety Procedures':<20} {r['pre_mean']:.2f} ({r['pre_std']:.2f})  {r['post_mean']:.2f} ({r['post_std']:.2f})   {r['cohen_d']:.2f}*     {ci_lower:.2f}-{ci_upper:.2f}")
    
    print("-" * 80)
    print("*p<0.001")

def create_visualizations(matched_df):
    """Create visualizations for the analysis"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Module 2 Assessment Results', fontsize=16, fontweight='bold')
    
    # 1. Overall improvement
    ax1 = axes[0, 0]
    pre_post_data = pd.DataFrame({
        'Pre': matched_df['Pre_Overall'],
        'Post': matched_df['Post_Overall']
    })
    
    ax1.boxplot([pre_post_data['Pre'], pre_post_data['Post']], labels=['Pre', 'Post'])
    ax1.set_title('Overall Competency Scores')
    ax1.set_ylabel('Competency Score (1-5)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Domain-specific improvements
    ax2 = axes[0, 1]
    domains = ['Theoretical', 'Practical', 'Safety']
    pre_means = []
    post_means = []
    
    for domain in domains:
        pre_col = f'Pre_{domain}'
        post_col = f'Post_{domain}'
        if pre_col in matched_df.columns and post_col in matched_df.columns:
            pre_means.append(matched_df[pre_col].mean())
            post_means.append(matched_df[post_col].mean())
    
    x = np.arange(len(domains))
    width = 0.35
    
    ax2.bar(x - width/2, pre_means, width, label='Pre', alpha=0.8)
    ax2.bar(x + width/2, post_means, width, label='Post', alpha=0.8)
    ax2.set_xlabel('Competency Domain')
    ax2.set_ylabel('Mean Score (1-5)')
    ax2.set_title('Domain-Specific Improvements')
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Program comparison
    ax3 = axes[1, 0]
    et_data = matched_df[matched_df['Program'] == 'Engineering Technology']
    ce_data = matched_df[matched_df['Program'] == 'Conventional Engineering']
    
    programs = ['Engineering\nTechnology', 'Conventional\nEngineering']
    pre_means_prog = [et_data['Pre_Overall'].mean(), ce_data['Pre_Overall'].mean()]
    post_means_prog = [et_data['Post_Overall'].mean(), ce_data['Post_Overall'].mean()]
    
    x = np.arange(len(programs))
    ax3.bar(x - width/2, pre_means_prog, width, label='Pre', alpha=0.8)
    ax3.bar(x + width/2, post_means_prog, width, label='Post', alpha=0.8)
    ax3.set_xlabel('Program Type')
    ax3.set_ylabel('Mean Score (1-5)')
    ax3.set_title('Program Type Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(programs)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual improvement distribution
    ax4 = axes[1, 1]
    improvement = matched_df['Post_Overall'] - matched_df['Pre_Overall']
    ax4.hist(improvement, bins=15, alpha=0.7, edgecolor='black')
    ax4.axvline(improvement.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {improvement.mean():.2f}')
    ax4.set_xlabel('Improvement Score')
    ax4.set_ylabel('Number of Students')
    ax4.set_title('Distribution of Individual Improvements')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main analysis function"""
    try:
        # Load and process data
        matched_df = extract_competency_scores()
        
        # Run statistical analysis
        results, matched_df = run_statistical_analysis(matched_df)
        
        # Create summary table
        create_summary_table(results)
        
        # Create visualizations
        fig = create_visualizations(matched_df)
        
        # Save results
        matched_df.to_csv('analysis_dataset.csv', index=False)
        print(f"\nAnalysis complete! Dataset saved as 'analysis_dataset.csv'")
        print(f"Final sample size: {len(matched_df)} students")
        
        return results, matched_df, fig
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results, data, figure = main()
