import json
import numpy as np
from pathlib import Path
import zipfile
from typing import Dict, List
import matplotlib.pyplot as plt

def load_attention_matrices_from_zip(zip_path: str) -> List[np.ndarray]:
    """Load all attention matrices from a zip file."""
    matrices = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith('.json'):
                with zip_ref.open(filename) as f:
                    data = json.load(f)
                    # Extract the attention matrix
                    attn_matrix = np.array(data['mean_attention_final_step'][0])
                    matrices.append(attn_matrix)
    
    return matrices

def estimate_input_length(matrix: np.ndarray) -> int:
    """
    Estimate where input ends and generation begins using multiple heuristics.
    """
    seq_len = len(matrix)
    
    # Method 1: Look for causal attention pattern (lower triangular dominance)
    causal_scores = []
    for i in range(max(10, int(seq_len * 0.2)), min(seq_len - 10, int(seq_len * 0.8))):
        # Ratio of attention to past vs future
        attn_to_past = matrix[i, :i].sum()
        attn_to_future = matrix[i, i+1:].sum()
        
        if attn_to_past + attn_to_future > 0:
            causal_ratio = attn_to_past / (attn_to_past + attn_to_future)
            causal_scores.append((i, causal_ratio))
    
    # Find where causal ratio jumps above 0.8
    for i, ratio in causal_scores:
        if ratio > 0.8:
            return i
    
    # Method 2: Look for drop in attention diversity
    attention_spreads = []
    for i in range(seq_len):
        # Standard deviation of attention weights (how spread out)
        spread = np.std(matrix[i])
        attention_spreads.append(spread)
    
    # Find where spread changes significantly
    if len(attention_spreads) > 20:
        window = 10
        for i in range(window, len(attention_spreads) - window):
            before = np.mean(attention_spreads[max(0, i-window):i])
            after = np.mean(attention_spreads[i:min(len(attention_spreads), i+window)])
            if before > after * 1.5:  # Significant drop
                return i
    
    # Method 3: Conservative fallback - use fixed percentage
    # Most prompts are 30-50% of total sequence
    return int(seq_len * 0.45)

def compute_context_grounding_score(matrix: np.ndarray) -> float:
    """
    Higher CGS = more attention to input context = less hallucination.
    
    CGS = (average attention to input tokens) / (average attention to generated tokens)
    
    Uses robust calculation to handle edge cases.
    """
    seq_len = len(matrix)
    
    if seq_len < 2:
        return 0.0
    
    input_len = estimate_input_length(matrix)
    
    # Ensure we have both input and generated portions
    if input_len <= 0 or input_len >= seq_len:
        input_len = max(1, int(seq_len * 0.45))
    
    # Extract relevant portions with bounds checking
    input_slice = matrix[:, :input_len]
    generated_slice = matrix[:, input_len:]
    
    # Check for empty slices
    if input_slice.size == 0 or generated_slice.size == 0:
        return 0.0
    
    # Average attention to input tokens
    attn_to_input = np.nanmean(input_slice)
    
    # Average attention to generated tokens  
    attn_to_generated = np.nanmean(generated_slice)
    
    # Handle edge cases
    if np.isnan(attn_to_input) or np.isnan(attn_to_generated):
        return 0.0
    
    if attn_to_generated < 1e-6:  # Avoid division by near-zero
        return 0.0
    
    cgs = attn_to_input / attn_to_generated
    
    # Sanity check: CGS shouldn't be extreme
    if cgs > 100 or cgs < 0:
        return 0.0
    
    return cgs

def compute_attention_entropy(matrix: np.ndarray) -> float:
    """
    Lower entropy = more focused attention = less hallucination.
    
    Entropy measures how uniform the attention distribution is.
    """
    entropies = []
    
    for row in matrix:
        # Compute Shannon entropy for this row
        # Filter out zeros to avoid log(0)
        probs = row[row > 0]
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)
    
    return np.mean(entropies)

def compute_attention_concentration(matrix: np.ndarray) -> float:
    """
    Higher concentration = attention focused on fewer tokens = less hallucination.
    
    Measures the average maximum attention weight per position.
    """
    max_weights = np.max(matrix, axis=1)
    return np.mean(max_weights)

def analyze_strategy(zip_path: str, strategy_name: str) -> Dict:
    """Analyze all matrices for one strategy."""
    print(f"\nAnalyzing {strategy_name}...")
    
    matrices = load_attention_matrices_from_zip(zip_path)
    print(f"  Loaded {len(matrices)} attention matrices")
    
    cgs_scores = []
    entropy_scores = []
    concentration_scores = []
    
    for i, matrix in enumerate(matrices):
        cgs = compute_context_grounding_score(matrix)
        entropy = compute_attention_entropy(matrix)
        concentration = compute_attention_concentration(matrix)
        
        # Filter out invalid scores
        if not np.isnan(cgs) and not np.isinf(cgs) and cgs > 0:
            cgs_scores.append(cgs)
        
        if not np.isnan(entropy) and not np.isinf(entropy):
            entropy_scores.append(entropy)
            
        if not np.isnan(concentration) and not np.isinf(concentration):
            concentration_scores.append(concentration)
    
    print(f"  Valid CGS scores: {len(cgs_scores)}/{len(matrices)}")
    
    results = {
        'strategy': strategy_name,
        'num_matrices': len(matrices),
        'cgs_mean': np.mean(cgs_scores) if len(cgs_scores) > 0 else 0.0,
        'cgs_std': np.std(cgs_scores) if len(cgs_scores) > 0 else 0.0,
        'entropy_mean': np.mean(entropy_scores) if len(entropy_scores) > 0 else 0.0,
        'entropy_std': np.std(entropy_scores) if len(entropy_scores) > 0 else 0.0,
        'concentration_mean': np.mean(concentration_scores) if len(concentration_scores) > 0 else 0.0,
        'concentration_std': np.std(concentration_scores) if len(concentration_scores) > 0 else 0.0,
        'cgs_scores': cgs_scores,
        'entropy_scores': entropy_scores,
        'concentration_scores': concentration_scores
    }
    
    return results

def print_comparison(all_results: List[Dict]):
    """Print comparison table of all strategies."""
    print("\n" + "="*80)
    print("HALLUCINATION MITIGATION COMPARISON")
    print("="*80)
    print(f"\n{'Strategy':<20} {'CGS (↑)':<15} {'Entropy (↓)':<15} {'Concentration':<15}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['strategy']:<20} "
              f"{result['cgs_mean']:>6.3f} ± {result['cgs_std']:<5.3f} "
              f"{result['entropy_mean']:>6.3f} ± {result['entropy_std']:<5.3f} "
              f"{result['concentration_mean']:>6.3f} ± {result['concentration_std']:<5.3f}")
    
    print("\n" + "="*80)
    print("RANKINGS (Best to Worst)")
    print("="*80)
    
    # Rank by CGS (higher is better)
    cgs_sorted = sorted(all_results, key=lambda x: x['cgs_mean'], reverse=True)
    print("\nBy Context Grounding Score (higher = less hallucination):")
    for i, result in enumerate(cgs_sorted, 1):
        print(f"  {i}. {result['strategy']}: {result['cgs_mean']:.3f}")
    
    # Rank by Entropy (lower is better)
    entropy_sorted = sorted(all_results, key=lambda x: x['entropy_mean'])
    print("\nBy Attention Entropy (lower = more focused, less hallucination):")
    for i, result in enumerate(entropy_sorted, 1):
        print(f"  {i}. {result['strategy']}: {result['entropy_mean']:.3f}")
    
    print("\n" + "="*80)

def plot_results(all_results: List[Dict]):
    """Create visualization of results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    strategies = [r['strategy'] for r in all_results]
    cgs_means = [r['cgs_mean'] for r in all_results]
    cgs_stds = [r['cgs_std'] for r in all_results]
    entropy_means = [r['entropy_mean'] for r in all_results]
    entropy_stds = [r['entropy_std'] for r in all_results]
    concentration_means = [r['concentration_mean'] for r in all_results]
    
    # CGS comparison
    axes[0].bar(strategies, cgs_means, yerr=cgs_stds, capsize=5, color='green', alpha=0.7)
    axes[0].set_title('Context Grounding Score\n(Higher = Better)', fontweight='bold')
    axes[0].set_ylabel('CGS')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Entropy comparison
    axes[1].bar(strategies, entropy_means, yerr=entropy_stds, capsize=5, color='blue', alpha=0.7)
    axes[1].set_title('Attention Entropy\n(Lower = Better)', fontweight='bold')
    axes[1].set_ylabel('Entropy')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Concentration comparison
    axes[2].bar(strategies, concentration_means, color='orange', alpha=0.7)
    axes[2].set_title('Attention Concentration', fontweight='bold')
    axes[2].set_ylabel('Concentration')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'attention_analysis_results.png'")
    plt.show()

def main():
    """Main analysis function."""
    
    # Define your zip file paths
    strategies = {
        'baseline': 'Attention_Matrices/attention_matrices.zip',
        'cad_attn': 'Attention_Matrices/cad_attn.zip',
        'kg_attention': 'Attention_Matrices/kg_attention (1).zip'
    }
    
    # Analyze each strategy
    all_results = []
    for name, zip_path in strategies.items():
        results = analyze_strategy(zip_path, name)
        all_results.append(results)
    
    # Print comparison
    print_comparison(all_results)
    
    # Plot results
    plot_results(all_results)
    
    # Determine best strategy
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    best_cgs = max(all_results, key=lambda x: x['cgs_mean'])
    best_entropy = min(all_results, key=lambda x: x['entropy_mean'])
    
    print(f"\nBest by Context Grounding: {best_cgs['strategy']}")
    print(f"Best by Attention Focus: {best_entropy['strategy']}")
    
    if best_cgs['strategy'] == best_entropy['strategy']:
        print(f"\n✓ WINNER: {best_cgs['strategy']} performs best on both metrics!")
    else:
        print(f"\n→ Consider: {best_cgs['strategy']} for better context grounding")
        print(f"→ Consider: {best_entropy['strategy']} for more focused attention")

if __name__ == "__main__":
    main()
