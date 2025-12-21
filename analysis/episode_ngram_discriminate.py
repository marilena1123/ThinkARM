#!/usr/bin/env python3
"""
N-gram Analysis Tool for Comparing Model Response Patterns

This tool identifies important n-gram patterns between two kinds of model responses
by analyzing frequency distributions, statistical significance, and pattern characteristics.
"""

import json
import argparse
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set

# Optional dependency for numerical operations
try:
    # import numpy as np
    # NUMPY_AVAILABLE = True
    NUMPY_AVAILABLE = False
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Falling back to approximate MI.")


class NGramAnalyzer:
    """Analyzes n-gram patterns between two sets of sequences."""
    
    def __init__(self, sequences1: List[str], sequences2: List[str], label1: str = "Dataset1", label2: str = "Dataset2"):
        self.sequences1 = sequences1
        self.sequences2 = sequences2
        self.label1 = label1
        self.label2 = label2
        self.variable_ngrams = {}  # For variable-length analysis

    
    def extract_variable_length_ngrams(self, sequence: str, min_len: int = 1, max_len: int = 10) -> List[Tuple[str, int]]:
        """Extract n-grams of variable lengths from a sequence.
        
        Returns:
            List of tuples (ngram, length) for all n-grams from min_len to max_len
        """
        ngrams = []
        for n in range(min_len, min(max_len + 1, len(sequence) + 1)):
            if len(sequence) >= n:
                for i in range(len(sequence) - n + 1):
                    ngram = sequence[i:i+n]
                    ngrams.append((ngram, n))
        return ngrams
    
    def count_variable_length_ngrams(self, min_len: int = 1, max_len: int = 10) -> None:
        """Count variable-length n-grams in both datasets."""
        self.variable_ngrams = {
            'dataset1': Counter(),
            'dataset2': Counter(),
            'lengths': {}
        }
        
        # Count n-grams in dataset 1
        for seq in self.sequences1:
            ngrams = self.extract_variable_length_ngrams(seq, min_len, max_len)
            for ngram, length in ngrams:
                self.variable_ngrams['dataset1'][ngram] += 1
                if ngram not in self.variable_ngrams['lengths']:
                    self.variable_ngrams['lengths'][ngram] = length
        
        # Count n-grams in dataset 2
        for seq in self.sequences2:
            ngrams = self.extract_variable_length_ngrams(seq, min_len, max_len)
            for ngram, length in ngrams:
                self.variable_ngrams['dataset2'][ngram] += 1
                if ngram not in self.variable_ngrams['lengths']:
                    self.variable_ngrams['lengths'][ngram] = length
    
    
    
    def _mutual_information_from_contingency(self, contingency: 'np.ndarray') -> float:
        """Calculate mutual information from contingency table."""
        total = contingency.sum()
        if total == 0:
            return 0.0
        
        mi = 0.0
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                if contingency[i, j] > 0:
                    p_xy = contingency[i, j] / total
                    p_x = contingency[i, :].sum() / total
                    p_y = contingency[:, j].sum() / total
                    if p_x > 0 and p_y > 0:
                        mi += p_xy * math.log2(p_xy / (p_x * p_y))
        
        return mi
    
    
    
    
    def calculate_length_penalty_score(self, ngram: str, base_score: float, length_bonus: float = 0.1) -> float:
        """Calculate a score that encourages longer n-grams.
        
        Args:
            ngram: The n-gram string
            base_score: The base statistical score (e.g., mutual information)
            length_bonus: Bonus factor for length (default: 0.1)
        
        Returns:
            Adjusted score that favors longer n-grams
        """
        length = len(ngram)
        # Apply a logarithmic bonus for length to encourage longer n-grams
        # but not too aggressively
        length_adjustment = 1 + length_bonus * math.log(length + 1)
        return base_score * length_adjustment
    
    def find_dynamic_patterns(self, min_len: int = 1, max_len: int = 10, top_k: int = 20, 
                            min_count: int = 3, length_bonus: float = 0.1) -> Dict[str, List[Tuple[str, float, int]]]:
        """Find the most informative n-grams using dynamic length selection.
        
        This method encourages longer n-grams by applying length bonuses to statistical scores.
        
        Returns:
            Dictionary with different ranking methods, each containing tuples of (ngram, score, length)
        """
        # Count variable-length n-grams
        self.count_variable_length_ngrams(min_len, max_len)
        
        # Filter by minimum count
        filtered_ngrams = set()
        for ngram in set(self.variable_ngrams['dataset1'].keys()) | set(self.variable_ngrams['dataset2'].keys()):
            total_count = (self.variable_ngrams['dataset1'].get(ngram, 0) + 
                          self.variable_ngrams['dataset2'].get(ngram, 0))
            if total_count >= min_count:
                filtered_ngrams.add(ngram)
        
        results = {
            'mutual_information_variable': [],
            'mutual_information_with_length_bonus': []
        }
        
        # Calculate mutual information for variable-length n-grams
        mi_scores = {}
        for ngram in filtered_ngrams:
            count_1_with = sum(1 for seq in self.sequences1 if ngram in seq)
            count_1_without = len(self.sequences1) - count_1_with
            count_2_with = sum(1 for seq in self.sequences2 if ngram in seq)
            count_2_without = len(self.sequences2) - count_2_with
            
            if NUMPY_AVAILABLE:
                contingency = np.array([[count_1_with, count_1_without],
                                      [count_2_with, count_2_without]])
                
                if contingency.sum() > 0:
                    mi = self._mutual_information_from_contingency(contingency)
                    mi_scores[ngram] = mi
                else:
                    mi_scores[ngram] = 0.0
            else:
                # Fallback: use simple frequency-based mutual information
                total_with = count_1_with + count_2_with
                total_without = count_1_without + count_2_without
                total_sequences = len(self.sequences1) + len(self.sequences2)
                
                if total_with > 0 and total_without > 0:
                    p_with = total_with / total_sequences
                    p_without = total_without / total_sequences
                    p1_given_with = count_1_with / total_with
                    p2_given_with = count_2_with / total_with
                    p1_given_without = count_1_without / total_without
                    p2_given_without = count_2_without / total_without
                    
                    mi = 0
                    if p1_given_with > 0:
                        mi += p_with * p1_given_with * math.log2(p1_given_with / (len(self.sequences1) / total_sequences))
                    if p2_given_with > 0:
                        mi += p_with * p2_given_with * math.log2(p2_given_with / (len(self.sequences2) / total_sequences))
                    if p1_given_without > 0:
                        mi += p_without * p1_given_without * math.log2(p1_given_without / (len(self.sequences1) / total_sequences))
                    if p2_given_without > 0:
                        mi += p_without * p2_given_without * math.log2(p2_given_without / (len(self.sequences2) / total_sequences))
                    
                    mi_scores[ngram] = mi
                else:
                    mi_scores[ngram] = 0.0
        
        # Rank by mutual information
        mi_sorted = sorted([(ngram, score) for ngram, score in mi_scores.items() 
                           if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        results['mutual_information_variable'] = [(ngram, score, self.variable_ngrams['lengths'][ngram]) 
                                                 for ngram, score in mi_sorted[:top_k]]
        
        # # Rank by mutual information with length bonus
        # mi_length_sorted = sorted([(ngram, self.calculate_length_penalty_score(ngram, score, length_bonus)) 
        #                           for ngram, score in mi_scores.items() 
        #                           if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        # results['mutual_information_with_length_bonus'] = [(ngram, score, self.variable_ngrams['lengths'][ngram]) 
        #                                                   for ngram, score in mi_length_sorted[:top_k]]
        
        return results
    
    
    
    def print_dynamic_analysis_results(self, min_len: int = 1, max_len: int = 10, top_k: int = 20, 
                                     min_count: int = 3, length_bonus: float = 0.1):
        """Print comprehensive dynamic n-gram analysis results."""
        print(f"\n{'='*70}")
        print(f"DYNAMIC N-GRAM ANALYSIS RESULTS (lengths {min_len}-{max_len})")
        print(f"{'='*70}")
        print(f"Dataset 1 ({self.label1}): {len(self.sequences1)} sequences")
        print(f"Dataset 2 ({self.label2}): {len(self.sequences2)} sequences")
        print(f"Length range: {min_len}-{max_len}")
        print(f"Minimum count threshold: {min_count}")
        print(f"Length bonus factor: {length_bonus}")
        
        results = self.find_dynamic_patterns(min_len, max_len, top_k, min_count, length_bonus)
        
        # Print results for each metric
        for metric, patterns in results.items():
            if not patterns:
                continue
                
            print(f"\n{'-'*60}")
            print(f"TOP PATTERNS BY {metric.upper().replace('_', ' ')}")
            print(f"{'-'*60}")
            
            for i, (ngram, score, length) in enumerate(patterns, 1):
                count1 = self.variable_ngrams['dataset1'].get(ngram, 0)
                count2 = self.variable_ngrams['dataset2'].get(ngram, 0)
                
                # Determine which dataset the pattern characterizes more
                len1 = len(self.sequences1)
                len2 = len(self.sequences2)
                
                # Calculate occurrences per sequence (can be > 100%)
                avg_count1 = count1 / len1 if len1 > 0 else 0
                avg_count2 = count2 / len2 if len2 > 0 else 0
                
                # Calculate sequence coverage (percentage of sequences containing the ngram)
                seq_count1 = sum(1 for seq in self.sequences1 if ngram in seq)
                seq_count2 = sum(1 for seq in self.sequences2 if ngram in seq)
                prop1 = seq_count1 / len1 if len1 > 0 else 0
                prop2 = seq_count2 / len2 if len2 > 0 else 0
                
                if avg_count1 > avg_count2:
                    dominant = f"-> {self.label1}"
                elif avg_count2 > avg_count1:
                    dominant = f"-> {self.label2}"
                else:
                    dominant = "="
                
                print(f"{i:2d}. '{ngram}' (length: {length}, score: {score:.4f}) {dominant}")
                print(f"    {self.label1}: {count1:4d} occurrences (avg {avg_count1:.1f}/seq, {prop1:.1%} of seqs) | {self.label2}: {count2:4d} (avg {avg_count2:.1f}/seq, {prop2:.1%} of seqs)")
        
        # Print summary statistics
        print(f"\n{'-'*60}")
        print("SUMMARY STATISTICS")
        print(f"{'-'*60}")
        
        total_ngrams_1 = sum(self.variable_ngrams['dataset1'].values())
        total_ngrams_2 = sum(self.variable_ngrams['dataset2'].values())
        
        print(f"Total n-grams in {self.label1}: {total_ngrams_1}")
        print(f"Total n-grams in {self.label2}: {total_ngrams_2}")
        print(f"Unique n-grams in {self.label1}: {len(self.variable_ngrams['dataset1'])}")
        print(f"Unique n-grams in {self.label2}: {len(self.variable_ngrams['dataset2'])}")
        
        # Length distribution
        length_dist = {}
        for ngram, length in self.variable_ngrams['lengths'].items():
            length_dist[length] = length_dist.get(length, 0) + 1
        
        print(f"\nLength distribution:")
        for length in sorted(length_dist.keys()):
            print(f"  Length {length}: {length_dist[length]} unique n-grams")
        
        # Overlap statistics
        common_ngrams = set(self.variable_ngrams['dataset1'].keys()) & set(self.variable_ngrams['dataset2'].keys())
        unique_to_1 = set(self.variable_ngrams['dataset1'].keys()) - set(self.variable_ngrams['dataset2'].keys())
        unique_to_2 = set(self.variable_ngrams['dataset2'].keys()) - set(self.variable_ngrams['dataset1'].keys())
        
        print(f"\nOverlap analysis:")
        print(f"Common n-grams: {len(common_ngrams)}")
        print(f"Unique to {self.label1}: {len(unique_to_1)}")
        print(f"Unique to {self.label2}: {len(unique_to_2)}")


def load_sequences_from_json(file_path: str) -> List[str]:
    """Load sequences from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [str(item) for item in data]
        else:
            raise ValueError(f"Expected a list in JSON file, got {type(data)}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def main():
    """Main function to run the n-gram analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze n-gram patterns between two sets of model responses (dynamic length)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis.py file1.json file2.json --min-length 1 --max-length 10 -k 20 --min-count 5
        """
    )
    
    parser.add_argument('file1', help='Path to the first JSON file containing sequences')
    parser.add_argument('file2', help='Path to the second JSON file containing sequences')
    parser.add_argument('-k', '--top-k', type=int, default=15,
                       help='Number of top patterns to show for each metric (default: 15)')
    parser.add_argument('--min-count', type=int, default=2,
                       help='Minimum count threshold for n-grams (default: 2)')
    parser.add_argument('--labels', nargs=2, default=['Dataset1', 'Dataset2'],
                       help='Labels for the two datasets (default: Dataset1 Dataset2)')
    parser.add_argument('--min-length', type=int, default=2,
                       help='Minimum n-gram length for dynamic analysis (default: 2)')
    parser.add_argument('--max-length', type=int, default=3,
                       help='Maximum n-gram length for dynamic analysis (default: 3)')
    parser.add_argument('--length-bonus', type=float, default=0,
                       help='Length bonus factor for encouraging longer n-grams (default: 0)')
    
    args = parser.parse_args()
    
    try:
        # Load sequences
        print("Loading sequences...")
        sequences1 = load_sequences_from_json(args.file1)
        sequences2 = load_sequences_from_json(args.file2)
        
        print(f"Loaded {len(sequences1)} sequences from {args.file1}")
        print(f"Loaded {len(sequences2)} sequences from {args.file2}")
        
        # Initialize analyzer
        analyzer = NGramAnalyzer(sequences1, sequences2, args.labels[0], args.labels[1])
        
        # Always run dynamic n-gram analysis
        analyzer.print_dynamic_analysis_results(
            args.min_length, args.max_length, args.top_k, 
            args.min_count, args.length_bonus
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())