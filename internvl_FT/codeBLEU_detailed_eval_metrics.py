import re
import numpy as np
from collections import Counter
import math
import argparse
import os
import json

class LaTeXTableParser:
    """Parser for LaTeX tables without explicit tabular environment tags"""
    
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            self.vocabulary = {
                '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'CELL',
                '\\', '\\hline', '\\hspace', '\\multirow', '\\multicolumn',
                '\\toprule', '\\midrule', '\\bottomrule', 'c', 'l', 'r', '|',
                '{', '}'
            }
        else:
            self.vocabulary = set(vocabulary)
        
        self.file_table_pattern = r'^([\w\d\-\.]+\.jpg)?\s*(\{.*\}.*(?:\\\\.*)*)'
        
    def extract_table_entries(self, file_content):
        """Extract table entries from the file content with possible filenames"""
        entries = []
        
        lines = file_content.strip().split('\n')
        current_entry = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(self.file_table_pattern, line)
            if match:
                if current_entry:
                    entries.append(current_entry)
                
                filename = match.group(1)
                table_code = match.group(2)
                current_entry = {'filename': filename, 'table_code': table_code}
            elif current_entry:
                current_entry['table_code'] += '\n' + line
        
        if current_entry:
            entries.append(current_entry)
            
        return entries
    
    def tokenize_table(self, table_content):
        """Tokenize table content based on the vocabulary"""
        tokens = []
        i = 0
        current_token = ""
        
        while i < len(table_content):
            if table_content[i:i+8] == "\\toprule":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append("\\toprule")
                i += 8
            elif table_content[i:i+8] == "\\midrule":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append("\\midrule")
                i += 8
            elif table_content[i:i+10] == "\\bottomrule":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append("\\bottomrule")
                i += 10
            elif table_content[i:i+6] == "\\hline":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append("\\hline")
                i += 6
            elif table_content[i:i+12] == "\\multicolumn":
                match = re.search(r'\\multicolumn\s*\{\s*(\d+)\s*\}\s*\{\s*([^}]*)\s*\}\s*\{([^}]*)\}', table_content[i:])
                if match:
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    multicolumn_text = match.group(0)
                    tokens.append("\\multicolumn")
                    tokens.append("{")
                    tokens.append(match.group(1)) 
                    tokens.append("}")
                    tokens.append("{")
                    tokens.append(match.group(2)) 
                    tokens.append("}")
                    tokens.append("{")
                
                    content = match.group(3)
                    if content == "CELL":
                        tokens.append("CELL")
                    else:
                        tokens.append(content)
                    tokens.append("}")
                    i += len(multicolumn_text)
                else:
                    current_token += table_content[i]
                    i += 1
            elif table_content[i:i+9] == "\\multirow":
    
                match = re.search(r'\\multirow\s*\{\s*(\d+)\s*\}\s*(\{[^}]*\})?\s*\{([^}]*)\}', table_content[i:])
                if match:
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    multirow_text = match.group(0)
                    tokens.append("\\multirow")
                    tokens.append("{")
                    tokens.append(match.group(1))
                    tokens.append("}")
                    if match.group(2): 
                        tokens.append(match.group(2))
                    tokens.append("{")
                    content = match.group(3)
                    if content == "CELL":
                        tokens.append("CELL")
                    else:
                        tokens.append(content)
                    tokens.append("}")
                    i += len(multirow_text)
                else:
                    current_token += table_content[i]
                    i += 1
            elif table_content[i] == '&':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append('&')
                i += 1
            elif table_content[i:i+2] == '\\\\':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append('\\\\')
                i += 2
            elif table_content[i] == '{' or table_content[i] == '}' or table_content[i] == '|':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(table_content[i])
                i += 1
            elif table_content[i:i+4] == "CELL":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append("CELL")
                i += 4
            elif table_content[i].isdigit():
                if current_token and not current_token.isdigit():
                    tokens.append(current_token)
                    current_token = table_content[i]
                else:
                    current_token += table_content[i]
                i += 1
            elif table_content[i].isspace():
             
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                i += 1
            else:
           
                current_token += table_content[i]
                i += 1
        
     
        if current_token:
            tokens.append(current_token)
        

        return [token for token in tokens if token in self.vocabulary or token.isdigit()]
    
    def get_table_structure(self, table_content):
        """Extract the structure of a table as a list of rows and cells"""
        rows = re.split(r'\\\\', table_content)
        
        structure = []
        for row in rows:
            if not row.strip():
                continue
                
            if re.match(r'\s*\\hline\s*$', row) or re.match(r'\s*\\toprule\s*$', row) or \
               re.match(r'\s*\\midrule\s*$', row) or re.match(r'\s*\\bottomrule\s*$', row):
                structure.append(('hline',))
                continue
                
            cells = re.split(r'&', row)
            row_structure = []
            
            for cell in cells:
                cell = cell.strip()
                
                multicolumn_match = re.search(r'\\multicolumn\s*\{\s*(\d+)\s*\}\s*\{\s*([^}]*)\s*\}\s*\{([^}]*)\}', cell)
                if multicolumn_match:
                    span = int(multicolumn_match.group(1))
                    align = multicolumn_match.group(2)
                    content = multicolumn_match.group(3)
                    row_structure.append(('multicolumn', span, align, content))
                    continue
                    
                multirow_match = re.search(r'\\multirow\s*\{\s*(\d+)\s*\}\s*(\{[^}]*\})?\s*\{([^}]*)\}', cell)
                if multirow_match:
                    span = int(multirow_match.group(1))
                    content = multirow_match.group(3)
                    row_structure.append(('multirow', span, content))
                    continue
                    
                if 'CELL' in cell:
                    row_structure.append(('cell', 'CELL'))
                elif cell:
                    row_structure.append(('cell', cell))
                else:
                    row_structure.append(('cell', ''))
            
            structure.append(tuple(row_structure))
            
        return structure


class LaTeXTableCodeBLEU:
    """
    Implementation of CodeBLEU adapted specifically for LaTeX tables
    """
    
    def __init__(self, vocabulary=None, weights=(0.25, 0.25, 0.25, 0.25)):
        """
        Initialize with component weights
        weights: tuple of floats (n_gram, syntax_match, semantic_match, cell_match)
        """
        self.weights = weights
        self.parser = LaTeXTableParser(vocabulary)
    
    def _calculate_bleu(self, reference_tokens, candidate_tokens, max_ngram=4):
        """Calculate BLEU component"""
        if not reference_tokens or not candidate_tokens:
            return 0.0
            
        # Calculate n-gram precision for n from 1 to max_ngram
        precisions = []
        
        for n in range(1, max_ngram + 1):
            if len(candidate_tokens) < n or len(reference_tokens) < n:
                precisions.append(0.0)
                continue
                
            ref_ngrams = Counter(tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1))
            cand_ngrams = Counter(tuple(candidate_tokens[i:i+n]) for i in range(len(candidate_tokens) - n + 1))
            
            matches = sum((cand_ngrams & ref_ngrams).values())
            total = sum(cand_ngrams.values())
            
            precisions.append(matches / total if total > 0 else 0.0)
        
        if all(p == 0 for p in precisions):
            return 0.0
            
        bp = min(1.0, math.exp(1 - len(reference_tokens) / len(candidate_tokens)) if len(candidate_tokens) > 0 else 0.0)
        
        # Calculate final BLEU score
        valid_precisions = [p for p in precisions if p > 0]
        if valid_precisions:
            bleu = bp * math.exp(sum(math.log(p) for p in valid_precisions) / len(valid_precisions))
        else:
            bleu = 0.0
            
        return bleu
    
    def _calculate_syntax_match(self, reference_structure, candidate_structure):
        """Calculate syntax structure match component"""
        if not reference_structure or not candidate_structure:
            return 0.0
            
        ref_struct_str = str(reference_structure)
        cand_struct_str = str(candidate_structure)
        
        # Use string edit distance as approximation for structure similarity
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein(ref_struct_str, cand_struct_str)
        max_len = max(len(ref_struct_str), len(cand_struct_str))
        
        similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
        return similarity
    
    def _calculate_command_match(self, reference_tokens, candidate_tokens):
        """Calculate semantic match component based on command usage"""
        if not reference_tokens or not candidate_tokens:
            return 0.0
            
        ref_commands = [token for token in reference_tokens if token.startswith('\\') and token != '\\\\']
        cand_commands = [token for token in candidate_tokens if token.startswith('\\') and token != '\\\\']
        
        if not ref_commands and not cand_commands:
            return 1.0  
        
        if not ref_commands or not cand_commands:
            return 0.0  
            
        ref_cmd_freq = Counter(ref_commands)
        cand_cmd_freq = Counter(cand_commands)
        
        # Calculate command usage similarity (Jaccard similarity)
        common_cmds = set(ref_cmd_freq.keys()) & set(cand_cmd_freq.keys())
        all_cmds = set(ref_cmd_freq.keys()) | set(cand_cmd_freq.keys())
        
        jaccard = len(common_cmds) / len(all_cmds)
        
        freq_sim = 0.0
        if common_cmds:
            similarities = []
            for cmd in common_cmds:
                ref_count = ref_cmd_freq[cmd]
                cand_count = cand_cmd_freq[cmd]
                max_count = max(ref_count, cand_count)
                min_count = min(ref_count, cand_count)
                similarities.append(min_count / max_count if max_count > 0 else 1.0)
            freq_sim = sum(similarities) / len(similarities)
        
        return 0.7 * jaccard + 0.3 * freq_sim
    
    def _calculate_cell_pattern_match(self, reference_tokens, candidate_tokens):
        """Calculate cell pattern match - how well the pattern of cells is preserved"""
        if not reference_tokens or not candidate_tokens:
            return 0.0
            
        def simplify_to_cell_pattern(tokens):
            pattern = []
            for token in tokens:
                if token == 'CELL':
                    pattern.append('C')
                elif token == '&':
                    pattern.append('&')
                elif token == '\\\\':
                    pattern.append('R') 
                elif token == '\\hline' or token == '\\toprule' or token == '\\midrule' or token == '\\bottomrule':
                    pattern.append('H') 
            return pattern
        
        ref_pattern = simplify_to_cell_pattern(reference_tokens)
        cand_pattern = simplify_to_cell_pattern(candidate_tokens)
        
        if not ref_pattern or not cand_pattern:
            return 0.0
            
        ref_cells = ref_pattern.count('C')
        cand_cells = cand_pattern.count('C')
        ref_separators = ref_pattern.count('&')
        cand_separators = cand_pattern.count('&')
        ref_rows = ref_pattern.count('R')
        cand_rows = cand_pattern.count('R')
        ref_hlines = ref_pattern.count('H')
        cand_hlines = cand_pattern.count('H')
        
        cell_ratio = min(ref_cells, cand_cells) / max(ref_cells, cand_cells) if max(ref_cells, cand_cells) > 0 else (1.0 if ref_cells == cand_cells else 0.0)
        sep_ratio = min(ref_separators, cand_separators) / max(ref_separators, cand_separators) if max(ref_separators, cand_separators) > 0 else (1.0 if ref_separators == cand_separators else 0.0)
        row_ratio = min(ref_rows, cand_rows) / max(ref_rows, cand_rows) if max(ref_rows, cand_rows) > 0 else (1.0 if ref_rows == cand_rows else 0.0)
        hline_ratio = min(ref_hlines, cand_hlines) / max(ref_hlines, cand_hlines) if max(ref_hlines, cand_hlines) > 0 else (1.0 if ref_hlines == cand_hlines else 0.0)
        
        return (cell_ratio + sep_ratio + row_ratio + hline_ratio) / 4
    
    def match_entries_by_filename(self, ref_entries, cand_entries):
        """Match reference and candidate entries by filename"""
        matched_pairs = []
        cand_dict = {entry['filename']: entry for entry in cand_entries if entry['filename']}
        
        for ref_entry in ref_entries:
            if ref_entry['filename'] and ref_entry['filename'] in cand_dict:
                matched_pairs.append((ref_entry, cand_dict[ref_entry['filename']]))
        
        unmatched_refs = [e for e in ref_entries if not any(p[0] == e for p in matched_pairs)]
        unmatched_cands = [e for e in cand_entries if not any(p[1] == e for p in matched_pairs)]
        
        for i in range(min(len(unmatched_refs), len(unmatched_cands))):
            matched_pairs.append((unmatched_refs[i], unmatched_cands[i]))
        
        return matched_pairs
    
    def compute_by_entry(self, references_file, candidates_file):
        """
        Compute CodeBLEU scores by parsing entries from files and matching them
        
        Args:
            references_file: path to reference file
            candidates_file: path to candidate file
            
        Returns:
            Dictionary with scores and detailed scores by entry
        """
        with open(references_file, 'r', encoding='utf-8') as f:
            references_content = f.read()
        
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates_content = f.read()
        
        ref_entries = self.parser.extract_table_entries(references_content)
        cand_entries = self.parser.extract_table_entries(candidates_content)
        
        matched_pairs = self.match_entries_by_filename(ref_entries, cand_entries)
        
        entry_scores = []
        total_bleu = 0
        total_syntax = 0
        total_command = 0
        total_cell = 0
        
        for ref_entry, cand_entry in matched_pairs:
            ref_tokens = self.parser.tokenize_table(ref_entry['table_code'])
            cand_tokens = self.parser.tokenize_table(cand_entry['table_code'])
            
            ref_structure = self.parser.get_table_structure(ref_entry['table_code'])
            cand_structure = self.parser.get_table_structure(cand_entry['table_code'])
            
            ngram_match = self._calculate_bleu(ref_tokens, cand_tokens)
            syntax_match = self._calculate_syntax_match(ref_structure, cand_structure)
            command_match = self._calculate_command_match(ref_tokens, cand_tokens)
            cell_pattern_match = self._calculate_cell_pattern_match(ref_tokens, cand_tokens)
            
            score = sum(w * c for w, c in zip(self.weights, [ngram_match, syntax_match, command_match, cell_pattern_match]))
            
            entry_score = {
                'filename': cand_entry['filename'] or f"entry_{len(entry_scores)+1}",
                'codebleu': score,
                'ngram_match': ngram_match,
                'syntax_match': syntax_match,
                'command_match': command_match,
                'cell_pattern_match': cell_pattern_match
            }
            
            entry_scores.append(entry_score)
            
            total_bleu += ngram_match
            total_syntax += syntax_match
            total_command += command_match
            total_cell += cell_pattern_match
        
        n_entries = len(matched_pairs)
        if n_entries > 0:
            avg_bleu = total_bleu / n_entries
            avg_syntax = total_syntax / n_entries
            avg_command = total_command / n_entries
            avg_cell = total_cell / n_entries
            overall_score = sum(w * s for w, s in zip(self.weights, [avg_bleu, avg_syntax, avg_command, avg_cell]))
        else:
            avg_bleu = avg_syntax = avg_command = avg_cell = overall_score = 0.0
        
        return {
            'overall': {
                'codebleu': overall_score,
                'ngram_match': avg_bleu,
                'syntax_match': avg_syntax,
                'command_match': avg_command,
                'cell_pattern_match': avg_cell,
                'n_entries': n_entries
            },
            'entries': entry_scores
        }
    
    def compute(self, references, candidate):
        """Legacy method for compatibility"""
        if isinstance(references, str):
            references = [references]
            
        candidate_tokens = self.parser.tokenize_table(candidate)
        candidate_structure = self.parser.get_table_structure(candidate)
        
        max_score = 0
        max_components = [0, 0, 0, 0]
        
        for reference in references:
            reference_tokens = self.parser.tokenize_table(reference)
            reference_structure = self.parser.get_table_structure(reference)
            
            ngram_match = self._calculate_bleu(reference_tokens, candidate_tokens)
            syntax_match = self._calculate_syntax_match(reference_structure, candidate_structure)
            command_match = self._calculate_command_match(reference_tokens, candidate_tokens)
            cell_pattern_match = self._calculate_cell_pattern_match(reference_tokens, candidate_tokens)
            
            components = [ngram_match, syntax_match, command_match, cell_pattern_match]
            
            score = sum(w * c for w, c in zip(self.weights, components))
            
            if score > max_score:
                max_score = score
                max_components = components
                
        return {
            'codebleu': max_score,
            'ngram_match': max_components[0],
            'syntax_match': max_components[1],
            'command_match': max_components[2],
            'cell_pattern_match': max_components[3]
        }


def main():
    parser = argparse.ArgumentParser(description='Calculate CodeBLEU for LaTeX tables')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference LaTeX file')
    parser.add_argument('--candidate', type=str, required=True, help='Path to candidate LaTeX file')
    parser.add_argument('--weights', nargs=4, type=float, default=[0.25, 0.25, 0.25, 0.25],
                        help='Weights for the four components (n-gram, syntax, command, cell-pattern)')
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file')
    parser.add_argument('--vocabulary', type=str, default=None, 
                        help='Custom vocabulary (comma-separated list of tokens)')
    parser.add_argument('--detailed', action='store_true', help='Print detailed scores for each entry')
    
    args = parser.parse_args()
    
    vocabulary = None
    if args.vocabulary:
        vocabulary = [token.strip() for token in args.vocabulary.split(',')]
    else:
        # Default TSR vocabulary
        vocabulary = [
            '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'CELL',
            '\\', '\\hline', '\\hspace', '\\multirow', '\\multicolumn',
            '\\toprule', '\\midrule', '\\bottomrule', 'c', 'l', 'r', '|',
            '{', '}', '\\\\'
        ]
    
    if sum(args.weights) != 1.0:
        print(f"Warning: Weights sum to {sum(args.weights)}, not 1.0. Normalizing...")
        total = sum(args.weights)
        args.weights = [w / total for w in args.weights]
    
    codebleu = LaTeXTableCodeBLEU(vocabulary=vocabulary, weights=tuple(args.weights))
    
    results = codebleu.compute_by_entry(args.reference, args.candidate)
    
    print("\nOverall CodeBLEU Results:")
    print(f"  CodeBLEU: {results['overall']['codebleu']:.4f}")
    print(f"  N-gram Match: {results['overall']['ngram_match']:.4f}")
    print(f"  Syntax Match: {results['overall']['syntax_match']:.4f}")
    print(f"  Command Match: {results['overall']['command_match']:.4f}")
    print(f"  Cell Pattern Match: {results['overall']['cell_pattern_match']:.4f}")
    print(f"  Number of entries: {results['overall']['n_entries']}")
    
    if args.detailed:
        print("\nDetailed Scores by Entry:")
        for entry in results['entries']:
            print(f"\n  {entry['filename']}:")
            print(f"    CodeBLEU: {entry['codebleu']:.4f}")
            print(f"    N-gram Match: {entry['ngram_match']:.4f}")
            print(f"    Syntax Match: {entry['syntax_match']:.4f}")
            print(f"    Command Match: {entry['command_match']:.4f}")
            print(f"    Cell Pattern Match: {entry['cell_pattern_match']:.4f}")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()

# python codeBLEU_detailed_claude.py --reference /iitjhome/m23cse017/mtp/qwen/tgt-test.txt --candidate /iitjhome/m23cse017/unitable/FT_internvl_onlyTableImage_lora_llm256_vit128_FAT_6epochs_complete.txt --output /iitjhome/m23cse017/mtp/qwen/codebleu_results_image_only_FAT.json