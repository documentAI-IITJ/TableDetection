import math
import contextlib

def tokenize_latex(latex_str):
    """
    Tokenize the LaTeX string.
    Here we use a simple whitespace split.
    """
    return latex_str.strip().split()

def remove_rules(tokens):
    """
    Remove tokens corresponding to \hline, \toprule, \midrule, or \bottomrule.
    """
    rules = {"\\hline", "\\toprule", "\\midrule", "\\bottomrule"}
    return [t for t in tokens if t not in rules]

def longest_common_substring_info(tokens1, tokens2):
    """
    Compute the longest contiguous common substring between two lists of tokens
    using dynamic programming.
    
    Returns a dictionary with:
      - length: Length of the longest common substring.
      - start_idx_tokens1: Starting index in tokens1 (0-indexed).
      - end_idx_tokens1: Ending index in tokens1 (0-indexed, inclusive).
      - start_idx_tokens2: Starting index in tokens2 (0-indexed).
      - end_idx_tokens2: Ending index in tokens2 (0-indexed, inclusive).
      - substring: The matching tokens as a list.
    """
    m = len(tokens1)
    n = len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_idx_tokens1 = 0
    end_idx_tokens2 = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx_tokens1 = i - 1
                    end_idx_tokens2 = j - 1
            else:
                dp[i][j] = 0

    if max_len == 0:
        return {
            "length": 0,
            "start_idx_tokens1": None,
            "end_idx_tokens1": None,
            "start_idx_tokens2": None,
            "end_idx_tokens2": None,
            "substring": []
        }
    
    start_idx_tokens1 = end_idx_tokens1 - max_len + 1
    start_idx_tokens2 = end_idx_tokens2 - max_len + 1
    substring = tokens1[start_idx_tokens1: end_idx_tokens1 + 1]
    
    return {
        "length": max_len,
        "start_idx_tokens1": start_idx_tokens1,
        "end_idx_tokens1": end_idx_tokens1,
        "start_idx_tokens2": start_idx_tokens2,
        "end_idx_tokens2": end_idx_tokens2,
        "substring": substring
    }

def evaluate_metrics(ground_truth, prediction, threshold_ratio=0.95, apply_filter=False):
    """
    Evaluates EA, E95, and similarity percentage for a given pair of ground-truth
    and predicted LaTeX code.
    
    Parameters:
      - ground_truth: Ground truth LaTeX code (string).
      - prediction: Predicted LaTeX code (string).
      - threshold_ratio: Ratio for E95 (default 0.95).
      - apply_filter: If True, removes the rule tokens.
    
    Returns a dictionary with:
      - ea: True if Exact Accuracy (EA) is met.
      - e95: True if E95 is met.
      - similarity_percentage: (Longest common substring length / total ground truth tokens)*100.
      - lcs_info: Information about the longest common substring.
      - gt_tokens: Final ground truth tokens.
      - pred_tokens: Final predicted tokens.
      - threshold: The computed token threshold for E95.
    """
    # Tokenize both strings.
    gt_tokens = tokenize_latex(ground_truth)
    pred_tokens = tokenize_latex(prediction)
    
    if apply_filter:
        gt_tokens = remove_rules(gt_tokens)
        pred_tokens = remove_rules(pred_tokens)
    
    # Exact Accuracy (EA): complete token list match.
    ea = (gt_tokens == pred_tokens)

    pred_len = len(pred_tokens)
    
    # Calculate the threshold (95% of ground truth tokens, rounded up).
    gt_len = len(gt_tokens)
    threshold = math.ceil(threshold_ratio * gt_len) if gt_len > 0 else 0
    
    # Compute longest contiguous common substring.
    lcs_info = longest_common_substring_info(gt_tokens, pred_tokens)
    e95 = (lcs_info["length"] >= threshold)
    
    # Compute similarity percentage.
    similarity_percentage = (lcs_info["length"] / gt_len * 100) if gt_len > 0 else 0.0
    
    filter_status = "Filtered (without rules)" if apply_filter else "Unfiltered (all tokens)"
    print("----- Evaluation ({}) -----".format(filter_status))
    print("Ground Truth Tokens:", gt_tokens)
    print("Prediction Tokens:  ", pred_tokens)
    print("Total Ground Truth Tokens:", gt_len)
    print("Total Prediction Tokens:", pred_len)
    print("Exact Accuracy (EA):", ea)
    print("Longest Contiguous Common Substring Info:")
    print("  - Matching Tokens:", lcs_info["substring"])
    print("  - Length:", lcs_info["length"])
    if lcs_info["length"] > 0:
        print("  - Ground Truth Location: from index {} to {}"
              .format(lcs_info["start_idx_tokens1"], lcs_info["end_idx_tokens1"]))
        print("  - Prediction Location: from index {} to {}"
              .format(lcs_info["start_idx_tokens2"], lcs_info["end_idx_tokens2"]))
    print("Threshold for E95 ({}% of ground truth tokens): {}"
          .format(int(threshold_ratio * 100), threshold))
    print("E95 Satisfied:", e95)
    print("Similarity Percentage: {:.2f}%".format(similarity_percentage))
    print()
    
    return {
        "ea": ea,
        "e95": e95,
        "similarity_percentage": similarity_percentage,
        "lcs_info": lcs_info,
        "gt_tokens": gt_tokens,
        "pred_tokens": pred_tokens,
        "threshold": threshold
    }

def evaluate_all_metrics(ground_truth, prediction, threshold_ratio=0.95):
    """
    For a given pair of ground-truth and predicted LaTeX code, performs both:
      - Unfiltered evaluation (using all tokens)
      - Filtered evaluation (removing \hline, \toprule, \midrule, \\bottomrule)
    
    Returns a dictionary with both sets of results.
    """
    print("======== Unfiltered Evaluation (All Tokens) ========")
    results_unfiltered = evaluate_metrics(ground_truth, prediction, threshold_ratio, apply_filter=False)
    
    print("======== Filtered Evaluation (Without \\hline, \\toprule, \\midrule, \\bottomrule) ========")
    results_filtered = evaluate_metrics(ground_truth, prediction, threshold_ratio, apply_filter=True)
    
    return {"unfiltered": results_unfiltered, "filtered": results_filtered}

def evaluate_all_pairs(ground_truth_file, predicted_file, threshold_ratio=0.95):
    """
    Reads the ground truth and predicted LaTeX code files (one pair per line) and
    computes EA, E95, and similarity percentage (both unfiltered and filtered) for each pair.
    
    The ground truth file should have one LaTeX code per line.
    The predicted file should have one line per prediction, formatted as:
         filename<TAB>latex code
    
    At the end, overall EA, E95, and average similarity percentages (as percentages)
    for all pairs are printed.
    """
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt_lines = [line.strip() for line in f if line.strip()]
    
    with open(predicted_file, 'r', encoding='utf-8') as f:
        pred_lines = [line.strip() for line in f if line.strip()]
    
    if len(gt_lines) != len(pred_lines):
        print("Warning: The number of ground truth lines and predicted lines do not match!")
    
    total = len(gt_lines)
    unfiltered_ea_count = 0
    unfiltered_e95_count = 0
    filtered_ea_count = 0
    filtered_e95_count = 0
    total_similarity_unfiltered = 0.0
    total_similarity_filtered = 0.0
    
    for idx, (gt, pred_line) in enumerate(zip(gt_lines, pred_lines), start=1):
        # Each predicted line is expected to be: "filename<TAB>latex code"
        parts = pred_line.split('\t')
        if len(parts) < 2:
            predicted_latex = pred_line
            filename = "Unknown"
        else:
            filename = parts[0]
            predicted_latex = parts[1]
        
        print(f"\n***** Pair {idx}: Filename: {filename} *****")
        result = evaluate_all_metrics(gt, predicted_latex, threshold_ratio)
        
        if result["unfiltered"]["ea"]:
            unfiltered_ea_count += 1
        if result["unfiltered"]["e95"]:
            unfiltered_e95_count += 1
        if result["filtered"]["ea"]:
            filtered_ea_count += 1
        if result["filtered"]["e95"]:
            filtered_e95_count += 1
        
        total_similarity_unfiltered += result["unfiltered"]["similarity_percentage"]
        total_similarity_filtered += result["filtered"]["similarity_percentage"]
    
    overall_unfiltered_ea = unfiltered_ea_count / total * 100
    overall_unfiltered_e95 = unfiltered_e95_count / total * 100
    overall_filtered_ea = filtered_ea_count / total * 100
    overall_filtered_e95 = filtered_e95_count / total * 100
    average_similarity_unfiltered = total_similarity_unfiltered / total
    average_similarity_filtered = total_similarity_filtered / total
    
    print("\n==================== Overall Metrics ====================")
    print("Total pairs evaluated:", total)
    print("\n-- Unfiltered Evaluation (All Tokens) --")
    print("Overall Exact Accuracy (EA): {:.2f}%".format(overall_unfiltered_ea))
    print("Overall E95 Accuracy: {:.2f}%".format(overall_unfiltered_e95))
    print("Average Similarity: {:.2f}%".format(average_similarity_unfiltered))
    print("\n-- Filtered Evaluation (Without \\hline, \\toprule, \\midrule, \\bottomrule) --")
    print("Overall Exact Accuracy (EA): {:.2f}%".format(overall_filtered_ea))
    print("Overall E95 Accuracy: {:.2f}%".format(overall_filtered_e95))
    print("Average Similarity: {:.2f}%".format(average_similarity_filtered))

if __name__ == '__main__':
    ground_truth_file = '/../../mtp/qwen/tgt-test.txt'
    # InternVL_2.5_4B
    # predicted_file = '/../../unitable/FT_internvl_HTML_TableImage_lora_llm256_vit128_full_FAT_finetuned.txt'

    # predicted_file = '/../../unitable/FT_internvl_onlyTableImage_lora_llm256_vit128_FAT_6epochs_complete_row_col.txt'
    predicted_file = '/iitjhome/m23cse017/unitable/ZS_internvl_onlyTableImage_descriptive_prompt_TSR_format_change.txt'

    # output_filename = "/../../mtp/qwen/evaluation_output_internvl2_5_4B_onlyTableImage_lora_llm256_vit128_FAT_6epochs_complete_row_col.txt"
    output_filename = "/../../mtp/qwen/evaluation_output_internvl2_5_4B_onlyTableImage_ZS_descriptive_prompt.txt"

    with open(output_filename, "w", encoding="utf-8") as out_file:
        with contextlib.redirect_stdout(out_file):
            evaluate_all_pairs(ground_truth_file, predicted_file, threshold_ratio=0.95)
    
    # Also print a short message to the console indicating that the evaluation is complete.
    print(f"Evaluation complete. Detailed output written to {output_filename}")