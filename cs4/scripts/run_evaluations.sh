#!/bin/bash

# User-defined input paths for the three models to be evaluated
MODEL1_PATH="story_generator/output/cs4_v1/llama3_8b_instr/all_stories.csv"  # Path to the first model's output (e.g., mistral model)
MODEL2_PATH="story_generator/output/cs4_v1/mistral_7b_instr/all_stories.csv"  # Path to the second model's output (e.g., llama model)
MODEL3_PATH="story_generator/output/cs4_v1/qwen_7b_instruct/all_stories.csv"  # Path to the third model's output (e.g., qwen model)

# Labels for the three models (used in graphs and output files)
LABEL1="mistral"  # Label for the first model
LABEL2="llama"  # Label for the second model
LABEL3="qwen"  # Label for the third model

# User-defined input file paths for specific evaluations
# Note: These can be the same as one of the model paths or a different file specific to the evaluation
INPUT_CONS_SATISF="story_generator/output/cs4_v1/llama3_8b_instr/all_stories.csv"  # Input for constraint satisfaction (usually one of the model outputs)
INPUT_DIVERSITY_CALC="story_generator/output/cs4_v1/llama3_8b_instr/all_stories.csv"  # Input for diversity calculation (usually one of the model outputs)
INPUT_COH_VS_CONS="story_generator/output/cs4_v1/llama3_8b_instr/all_stories.csv"  # Input for coherence vs. consistency evaluation (usually one of the model outputs)
# INPUT_JSON_QUC_AND_RCS="path/to/your/input.json"  # Input JSON for QUC and RCS evaluation (uncomment if needed)

# User-defined output directories for each type of evaluation
CONS_SATISF_DIR="cons_satisf"  # Directory for constraint satisfaction results
DIVERSITY_CALC_DIR="diversity_calc"  # Directory for diversity calculation results
PERPLEXITY_DIR="perplexity"  # Directory for perplexity evaluation results
COH_VS_CONS_DIR="coh_vs_cons"  # Directory for coherence vs. consistency results
# QUC_AND_RCS_DIR="quc_and_rcs"  # Directory for QUC and RCS results (uncomment if needed)

# User-defined output file names for each evaluation
CONS_SATISF_RESULTS="consistency_satisfaction_results.csv"  # Results file for constraint satisfaction
CONS_SATISF_GRAPH="consistency_satisfaction_graph.png"  # Graph file for constraint satisfaction
DIVERSITY_RESULTS="diversity_results.csv"  # Results file for diversity calculation
DIVERSITY_GRAPHS="diversity_graphs.png"  # Graph file for diversity calculation
PERPLEXITY_GRAPHS="perplexity_graphs.png"  # Graph file for perplexity evaluation
COH_VS_CONS_GRAPH="coherence_vs_consistency_graph.png"  # Graph file for coherence vs. consistency

# Create output directories (if they don't exist)
mkdir -p output_files/$CONS_SATISF_DIR
mkdir -p output_files/$DIVERSITY_CALC_DIR
mkdir -p output_files/$PERPLEXITY_DIR
mkdir -p output_files/$COH_VS_CONS_DIR
# mkdir -p output_files/$QUC_AND_RCS_DIR  # Uncomment if using QUC and RCS evaluation

# Run the Python script with updated arguments
python3 run_all_evals.py \
    --model1_path "$MODEL1_PATH" \
    --model2_path "$MODEL2_PATH" \
    --model3_path "$MODEL3_PATH" \
    --label1 "$LABEL1" \
    --label2 "$LABEL2" \
    --label3 "$LABEL3" \
    --input_file_path_cons_satisf "$INPUT_CONS_SATISF" \
    --output_file_path_cons_satisf "output_files/$CONS_SATISF_DIR/$CONS_SATISF_RESULTS" \
    --output_file_path_cons_satisf_graph "output_files/$CONS_SATISF_DIR/$CONS_SATISF_GRAPH" \
    --input_path_diversity_calc "$INPUT_DIVERSITY_CALC" \
    --output_path_diversity_calc "output_files/$DIVERSITY_CALC_DIR/$DIVERSITY_RESULTS" \
    --output_path_diversity_graphs "output_files/$DIVERSITY_CALC_DIR/$DIVERSITY_GRAPHS" \
    --output_path_perp_graphs "output_files/$PERPLEXITY_DIR/$PERPLEXITY_GRAPHS" \
    --input_path_coh_vs_cons_graph "$INPUT_COH_VS_CONS" \
    --output_path_coh_vs_cons_graph "output_files/$COH_VS_CONS_DIR/$COH_VS_CONS_GRAPH"\
    --use_vllm_cons_satisf 
    # --input_json_quc_and_rcs "$INPUT_JSON_QUC_AND_RCS" \
    # --output_dir_quc_and_rcs "output_files/$QUC_AND_RCS_DIR"