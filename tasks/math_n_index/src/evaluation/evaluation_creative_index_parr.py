import os
import re
import nltk
import json
import time
import requests
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from dataclasses import dataclass
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from transformers import AutoTokenizer
import concurrent.futures

md = MosesDetokenizer(lang='en')
API_URL = 'https://api.infini-gram.io/'

# We assume the HuggingFace token is set via an env variable.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)

@dataclass
class Document:
    doc_id: str
    tokens: List[str]

@dataclass
class Span:
    start_index: int
    end_index: int
    span_text: str
    occurrence: int

class Hypothesis:
    def __init__(self, target_doc: Document, min_ngram: int) -> None:
        self.target_doc = target_doc
        self.min_ngram = min_ngram
        self.spans = []
        self.finished = False

    def add_span(self, new_span: Span) -> None:
        self.spans.append(new_span)
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def replace_span(self, new_span: Span) -> None:
        self.spans = self.spans[:-1] + [new_span]
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def get_score(self) -> float:
        if not self.spans:
            return 0
        progress_len = self.spans[-1].end_index if not self.finished else len(self.target_doc.tokens)
        flags = [False] * progress_len
        for span in self.spans:
            span_length = span.end_index - span.start_index
            flags[span.start_index: span.end_index] = [True] * span_length
        coverage = sum(flags) / len(flags) if len(flags) > 0 else 0
        return coverage

    def format_span(self) -> str:
        return ' | '.join(s.span_text for s in self.spans)

    def __hash__(self):
        return hash(self.format_span())

    def __eq__(self, other):
        if isinstance(other, Hypothesis):
            return self.format_span() == other.format_span()
        return NotImplemented

    def get_avg_span_len(self) -> float:
        if not self.spans:
            return 0
        span_lengths = [s.end_index - s.start_index for s in self.spans]
        return sum(span_lengths) / len(span_lengths)

    def export_json(self) -> dict:
        matched_spans = [
            {
                'start_index': s.start_index,
                'end_index': s.end_index,
                'span_text': s.span_text,
                'occurrence': s.occurrence
            }
            for s in self.spans
        ]
        return {
            'matched_spans': matched_spans,
            'coverage': self.get_score(),
            'avg_span_len': self.get_avg_span_len(),
        }

def find_exact_match(detokenize: Callable, doc: Document, min_ngram: int):
    hypothesis = Hypothesis(doc, min_ngram)

    first_pointer, second_pointer = 0, min_ngram
    while second_pointer <= len(doc.tokens):
        span_text = detokenize(doc.tokens[first_pointer: second_pointer])
        request_data = {
            'corpus': 'v4_dolma-v1_7_llama',
            'engine': 'c++',
            'query_type': 'count',
            'query': span_text,
        }

        max_retries = 5
        retry_delay = 0.2
        search_result = {}  # Initialize to avoid NameError

        for attempt in range(max_retries):
            try:
                time.sleep(0.1)  # throttle request
                response = requests.post(API_URL, json=request_data)
                response.raise_for_status()  # raises HTTPError for bad status
                search_result = response.json()
                print(f"attempt {attempt} successful")
                break
            except (requests.RequestException, ValueError) as e:
                print(f"[Retry {attempt + 1}/{max_retries}] Error: {e}")
                if attempt == max_retries - 1:
                    print(f"[Failed] Span '{span_text}' skipped after {max_retries} attempts.")
                else:
                    time.sleep(retry_delay)
        # search_result = requests.post(API_URL, json=request_data).json()
        occurrence = 0 if 'count' not in search_result else search_result['count']

        if occurrence:
            matched_span = Span(start_index=first_pointer,
                                end_index=second_pointer,
                                span_text=span_text,
                                occurrence=occurrence)
            if not hypothesis.spans:
                hypothesis.add_span(matched_span)
            else:
                last_span = hypothesis.spans[-1]
                if matched_span.start_index <= last_span.start_index and last_span.end_index <= matched_span.end_index:
                    hypothesis.replace_span(matched_span)
                else:
                    hypothesis.add_span(matched_span)
            second_pointer += 1
            '''
            print("***************************************************************************************************")
            print(hypothesis.format_span())
            print(f'score: {hypothesis.get_score():.4f}  avg_span_length: {hypothesis.get_avg_span_len()}')
            print("***************************************************************************************************")
            '''
        else:
            if second_pointer - first_pointer > min_ngram:
                first_pointer += 1
            elif second_pointer - first_pointer == min_ngram:
                first_pointer += 1
                second_pointer += 1
            else:
                raise ValueError

    hypothesis.finished = True
    return hypothesis.export_json()

def process_document(t_doc, t_idx, tokenize_func, detokenize, min_ngram):
    """
    Runs find_exact_match on a single doc. Returns (t_idx, updated_doc or None).
    We return t_idx so we can reconstruct the original order later.
    """
    tokenized_text = tokenize_func(unidecode(t_doc['response']))
    tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)

    if len(tgt_doc.tokens) <= min_ngram:
        return (t_idx, None)

    output = find_exact_match(detokenize, tgt_doc, min_ngram)
    t_doc.update(output)
    return (t_idx, t_doc)

def extract_doc_index(doc_id):
    match = re.search(r"_(\d+)$", doc_id)
    return int(match.group(1)) if match else -1

def dj_search(data_path, output_file, min_ngram, subset=100, lm_tokenizer=False, num_workers=8):
    data = json.load(open(data_path, 'r'))[:subset]

    if not lm_tokenizer:
        tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
        detokenize = lambda x: md.detokenize(x)
    else:
        tokenize_func = lambda x: tokenizer.tokenize(x)
        detokenize = lambda x: tokenizer.decode(tokenizer.convert_tokens_to_ids(x))

    # Load previous outputs if available.
    outputs = []
    if os.path.isfile(output_file):
        try:
            outputs = json.load(open(output_file, 'r'))
        except Exception:
            outputs = []
        # Resume logic: skip the first len(outputs) items from data.
        data = data[len(outputs):]
        print(f"Resume from previous output file with {len(outputs)} items.")

    # We'll store parallel results in an array so we can replicate
    # the exact same 'for' iteration order as the original script.
    results_array = [None] * len(data)

    # Run concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for t_idx, t_doc in enumerate(data):
            future = executor.submit(process_document, t_doc, t_idx, tokenize_func, detokenize, min_ngram)
            future_to_idx[future] = t_idx

        # Collect results as they finish
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=len(future_to_idx),
                           desc="Processing docs"):
            idx = future_to_idx[future]
            result_idx, result_doc = future.result()
            results_array[idx] = result_doc

    # Now replicate the original loop's logic for coverage, partial writes, etc.
    # We do it in ascending order of t_idx, exactly like the original script.
    for t_idx, rdoc in tqdm(enumerate(results_array), desc='Finalize docs', total=len(results_array)):
        if rdoc is None:
            continue

        # The original script appends doc to outputs, then prints coverage, then writes the file.
        outputs.append(rdoc)

        avg_coverage = np.average([x['coverage'] for x in outputs])
        std_coverage = np.std([x['coverage'] for x in outputs])
        avg_len = np.average([x['avg_span_len'] for x in outputs])

        print(f"average {min_ngram}-ngram coverage: {avg_coverage:.3f}, "
              f"std: {std_coverage:.3f}, average length: {avg_len}")

        # Write partial results after each doc
    # Final sort by extracted index from doc_id
    outputs.sort(key=lambda x: extract_doc_index(x.get("doc_id", "")))
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=4)
    print(f"Finished processing. Total items saved: {len(outputs)}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Llama-31-8B-instruct_book',
                        help="Which type of corpus to analyze")
    parser.add_argument('--data', type=str,
                        default='data/outputs/creative_index/Llama-31-8B-instruct/book.json')
    parser.add_argument('--output_dir', type=str,
                        default='data/evaluation/creative_index_evaluations/creative_index_exact/Llama-31-8B-instruct')
    parser.add_argument("--min_ngram", type=int, default=5,
                        help="Minimum n-gram size")
    parser.add_argument("--subset", type=int, default=100,
                        help="Size of example subset to run search on")
    parser.add_argument('--lm_tokenizer', action='store_true',
                        help='Whether to use LM tokenizer instead of whitespace tokenizer')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel threads to use')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.task + '_exact_' + str(args.min_ngram) + '.json')
    dj_search(args.data, args.output_file, args.min_ngram, args.subset, args.lm_tokenizer,  args.num_workers)

if __name__ == '__main__':
    main()
