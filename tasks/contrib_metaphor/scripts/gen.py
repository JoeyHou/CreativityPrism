"""Generate metaphors. Written the way a lab script actually gets written."""
import argparse, csv, datetime, os, sys

# INTEGRATION NOTE. This originally did `import openai` and read a key from
# ~/.mgt_key. Nothing outside the task could fix that: the runner would have had
# to copy a credential into a second file, which is both a bad idea and against
# this repo's rules. Three lines of model_hub replace it, and in exchange the
# task can be pointed at any registered model -- including a local vLLM one --
# without knowing which provider it is talking to.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from model_hub import get_client  # noqa: E402

PROMPT = ("Write three original metaphors for the concept '{concept}'. "
          "One per line, no numbering, no preamble.")


def main():
    ap = argparse.ArgumentParser()
    # EDGE CASE 2: the limit flag is spelled differently from ours.
    ap.add_argument("--num-examples", type=int, default=5, dest="num_examples")
    ap.add_argument("--model_name", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    # EDGE CASE 3: the input path is relative to the CWD, so the script only
    # works when run from its own directory.
    concepts = [c.strip() for c in open("data/concepts.txt") if c.strip()]
    concepts = concepts[: args.num_examples]

    # EDGE CASE 4: the output directory is named by a timestamp decided at run
    # time, so no caller can predict the path.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", stamp)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "gen_%s.csv" % stamp)

    client = get_client(args.model_name, temperature=args.temperature,
                        max_tokens=200)
    # EDGE CASE 5: CSV, not JSON, and the row has no stable id -- position is
    # the only identifier.
    with open(outfile, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept", "metaphors"])
        for concept in concepts:
            text = client.generate(PROMPT.format(concept=concept))
            w.writerow([concept, (text or "").strip()])
    print("wrote", outfile)


if __name__ == "__main__":
    main()
