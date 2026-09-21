"""Score metaphors for originality with a judge. Separate script, separate flags."""
import argparse, csv, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from model_hub import get_client  # noqa: E402

RUBRIC = ("Rate the originality of these metaphors for '{concept}' from 1 to 5. "
          "Reply with the digit only.\n\n{metaphors}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    # EDGE CASE 6: the judge is named by a different flag, and defaults to a
    # model that is not in our registry.
    ap.add_argument("--judge", default="gpt-4o-mini")
    args = ap.parse_args()

    client = get_client(args.judge, max_tokens=8, temperature=0)

    rows = list(csv.DictReader(open(args.infile)))
    scored = []
    for row in rows:
        raw = (client.generate(RUBRIC.format(
            concept=row["concept"], metaphors=row["metaphors"])) or "").strip()
        scored.append({"concept": row["concept"], "metaphors": row["metaphors"],
                       "originality": raw})

    # EDGE CASE 7: the eval writes next to the input, with yet another name.
    out = os.path.join(os.path.dirname(args.infile), "originality_scores.json")
    json.dump(scored, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
