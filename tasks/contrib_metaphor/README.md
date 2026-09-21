# Metaphor Generation Task (MGT)

Ask a model for novel metaphors for an abstract concept; score originality with a
judge.

## Usage

```
cd tasks/contrib_metaphor
python scripts/gen.py --model_name gpt-4o-mini --num-examples 5
python scripts/score.py --infile results/<stamp>/gen_<stamp>.csv
```

Outputs land in `results/<timestamp>/`. The key is read from `~/.mgt_key`.
