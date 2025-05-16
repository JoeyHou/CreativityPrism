# CreativityPrism
## Codebase Structure 
- `\data_cleaning`: all data access, cleaning and formating code
- `\cs4`: evaluation codebase for [cs4](https://arxiv.org/pdf/2410.04197) 
- `\aut_ttcw_csshort`: evaluation codebase for [ttcw](https://arxiv.org/abs/2309.14556), [aut](https://kar.kent.ac.uk/101551/1/Pushing_the_Limits_of_GPT_s_Creativity_for_Alternative_Uses_and_Torrence_Tests.pdf), and [creative_short_story](https://arxiv.org/pdf/2411.02316)
- `neocoder_dat`: evaluation codebase for [neocoder](https://arxiv.org/pdf/2407.09007), [dat](https://openreview.net/forum?id=BpibUh0aB3)

## Requirement
- `vllm`: [0.7.2](https://docs.vllm.ai/en/v0.7.2/getting_started/installation/index.html) (at least 0.7.0, for the support of deepseek-v3)
- `Python`: 3.9 – 3.12
- `cuda`: >= 12.1
- Detailed requirements please check `requirements.txt` of each subfolder.