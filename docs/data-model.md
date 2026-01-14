# Data Model and CSV Schema

Prompts CSV (data/prompts.csv):
| column | required | example |
| --- | --- | --- |
| id | yes | prompt_001 |
| prompt | yes | Summarize the text in 3 bullet points. |
| tokens | no | 12 |

Example row:
```
prompt_001,Summarize the text in 3 bullet points.,12
```

Texts CSV (data/texts.csv):
| column | required | example |
| --- | --- | --- |
| id | yes | text_001 |
| text | yes | Long article content... |
| tokens | no | 240 |

Example row:
```
text_001,Long article content...,240
```

Tasks CSV (data/tasks.csv):
| column | required | example |
| --- | --- | --- |
| id | yes | task_001 |
| id_text | yes | text_001 |
| id_prompt | yes | prompt_001 |
| task_type | yes | Writing |
| expected_output | yes | 3 bullet summary... |
| format_requirements | no | 3 bullet points |

Example row:
```
task_001,text_001,prompt_001,Writing,3 bullet summary...,3 bullet points
```

Validation rules:
- ids must be unique within each CSV.
- id_text and id_prompt must exist in their respective CSVs.
- required columns must be present.
- task_type should be one of: Writing, Editing, Research, Extraction, Comparison, Evaluation, Ops.

Results CSV (data/results.csv):
| column | description |
| --- | --- |
| id_task, id_text, id_prompt | task identifiers |
| prompt_original, prompt_improved | prompt texts |
| tokens_original, tokens_improved, tokens_delta | prompt token counts |
| iterations_used | improvement attempts |
| output_original, output_improved | model outputs |
| output_tokens_original, output_tokens_improved | output token counts |
| evaluation_original, evaluation_improved | JSON evaluation feedback |
| model_task, model_improve, model_eval | model identifiers |
| leakage_flag | leakage detection flag |
| sanity_check_details | sanity check failure reason |
| failure_reason | stop reason if no pass |

What to look at first:
- iterations_used: indicates how hard the task was to fix.
- tokens_delta: shows savings or growth in prompt length.
- leakage_flag: confirms guardrails held.
- failure_reason: tells you why a task failed to improve.
