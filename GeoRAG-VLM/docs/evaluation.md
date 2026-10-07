# Evaluation protocol

1. Acquire imagery with verified signed coordinates, dates, native band resolution and source provenance. Preserve index legends and processing details.
2. Split by geographic region/event, not random near-duplicate crops. Keep reference captions and query answers outside the retrieval corpus. Decide beforehand whether target-scene captions are allowed as evidence; enforce this with exclusions.
3. Run image-only and image-plus-metadata Kosmos-2 with identical image, question, model revision, seed, beams and length settings. Legacy-tag prompting is a distinct ablation, not a native model capability.
4. Separately compare text-answer conditions with identical question, target metadata, system instructions and DeepSeek settings. Only the retrieved evidence changes. Log full prompts, document IDs/scores, corpus/query hashes, response IDs, reported model and usage. Repeat pairs when estimating variability.
5. Evaluate retrieval using relevance labels: Recall@K is the fraction of relevant documents retrieved; reciprocal rank is truncated at K. Average across queries for mean reciprocal rank at K.
6. Report ROUGE-L F1 and optional BLEU-4 with lowercased regex word tokenization and NLTK method-4 smoothing. The tokenizer differs from the original notebook; historical BLEU values are not directly comparable. These metrics measure textual overlap, not truth.
7. Annotate individual answer claims as supported, unsupported or unverifiable using independent scene evidence. Report unsupported claims / all annotated claims and unverifiable claims / all claims separately; no claims means an undefined rate (`null`). Use a second annotator and report agreement for a real study.
8. Report sample size, per-query outputs, model/corpus versions, paired differences and uncertainty. A tiny toy corpus cannot establish performance on real imagery.

Generated descriptions in the retrieval corpus can propagate errors. Spatial consistency, resolution feasibility, index interpretation and abstention should be audited explicitly. An object absent from a coarse land-cover label is not automatically a hallucination: evidence must be sufficient to label the claim.

The four-way research design proposed earlier spans both VLM prompt ablations and LLM evidence ablations. Those have different input modalities and inference stages. Do not present them as one perfectly matched comparison unless the final answer model and visual evidence are controlled across conditions.

The current code does not implement BERTScore, METEOR, automatic claim verification, remote-sensing encoder comparisons, confidence calibration, or bootstrap intervals. Add these only with appropriate reference data and a written protocol.
