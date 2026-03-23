# Dataset Quality and Robust Generalization in Tabular Machine Learning Under Natural and Operational Shift

## Abstract

Tabular machine learning (ML) is widely used in many real-world applications, where models are often used to support important decisions. But good test performance does not always mean a model will stay reliable in a different setting. In practice, tabular data often come with noisy labels, missing values, and features that may not stay available or consistent across environments. This raises an important question: which of these data-quality problems are most closely linked to poor robustness when conditions change?

To examine this, I use the UCI Diabetes 130-US hospitals dataset as the main case study and ACSEmployment as a secondary dataset, and compares Logistic Regression and LightGBM under clean conditions as well as under controlled degradations involving label noise, missingness, and feature-availability shift. Performance is evaluated on both in-domain and naturally shifted test sets using AUROC, AUPRC, Brier score, and generalization-gap analysis.

The results show that data-quality problems do not affect robustness in the same way. In the Diabetes experiments, the source-based cohort shift reduced performance, especially for LightGBM, which shows that stronger in-domain results did not automatically translate into stronger robustness under shift.
In ACSEmployment, the geographic shift from California to Texas was slightly favorable on average, but synthetic stress tests still revealed important vulnerabilities. Across both datasets, moderate missingness up to 20\% had limited impact under the implemented preprocessing pipelines, whereas label noise and feature-availability shift produced clearer robustness failures.
Label noise had its clearest effect on calibration, particularly in ACSEmployment, while feature-availability shift had its clearest effect on discrimination.

Overall, the findings suggest that robust generalization in tabular machine learning (ML) depends less on dataset degradation in general than on the specific way the data fail. Reduced target reliability and dependence on unstable feature groups emerged as stronger indicators of robustness risk than moderate missingness alone. This supports a data-centric view of reliable tabular ML, where natural-shift evaluation and targeted synthetic stress tests are used together rather than treated as separate concerns.
