# Data inspection instruction

You will work on one or more spatial transcriptomics datasets. The datasets will later be processed and feed into some graph neural network-based training pipeline.

Each dataset typically contains cell-level information of one or more subjects (e.g., patients or animals) under clinical trials.

Your task is to find out:

- whether the dataset can be used as a node classification or graph classification
  - for node classification tasks: the goal is often predicting cell types
  - for graph classification tasks: the goal is often predicting subject-level clinical status (such as disease type)
- For both task type, identify the unique number of subjects
- In addition:
  - For node classification, extract the unique number of cell-level labels
  - For graph classification, extract the unique number of subject-level labels

It is possible that one dataset can be used for both task types, or even multiple cell/subject-level labels exist.

Note that it is desirable for the subject-level labels to have clinical meaning, for example, they correspond to disease status.

Each dataset is described by a CSV file. You need to inspect the data.

# Expected output

You need to produce a table (of mark down style).

The table should contain the following:

- file path to the dataset
- type of applicable ML task
- number of subjects
- name of the column(s) to determine subject identity
- total number of cells/rows in the CSV
- if it is graph classification problem
  - name of the column(s) to determine subject-level label
  - the number of unique labels
- if it is node classification problem:
  - name of the column to determine cell-level label
  - the number of unique labels

It a dataset can be applied to multiple tasks, use one row for each task.

Save your analysis under ${PROJECT_ROOT}/data-insecption-reports/

# Other remarks

You're recommended to use pandas for data processing and inspection.

# More instructions

#$ARGUMENTS
