# Path redaction

Text configuration files, source snapshots, logs and checkpoint metadata replace an identifying
local directory name with `project_root`, so that the copies can be shared during double-anonymized
review. Model weights, numerical settings, predictions and split indices are unchanged. Set the
`project_root` paths to your local dataset locations before rerunning.
