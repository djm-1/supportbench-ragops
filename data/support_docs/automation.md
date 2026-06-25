# Automation Workflows

## Workflow Runs

Automation workflows run when their trigger conditions are met. Each completed run counts toward the workspace monthly automation usage limit. Failed runs also count if the workflow started execution.

## Run History

Workflow run history is retained for 90 days on Pro plans and 365 days on Enterprise plans. Free plans retain run history for 14 days.

## Secrets

Workflow secrets are encrypted at rest. Secret values are never returned through the API after creation. Users can rotate secrets from Settings > Automation > Secrets.

## Disabled Workflows

Workflows are disabled automatically after 10 consecutive failures caused by configuration errors. Admins can re-enable a workflow after fixing the configuration.
