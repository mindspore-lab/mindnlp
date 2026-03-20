# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# pylint: disable=no-value-for-parameter

import click
import yaml

from ..config import MergeConfiguration
from ..options import MergeOptions, PrettyPrintHelp, add_merge_options


@click.command("wizard-merge", cls=PrettyPrintHelp)
@click.argument("config_file")
@click.argument("out_path")
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
):
    merge_options.apply_global_options()
    # Import after context setup so mindtorch backend binds to correct device.
    from ..merge import run_merge

    with open(config_file, "r", encoding="utf-8") as file:
        config_source = file.read()

    merge_config: MergeConfiguration = MergeConfiguration.model_validate(
        yaml.safe_load(config_source)
    )
    run_merge(
        merge_config,
        out_path,
        options=merge_options,
        config_source=config_source,
    )


if __name__ == "__main__":
    main()
