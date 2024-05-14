from mindnlp.dataset import load_dataset


def get_squad_dataset(tokenizer, batch_size):
    # process squad data
    def preprocess_function(id, title, context, question, answer):
        inputs = tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        start_positions = 0
        end_positions = 0

        answer_start = answer["answer_start"][0]
        answer_text = answer["text"][0]
        start_char = answer_start
        end_char = answer_start + len(answer_text)
        sequence_ids = inputs.sequence_ids(0)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if (
            offset_mapping[context_start][0] > end_char
            or offset_mapping[context_end][1] < start_char
        ):
            start_positions = 0
            end_positions = 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset_mapping[idx][0] <= start_char:
                idx += 1
            start_positions = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[idx][1] >= end_char:
                idx -= 1
            end_positions = idx + 1

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return (
            inputs["input_ids"],
            inputs["token_type_ids"],
            inputs["attention_mask"],
            inputs["start_positions"],
            inputs["end_positions"],
        )

    squad = load_dataset("squad", split="train[:5000]")
    squad = squad.map(
        preprocess_function,
        input_columns=["id", "title", "context", "question", "answers"],
        output_columns=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
        ],
        num_parallel_workers=8,
    )
    squad = squad.batch(batch_size)
    return squad
