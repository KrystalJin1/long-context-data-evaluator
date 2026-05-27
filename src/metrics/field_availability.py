CORE_FIELDS = ["id", "sample_id", "text", "prompt", "candidate", "source", "language"]
TASK_FIELDS = ["pipeline_type", "doc_id", "target", "target_index", "evidence", "summary"]


def check_field_availability(sample: dict) -> dict:
    result = {}
    for field in CORE_FIELDS + TASK_FIELDS:
        value = sample.get(field)
        result[field] = value is not None and str(value).strip() != ""
    return result
