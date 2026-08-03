import difflib


def correct_symptom_names(user_typed_symptoms: list, valid_symptoms: list, cutoff: float = 0.6) -> dict:
    
    matched = []
    unmatched = []

    for typed in user_typed_symptoms:
        cleaned = typed.strip().lower().replace(" ", "_")

        # get_close_matches returns the best matches (if any) above the cutoff,
        # ranked by similarity. We only care about the single best one.
        close = difflib.get_close_matches(cleaned, valid_symptoms, n=1, cutoff=cutoff)

        if close:
            matched.append(close[0])
        else:
            unmatched.append(typed.strip())

    return {"matched": matched, "unmatched": unmatched}
