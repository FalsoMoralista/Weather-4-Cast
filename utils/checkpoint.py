def remove_prefix(state_dict, to_replace):
    """Remove prefix from state dict keys if present."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(to_replace, "")
        new_state_dict[new_key] = value
    return new_state_dict


def remove_with_name(state_dict, to_remove):
    """Remove keys containing a specific substring from state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if to_remove not in key:
            new_state_dict[key] = value
    return new_state_dict
