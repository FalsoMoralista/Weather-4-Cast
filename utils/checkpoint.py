def remove_prefix(state_dict, to_replace):
    """Remove prefix from state dict keys if present."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(to_replace, "")
        new_state_dict[new_key] = value
    return new_state_dict
