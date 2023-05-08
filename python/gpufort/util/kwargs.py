
def get_value(key, default, **kwargs):
    kwarg_not_found_result = "ARGUMENT NOT FOUND"
    value = next((v for k, v in kwargs.items() if key == k),
                 kwarg_not_found_result)
    kwarg_found = (value != kwarg_not_found_result)
    return ((value if kwarg_found else default), kwarg_found)


def set_from_kwargs(obj, attr, default, **kwargs):
    value, found = get_value(attr, default, **kwargs)
    setattr(obj, attr, value)
    return found
