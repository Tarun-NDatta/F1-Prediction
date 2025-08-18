from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Template filter to get dictionary item by key"""
    return dictionary.get(key) 




@register.filter
def lookup(d, key):
    """Try to get an item from a dictionary or list using the key/index"""
    try:
        if isinstance(d, dict):
            return d.get(key)
        elif isinstance(d, (list, tuple)) and isinstance(key, int):
            try:
                return d[key]
            except IndexError:
                return None
    except (TypeError, KeyError):
        return None
    return None