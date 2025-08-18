from django import template

register = template.Library()

@register.filter
def lookup(dictionary, key):
    """
    Template filter to look up a dictionary value by key
    Usage: {{ my_dict|lookup:my_key }}
    """
    if dictionary and key in dictionary:
        return dictionary[key]
    return None