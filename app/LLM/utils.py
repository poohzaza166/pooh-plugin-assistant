import re


def remove_xml_tags(text:str) -> str:
    """Remove xml tags and their content from the text."""
    text = re.sub('<[^>]*>.*?<[^>]*>', '', text, flags=re.DOTALL)
    # Remove JSON-like formats
    text = re.sub('{[^}]*}', '', text)
    return text

def keep_xml_and_json(text : str) -> str:
    """Keep only xml and json-like formats from the text."""
    # Extract XML-like formats
    xml_matches = re.findall(r'<[^>]*>.*?</[^>]*>', text, flags=re.DOTALL)
    # Extract JSON-like formats
    json_matches = re.findall(r'{[^}]*}', text)
    # Combine extracted parts
    combined_matches = xml_matches + json_matches
    # Join the extracted parts into a single string
    result = '\n'.join(combined_matches)
    print("debug")
    print(result)
    return result
