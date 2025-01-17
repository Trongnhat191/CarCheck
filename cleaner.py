from collections import Counter
import re

def clean_license_plates(plate_numbers):
    
    def clean_single_plate(plate):
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', plate)
        return cleaned
    
    # Clean each plate number and remove duplicates
    cleaned_plates = [clean_single_plate(plate) for plate in plate_numbers]
    # Remove duplicates while maintaining order
    # unique_plates = list(dict.fromkeys(cleaned_plates))
    res = []
    for i in cleaned_plates:
        
        if i[2].isalpha() and i[2].isupper():
            res.append(i)

    return res

def vote_plate_number(plates):
    # Find the maximum length among all plates
    max_length = max(len(plate) for plate in plates)
    
    # Initialize dictionary to store characters at each position
    position_chars = {i: [] for i in range(max_length)}
    
    # Collect all characters at each position
    for plate in plates:
        for i, char in enumerate(plate):
            position_chars[i].append(char)
    
    # Vote for most common character at each position
    voted_plate = ''
    voting_details = {}
    
    for pos in range(max_length):
        if position_chars[pos]:
            char_counts = Counter(position_chars[pos])
            most_common_char = char_counts.most_common(1)[0][0]
            voted_plate += most_common_char
            voting_details[pos] = dict(char_counts)
    
    return voted_plate, voting_details