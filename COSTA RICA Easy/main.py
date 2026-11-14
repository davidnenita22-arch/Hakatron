#!/usr/bin/env python3
"""Offline PII redactor. Run with Python 3.10+; no extra files required.

Evaluation: test-input.json -> test-output.json.
Practice: train-input.json -> train-output.json, or train-output.redacted.json
when the golden train-output.json already exists.
"""
import calendar
import datetime
import json
from pathlib import Path
import re
import unicodedata


def folded(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value.casefold())
                   if c.isalnum() and not unicodedata.combining(c))


WORD = r"[^\W\d_]+(?:['’\-][^\W\d_]+)*"
NAME = WORD + r"(?:[ \t]+" + WORD + r"){0,5}"
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS.update(dict(zip(('enero febrero marzo abril mayo junio julio agosto septiembre '
                       'octubre noviembre diciembre').split(), range(1, 13))))
MONTHS['setiembre'] = 9
MONTH_PATTERN = '|'.join(sorted(MONTHS, key=len, reverse=True))
DATE_PATTERN = (r'\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|'
                r'(?:' + MONTH_PATTERN + r')\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|'
                r'\d{1,2}\s+(?:de\s+)?(?:' + MONTH_PATTERN + r')\.?\s+(?:de\s+)?\d{4})\b')


def date_variants(value):
    """Generate equivalent DOB spellings, including unpadded numeric dates."""
    parts = re.findall(r'\d+|[^\W\d_]+', value.lower())
    try:
        if any(p in MONTHS for p in parts):
            month = next(MONTHS[p] for p in parts if p in MONTHS)
            nums = [int(p) for p in parts if p.isdigit()]
            day, year = nums
        else:
            nums = [int(p) for p in parts]
            if len(parts[0]) == 4:
                year, month, day = nums
            else:
                day, month, year = nums
            datetime.date(year, month, day)
        datetime.date(year, month, day)
    except (ValueError, StopIteration, IndexError):
        return {value}
    values = {value}
    for d in {str(day), f'{day:02}'}:
        for m in {str(month), f'{month:02}'}:
            values.update((f'{d}/{m}/{year}', f'{year}-{m}-{d}'))
        for name, number in MONTHS.items():
            if number == month:
                values.update((f'{name} {d}, {year}', f'{d} {name} {year}',
                               f'{d} de {name} de {year}'))
    return values


def redact(text):
    spans = []
    values = {}
    references = [m.span() for m in re.finditer(r'\bPV-[A-Za-z0-9]+\b', text, re.I)]

    def add_span(start, end, category):
        if start < end and not any(start < b and end > a for a, b in references):
            spans.append((start, end, category))

    def remember(value, category):
        if folded(value):
            values[(value, category)] = None

    def capture(pattern, category, group=0, flags=re.I):
        for match in re.finditer(pattern, text, flags):
            start, end = match.span(group)
            add_span(start, end, category)
            remember(match.group(group), category)

    # Context is important: travel destinations and the agency are not people.
    name_patterns = [
        r'\b(?:my name is|full name\s*[:=]|passenger name\s*[:=]|name\s*:)\s*(' + NAME + r')(?=[.,;!\n]|$)',
        r'\bmy\s+(?:wife|husband|son|daughter|partner|friend|mother|father|sister|brother|spouse|child)\s+('
        + NAME + r')(?=\s+(?:will|is|was|and I|also|can|has)\b)',
        r'(?im)^(?:regards|kind regards|best regards|sincerely|thanks|thank you|saludos|atentamente)[,!:.]?\s*\n[ \t]*('
        + NAME + r')(?=[ \t]*(?:\n|$))',
    ]
    for pattern in name_patterns:
        for match in re.finditer(pattern, text, re.I):
            name = match.group(1).strip()
            remember(name, 'NAME')
            # First and last names must disappear even when used alone.
            for token in re.findall(WORD, name):
                if folded(token) not in {'de', 'del', 'la', 'las', 'los', 'y', 'da', 'dos'}:
                    remember(token, 'NAME')
            add_span(*match.span(1), 'NAME')

    for match in re.finditer(r"(?<![\w.+\-])[\w.!#$%&'*+/=?^`{|}~\-]+@[\w\-]+(?:\.[\w\-]+)+", text):
        value = match.group()
        add_span(*match.span(), 'EMAIL')
        remember(value, 'EMAIL')
        remember(value.split('@')[0], 'EMAIL')

    # Country code plus 20 digits is the Costa Rican IBAN shape used here.
    capture(r'\bCR\s*\d{2}(?:[ \t\-]*\d){18,20}(?!\d)', 'ACCOUNT')
    capture(r'(?<!\d)[1-9][ \t\-]*\d{4}[ \t\-]*\d{4}(?!\d)', 'ID')
    phone_pattern = r'(?<![\w])(?:\+?506[ \t().\-]*)?[2-9](?:[ \t().\-]*\d){7}(?!\d)'
    for match in re.finditer(phone_pattern, text):
        value = match.group()
        # Do not mistake a substring of a bank account or national ID for a phone.
        if any(match.start() < b and match.end() > a for a, b, _ in spans):
            continue
        add_span(*match.span(), 'PHONE')
        digits = re.sub(r'\D', '', value)
        national = digits[-8:]
        remember(national, 'PHONE')
        remember('506' + national, 'PHONE')

    for match in re.finditer(DATE_PATTERN, text, re.I):
        add_span(*match.span(), 'DOB')
        for variant in date_variants(match.group()):
            remember(variant, 'DOB')

    # Stop before the functional booking clause, retaining every word in it.
    address_patterns = [
        r'\b(?:pickup at|paperwork to|written reply to|my address is|address\s*:)\s+(.+?)(?=\s+to our booking\b|[.;\n]|$)',
        r'\b((?:Calle|Avenida|Av\.|Barrio|Residencial|Urbanizaci[oó]n)\s+[^\n;]+?)(?=\s+to our booking\b|\.(?:\s|$)|$)',
    ]
    for pattern in address_patterns:
        for match in re.finditer(pattern, text, re.I):
            address = match.group(1).strip()
            if not address:
                continue
            add_span(*match.span(1), 'ADDRESS')
            remember(address, 'ADDRESS')
            for chunk in address.split(','):
                chunk = chunk.strip()
                if chunk:
                    remember(chunk, 'ADDRESS')
                    # Echoes may omit "Frente al", distances, or "Contiguo a la".
                    landmark = re.search(r'\b((?:Hotel|Supermercado|Panader[ií]a|Iglesia|Parque|Casa)\s+.+)', chunk, re.I)
                    if landmark:
                        remember(landmark.group(1), 'ADDRESS')
                        distinctive = re.sub(
                            r'^(?:Hotel|Supermercado|Panader[ií]a|Iglesia|Parque|Casa)\s+',
                            '', landmark.group(1), flags=re.I)
                        distinctive = re.sub(r'^(?:de la|de|del)\s+', '', distinctive, flags=re.I)
                        if distinctive:
                            remember(distinctive, 'ADDRESS')

    # Match using an accent/case/separator-insensitive view, mapping back to the
    # original offsets so all non-PII characters remain exactly as supplied.
    normalized = []
    offsets = []
    for index, char in enumerate(text):
        for item in folded(char):
            normalized.append(item)
            offsets.append(index)
    normalized = ''.join(normalized)
    for value, category in values:
        key = folded(value)
        position = 0
        while True:
            found = normalized.find(key, position)
            if found < 0:
                break
            position = found + 1
            start, end = offsets[found], offsets[found + len(key) - 1] + 1
            # Word boundaries for names, digit boundaries for numeric values.
            numeric = key.isdigit()
            boundary = str.isdigit if numeric else str.isalnum
            if ((start and boundary(text[start - 1])) or
                    (end < len(text) and boundary(text[end]))):
                continue
            add_span(start, end, category)

    # Combine overlaps (e.g. name inside email); retain the widest match's label.
    merged = []
    for start, end, category in sorted(spans, key=lambda s: (s[0], -s[1])):
        if merged and start < merged[-1][1]:
            a, b, label = merged[-1]
            if end - start > b - a:
                label = category
            merged[-1] = (a, max(b, end), label)
        else:
            merged.append((start, end, category))
    result = []
    previous = 0
    for start, end, category in merged:
        result.extend((text[previous:start], '[' + category + ']'))
        previous = end
    result.append(text[previous:])
    return ''.join(result) or '[REDACTED]'


def main():
    if Path('test-input.json').is_file():
        source, destination = 'test-input.json', 'test-output.json'
    else:
        source = 'train-input.json'
        destination = ('train-output.redacted.json' if Path('train-output.json').exists()
                       else 'train-output.json')
    data = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    output = {'redacted_documents': {doc['id']: redact(doc['text'])
                                     for doc in data['documents']}}
    Path(destination).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
