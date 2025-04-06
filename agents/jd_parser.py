def parse_jd(t):
    skills = ['python', 'flask', 'sql', 'machine', 'learning', 'cloud', 'apis']
    words = t.lower().split()
    keywords = [w.strip('.,:') for w in words if w.strip('.,:') in skills]
    return keywords
