import json
import re

def parse_greek_laws(input_file, output_file, source_name="Αστικός Κώδικας"):
    print(f"📖 Reading {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("❌ Error: 'raw_laws.txt' not found. Paste the text there first!")
        return

    print(f"   - Read {len(text)} characters. Slicing into Articles...")

    # --- THE SLICER LOGIC ---
    # 1. Find all headers that look like "Άρθρο 1", "ΑΡΘΡΟ 500", etc.
    # The regex captures the "Start Position" of every article.
    regex_pattern = r'(?:Άρθρο|ΑΡΘΡΟ|Αρθ|Αρθρο|Άρθρον|Article)\s*[:\.]?\s*(\d+)'
    
    # Find every occurrence in the file
    matches = list(re.finditer(regex_pattern, text, re.IGNORECASE))
    
    articles_list = []
    print(f"   - Found {len(matches)} article headers. Processing...")

    for i in range(len(matches)):
        # A. Identify the current Article Number
        current_match = matches[i]
        article_num = current_match.group(1)
        
        # B. Determine where the text STARTS (End of the header "Άρθρο 1")
        start_pos = current_match.end()
        
        # C. Determine where the text ENDS
        # It ends exactly where the NEXT article starts.
        # If it's the last article, it ends at the end of the file.
        if i < len(matches) - 1:
            next_match = matches[i+1]
            end_pos = next_match.start()
        else:
            end_pos = len(text)
            
        # D. Extract the text slice
        raw_content = text[start_pos:end_pos].strip()
        
        # E. Clean it up
        # Usually the first line is the Title, the rest is the Body.
        lines = [line.strip() for line in raw_content.split('\n') if line.strip()]
        
        if not lines:
            continue # Skip if empty

        title = lines[0].strip("-:").strip() # First line is Title
        
        # Logic: If the "Title" is huge (>300 chars), it's probably not a title, just body text.
        if len(title) > 300:
            title = "Κείμενο Νόμου"
            body = "\n".join(lines)
        else:
            body = "\n".join(lines[1:]) # Everything else is the Law
        
        # F. Auto-Categorize
        category = "ΓΕΝΙΚΟ"
        full_text_upper = (title + body).upper()
        
        if "ΜΙΣΘΩΣ" in full_text_upper: category = "ΜΙΣΘΩΣΕΙΣ"
        elif "ΚΛΗΡΟΝ" in full_text_upper: category = "ΚΛΗΡΟΝΟΜΙΚΟ"
        elif "ΔΙΑΖΥΓ" in full_text_upper: category = "ΟΙΚΟΓΕΝΕΙΑΚΟ"
        elif "ΕΤΑΙΡ" in full_text_upper: category = "ΕΤΑΙΡΙΚΟ"

        # Add to list
        articles_list.append({
            "article": article_num,
            "title": title,
            "text": body,
            "category": category,
            "source": source_name
        })

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles_list, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Success! Created JSON with {len(articles_list)} articles.")
    if len(articles_list) > 0:
        print(f"   - Sample: Article {articles_list[0]['article']} ({articles_list[0]['title']})")

if __name__ == "__main__":
    parse_greek_laws("raw_laws.txt", "greek_laws.json")
