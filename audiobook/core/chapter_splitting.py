
import re
from typing import List, Dict, Any

class ChapterSplitter:
    """
    Splits book text into chapters and identifies sections to include/exclude.
    """
    
    CHAPTER_PATTERNS = [
        r'^\s*chapter\s+\d+',
        r'^\s*chapter\s+[ivxlcdm]+',
        r'^\s*chapter\s+[a-z]+',  # Generic word match for One, Two, Twenty
        r'^\s*\d+\.\s+.*',       # "1. The Beginnings"
        r'^\s*[ivxlcdm]+\.\s+.*', # "I. The Start"
        r'^\s*part\s+\d+',
        r'^\s*part\s+[ivxlcdm]+',
        r'^\s*book\s+\d+',
        r'^\s*prologue\s*',
        r'^\s*epilogue\s*',
        r'^\s*introduction\s*',
        r'^\s*preface\s*',
        r'^\s*foreword\s*',
        r'^\s*#+\s*chapter.*',    # Markdown style
        r'^[A-Z\s]{5,50}$',       # All caps short lines (heuristic)
    ]
    
    EXCLUDE_PATTERNS = [
        r'^\s*table\s+of\s+contents\s*',
        r'^\s*contents\s*',
        r'^\s*acknowledg?ments\s*',
        r'^\s*references\s*',
        r'^\s*bibliography\s*',
        r'^\s*index\s*',
        r'^\s*about\s+the\s+author\s*',
        r'^\s*copyright\s*',
        r'^\s*dedication\s*',
        r'^\s*notes\s*',
        r'^\s*endnotes\s*',
    ]

    def split(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into chapters.
        
        Returns:
            List of dicts: [{'title': str, 'content': str, 'include': bool}]
        """
        lines = text.split('\n')
        chapters = []
        current_title = "Front Matter"
        current_content = []
        current_include = True # Default for front matter? Usually no, but let's see.
        
        # Heuristic: Front matter usually implicitly starts at the beginning
        # We'll check if the first section looks like TOC/Copyright later
        
        def is_heading(line):
            line_lower = line.lower().strip()
            # Combine chapter and exclude patterns to find any split point
            patterns = self.CHAPTER_PATTERNS + self.EXCLUDE_PATTERNS
            for pattern in patterns:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    # Check if line is short enough to be a header (e.g. < 60 chars)
                    if len(line) < 80: 
                        return True
            return False

    def is_exclude_heading(self, title):
        title_lower = title.lower().strip()
        for pattern in self.EXCLUDE_PATTERNS:
            if re.match(pattern, title_lower, re.IGNORECASE):
                return True
        return False

        for line in lines:
            if is_heading(line):
                # Save previous chapter
                if current_content:
                    # Clean up content
                    content_str = '\n'.join(current_content).strip()
                    if content_str: # Only add if not empty
                        # Check include status for the COMPLETED chapter
                        include = not self.is_exclude_heading(current_title)
                        # Special case: "Front Matter" might need checking by content content
                        if current_title == "Front Matter":
                             # If front matter is huge, it's probably the whole book if no chapters found.
                             # If it's short, it's likely junk.
                             # But we'll default to False for Front Matter if it matches exclude patterns 
                             # (unlikely to match regex if name is Front Matter)
                             # Actually let's default Front Matter to False if it's small, True if large?
                             # Better: default False mostly.
                             include = False
                        
                        chapters.append({
                            'title': current_title,
                            'content': content_str,
                            'include': include
                        })
                
                # Start new chapter
                current_title = line.strip()
                current_content = []
                # current_include will be determined when pushing
            else:
                current_content.append(line)
        
        # Push last chapter
        if current_content:
            content_str = '\n'.join(current_content).strip()
            if content_str:
                include = not self.is_exclude_heading(current_title)
                if current_title == "Front Matter":
                    include = False # Default generic front matter to excluded
                
                chapters.append({
                    'title': current_title,
                    'content': content_str,
                    'include': include
                })
        
        # If no chapters found (just Front Matter), mark it as included so we don't return nothing
        if len(chapters) == 1 and chapters[0]['title'] == "Front Matter":
            chapters[0]['include'] = True
            chapters[0]['title'] = "Full Text (No chapters detected)"

        return chapters

