"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
import shutil
import sys
import time
import traceback
import os
import subprocess
from audiobook.utils.shell_commands import check_if_calibre_is_installed, run_shell_command_secure, validate_file_path_allowlist

def validate_book_path(book_path):
    """
    Validates that a book file path is safe using allowlist approach.
    
    Args:
        book_path (str): The path to the book file
        
    Returns:
        bool: True if path is safe and file exists, False otherwise
    """
    if not book_path or not isinstance(book_path, str):
        return False
    
    # Use allowlist-based validation  
    if not validate_file_path_allowlist(book_path):
        return False
        
    # Check if file exists and is readable
    try:
        return os.path.exists(book_path) and os.access(book_path, os.R_OK)
    except (OSError, TypeError):
        return False


def remove_front_matter(text: str) -> str:
    """
    Removes front matter like table of contents, copyright, etc.
    Looks for actual chapter content (substantial text after chapter heading),
    not just chapter headings that might appear in a table of contents.
    
    Args:
        text: The extracted book text
        
    Returns:
        Text with front matter removed
    """
    lines = text.split('\n')
    
    # Patterns that indicate chapter/content headers
    chapter_patterns = [
        r'^\s*chapter\s+\d+',
        r'^\s*chapter\s+[ivxlcdm]+\s*$',
        r'^\s*chapter\s+one\b',
        r'^\s*chapter\s+two\b',
        r'^\s*part\s+\d+',
        r'^\s*part\s+[ivxlcdm]+\s*$',
        r'^\s*part\s+one\b',
        r'^\s*book\s+\d+',
        r'^\s*book\s+one\b',
        r'^\s*prologue\s*$',
    ]
    
    def is_chapter_heading(line):
        line_lower = line.lower().strip()
        for pattern in chapter_patterns:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return True
        return False
    
    def has_substantial_content_after(idx, min_lines=5, min_chars=200):
        """Check if there's substantial content after a potential chapter heading."""
        content_lines = 0
        total_chars = 0
        
        for i in range(idx + 1, min(idx + 20, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            # Skip if we hit another chapter heading quickly (indicates TOC)
            if is_chapter_heading(line) and content_lines < 3:
                return False
            # Count substantive lines (not just short entries like TOC items)
            if len(line) > 30:  # Substantive line
                content_lines += 1
                total_chars += len(line)
        
        return content_lines >= min_lines or total_chars >= min_chars
    
    # Find the first chapter heading with substantial content after it
    for i, line in enumerate(lines):
        if is_chapter_heading(line):
            if has_substantial_content_after(i):
                # Found real chapter content, return from here
                return '\n'.join(lines[i:])
    
    # No chapter found, return original
    return text


def remove_back_matter(text: str) -> str:
    """
    Removes back matter like references, about the author, index, etc.
    Only removes if found after a substantial portion of the book.
    
    Args:
        text: The extracted book text
        
    Returns:
        Text with back matter removed
    """
    lines = text.split('\n')
    total_lines = len(lines)
    
    # Only look for back matter in the last 20% of the book
    search_start = int(total_lines * 0.8)
    
    # Patterns that indicate back matter sections to remove
    back_matter_patterns = [
        r'^\s*about\s+the\s+author\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*references?\s*$',
        r'^\s*end\s*notes?\s*$',
        r'^\s*index\s*$',
        r'^\s*glossary\s*$',
        r'^\s*appendix\s*$',
        r'^\s*appendices\s*$',
        r'^\s*also\s+by\s+',
        r'^\s*other\s+books?\s+by\s+',
        r'^\s*further\s+reading\s*$',
        r'^\s*suggested\s+reading\s*$',
        r'^\s*colophon\s*$',
    ]
    
    # Find where back matter starts (only in last 20% of book)
    for i in range(search_start, total_lines):
        line_lower = lines[i].lower().strip()
        
        for pattern in back_matter_patterns:
            if re.match(pattern, line_lower, re.IGNORECASE):
                # Found back matter, remove from here
                return '\n'.join(lines[:i])
    
    return text


def clean_extracted_text(text: str) -> str:
    """
    Cleans extracted text by removing front and back matter.
    
    Args:
        text: The raw extracted book text
        
    Returns:
        Cleaned text with front/back matter removed
    """
    text = remove_front_matter(text)
    text = remove_back_matter(text)
    return text


def extract_text_from_book_using_calibre(book_path):
    """
    Extracts text from a book using Calibre's ebook-convert utility.

    Args:
        book_path (str): The path to the book file.

    Returns:
        str: The extracted text from the book.
    """
    # Validate book path first
    if not validate_book_path(book_path):
        raise ValueError(f"Invalid or unsafe book path: {book_path}")
        
    # Get ebook-convert binary path
    ebook_convert_bin_result = shutil.which("ebook-convert")
    if not ebook_convert_bin_result or not os.path.exists(ebook_convert_bin_result.strip()):
        raise RuntimeError("ebook-convert command not found")
        
    ebook_convert_bin_path = ebook_convert_bin_result.strip()
    
    # Build secure command as list
    command = [ebook_convert_bin_path, book_path, "extracted_book.txt"]
    
    # Execute the command securely using our centralized function
    allowed_ebook_commands = ['ebook-convert']
    result = run_shell_command_secure(command, allowed_ebook_commands)
    
    if not result or result.returncode != 0:
        error_msg = result.stderr if result else "Unknown error"
        raise RuntimeError(f"Failed to convert book: {error_msg}")

    # Open the resulting text file and read its contents
    try:
        with open("extracted_book.txt", "r", encoding='utf-8') as f:
            book_text = f.read()
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to read extracted book: {e}")

    return book_text

def fix_unterminated_quotes(text: str):
    if not text:
        return text
    
    lines = text.splitlines()
    fixed_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        total_quotes = line.count('"')
        
        # If quotes are even, line is fine
        if total_quotes % 2 == 0:
            fixed_lines.append(line)
            continue
        
        print("Fixing unterminated quotes for line: ", line)
        # If odd number of quotes and line ends with a quote
        if line.endswith('"'):
            # Find the position of the last quote
            last_quote_pos = line.rfind('"')
            
            # Find all quote positions to avoid placing quotes next to each other
            quote_positions = [i for i, char in enumerate(line) if char == '"']
            
            # Find the best position to insert a quote (farthest from last quote)
            best_pos = 0
            max_distance = 0
            
            # Check each position in the line
            for pos in range(len(line)):
                # Skip if this position already has a quote
                if pos in quote_positions:
                    continue
                    
                # Skip if placing a quote here would create adjacent quotes
                if pos > 0 and line[pos-1] == '"':
                    continue
                if pos < len(line) - 1 and line[pos+1] == '"':
                    continue
                    
                # Calculate distance from last quote
                distance = abs(pos - last_quote_pos)
                
                # Update best position if this is farther
                if distance > max_distance:
                    max_distance = distance
                    best_pos = pos
            
            # Insert quote at the best position
            if max_distance > 0:
                line = line[:best_pos] + '"' + line[best_pos:]
            else:
                # Fallback: add quote at the beginning if no good position found
                line = '"' + line
        else:
            # If odd quotes but doesn't end with quote, add quote at the end
            line += '"'
        
        fixed_lines.append(line)
    
    result_text = "\n".join(fixed_lines)
    return result_text

def extract_main_content(text, start_marker="PROLOGUE", end_marker="ABOUT THE AUTHOR"):
    """
    Extracts the main content of a book between two markers (case-insensitive).
    Handles edge cases such as multiple marker occurrences and proper content boundaries.
    
    Args:
        text (str): The full text of the book.
        start_marker (str): The marker indicating the start of the main content.
        end_marker (str): The marker indicating the end of the main content.
    
    Returns:
        str: The extracted main content.
        
    Raises:
        ValueError: If markers are not found or if their positions are invalid.
    """
    
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        if not start_marker or not end_marker:
            raise ValueError("Markers must be non-empty strings")
        
        # Find all occurrences of markers
        start_positions = []
        end_positions = []
        pos = 0
        
        # Find all start marker positions
        while True:
            pos = text.find(start_marker, pos)
            if pos == -1:
                break
            start_positions.append(pos)
            pos += 1
            
        # Find all end marker positions
        pos = 0
        while True:
            pos = text.find(end_marker, pos)
            if pos == -1:
                break
            end_positions.append(pos)
            pos += 1
            
        # Validate marker existence
        if not start_positions:
            raise ValueError(f"Start marker '{start_marker}' not found in the text")
        if not end_positions:
            raise ValueError(f"End marker '{end_marker}' not found in the text")
            
        # Find the correct pair of markers
        start_index = start_positions[len(start_positions)-1]
        end_index = end_positions[len(end_positions)-1]
    
        if start_index is None or end_index is None:
            raise ValueError("Could not find valid marker positions with substantial content between them")
            
        # Extract and clean the content
        main_content = text[start_index:end_index].strip()
        
        # Validate extracted content
        if len(main_content) < 100:  # Adjust this threshold as needed
            raise ValueError("Extracted content is suspiciously short")
            
        # Remove any leading/trailing chapter markers or section headers
        lines = main_content.split('\n')
        while lines and (
            any(marker.lower() in lines[0].lower() 
                for marker in [start_marker, end_marker, 'chapter', 'part', 'book'])):
            lines.pop(0)
        while lines and (
            any(marker.lower() in lines[-1].lower() 
                for marker in [start_marker, end_marker, 'chapter', 'part', 'book'])):
            lines.pop()
            
        return '\n'.join(lines).strip()
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("Error", e, ", not extracting main content.")
        return text
    
def normalize_line_breaks(text):
    # Split the text into lines
    lines = text.splitlines()
    
    # Filter out empty lines and strip any leading/trailing whitespace
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Join the lines with a single line break
    normalized_text = '\n'.join(non_empty_lines)
    
    return normalized_text

def save_book(edited_text):
    with open("converted_book.txt", "w", encoding="utf-8") as fout:
        fout.write(edited_text)
    return "📖 Book saved successfully as 'converted_book.txt'! You can now proceed to the next optional step (Identifying Characters) or move onto Audiobook generation"

def process_book_and_extract_text(
    book_path: str,
    auto_clean: bool = True
):
    """
    Process a book file and extract text using Calibre.
    
    Args:
        book_path: Path to the book file
        auto_clean: Whether to automatically remove front/back matter (default: True)
    """
    # Early validation of book path
    if not validate_book_path(book_path):
        raise ValueError(f"Invalid or unsafe book file: {book_path}. Please check the file path and permissions.")
    
    # Always use calibre for text extraction
    text: str = extract_text_from_book_using_calibre(book_path)

    # Replace various Unicode characters with ASCII equivalents
    text = (text.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u2014", "-")
                .replace("\u2013", "-")
                .replace("\u2026", "...")
        )

    text = normalize_line_breaks(text)
    text = fix_unterminated_quotes(text)
    
    # Auto-clean front and back matter if enabled
    if auto_clean:
        text = clean_extracted_text(text)

    with open("converted_book.txt", 'w', encoding='utf-8') as fout:
        fout.write(text)
        yield text

def main():
    # Default book path
    book_path = "./sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub"

    # Check if a path is provided via command-line arguments
    if len(sys.argv) > 1:
        book_path = sys.argv[1]
        print(f"📂 Using book file from command-line argument: **{book_path}**")
    else:
        # Ask user for book file path if not provided
        input_path = input("\n📖 Enter the **path to the book file** (Press Enter to use default): ").strip()
        if input_path:
            book_path = input_path
        print(f"📂 Using book file: **{book_path}**")

    # Early validation of book path
    print("🔍 Validating book file...")
    if not validate_book_path(book_path):
        print(f"❌ **Book validation failed**: Invalid or inaccessible book file: {book_path}")
        print("\n💡 **Troubleshooting Tips:**")
        print("   • Ensure the book file path is correct and the file exists")
        print("   • Check that the file is readable and not corrupted")
        print("   • Verify file permissions")
        return
        
    print("✅ Book file validation successful!")
    print("✅ Book path set. Proceeding...\n")

    # Check calibre is installed
    is_calibre_installed = check_if_calibre_is_installed()
    if not is_calibre_installed:
        print("⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-convert** commands are available in your PATH.")
        return
    
    print("✅ Using Calibre to extract text...\n")

    # Ask about auto-cleanup
    auto_clean_input = input(
        "❓ Do you want to **automatically remove** table of contents, references, and other front/back matter?\n"
        "➡️ Answer (yes/no). Default is **yes**: "
    ).strip().lower()
    
    auto_clean = auto_clean_input != "no"
    
    print("✍️ Decoding the book...\n")
    
    text = None

    for output in process_book_and_extract_text(book_path, auto_clean=auto_clean):
        text = output

    # Ask user if they want to extract main content (manual override)
    if not auto_clean:
        have_to_extract_main_content = input(
            "❓ Do you want to extract the **main content** of the book using markers? (Optional)\n"
            "📌 You can specify start/end markers for finer control over the audiobook text.\n"
            "➡️ Answer (yes/no). Default is **no**: "
        ).strip().lower()

        if have_to_extract_main_content == "yes":
            start_marker = input("🔹 Enter the **start marker** for the main content (case-sensitive): Default is **PROLOGUE** :").strip()
            if(not start_marker):
                start_marker = "PROLOGUE"
            end_marker = input("🔹 Enter the **end marker** for the main content (case-sensitive): Default is **ABOUT THE AUTHOR** :").strip()
            if(not end_marker):
                end_marker = "ABOUT THE AUTHOR"
            text = extract_main_content(text, start_marker=start_marker, end_marker=end_marker)
            print("✅ Main content has been extracted!\n")

    print("\n🚀 Processing complete!\n")

    with open("converted_book.txt", 'w', encoding='utf-8') as fout:
        fout.write(text)

        print("📖 Your book has been successfully cleaned and converted!")
        print("✅ Saved as: converted_book.txt (in the current working directory)\n")

        print("🔍 Please manually review the converted book and remove any unnecessary content.\n")

        print("🎭 Next Steps:")
        print("  - If you want **multiple voices**, run:")
        print("    ➜ `python identify_characters_and_output_book_to_jsonl.py`")
        print("    (This script will identify characters and assign gender & age scores.)\n")
        print("  - If you want a **single voice**, directly run:")
        print("    ➜ `python generate_audiobook.py`")
        print("    (This will generate the audiobook immediately.)\n")

        print("🚀 Happy audiobook creation!")

if __name__ == "__main__":
    main()