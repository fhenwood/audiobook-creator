#!/usr/bin/env python3
"""
Audiobook Creator - Command Line Interface
Copyright (C) 2025

Usage:
    python cli.py extract <book_file>        Extract text from an ebook
    python cli.py emotion-tags               Add emotion tags to extracted text
    python cli.py generate <book_file>       Generate audiobook from book file
    python cli.py sample <voice> [text]      Generate a voice sample
    
Examples:
    python cli.py extract mybook.epub
    python cli.py emotion-tags
    python cli.py generate mybook.epub --voice zac --format m4b
    python cli.py sample zac "Hello, this is a test."
"""

import argparse
import asyncio
import os
import sys

def extract_text(args):
    """Extract text from an ebook file."""
    from audiobook.core.text_extraction import extract_text_from_book_using_calibre
    
    if not os.path.exists(args.book_file):
        print(f"❌ File not found: {args.book_file}")
        sys.exit(1)
    
    print(f"📖 Extracting text from: {args.book_file}")
    text = extract_text_from_book_using_calibre(args.book_file)
    
    output_file = args.output or "converted_book.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"✅ Text extracted successfully!")
    print(f"📄 Saved to: {output_file}")
    print(f"📊 Total characters: {len(text):,}")

async def add_emotion_tags_async(args):
    """Add emotion tags to the extracted text."""
    from audiobook.core.emotion_tags import process_emotion_tags
    
    if not os.path.exists("converted_book.txt"):
        print("❌ No converted_book.txt found. Run 'extract' first.")
        sys.exit(1)
    
    print("🎭 Adding emotion tags...")
    async for progress in process_emotion_tags(False):
        print(progress)
    
    print("✅ Emotion tags added successfully!")
    print("📄 Output saved to: tag_added_lines_chunks.txt")

def add_emotion_tags(args):
    """Wrapper for async emotion tags function."""
    asyncio.run(add_emotion_tags_async(args))

async def generate_audiobook_async(args):
    """Generate an audiobook from a book file."""
    from audiobook.core.job_service import job_service
    from audiobook.utils.job_manager import JobProgress

    if not os.path.exists(args.book_file):
        print(f"❌ File not found: {args.book_file}")
        sys.exit(1)
    
    # Determine output format
    format_map = {
        'm4b': 'M4B (Chapters & Cover)',
        'mp3': 'MP3',
        'wav': 'WAV',
        'aac': 'AAC',
        'flac': 'FLAC',
        'opus': 'OPUS',
    }
    output_format = format_map.get(args.format.lower(), 'M4B (Chapters & Cover)')
    
    print(f"🎧 Generating audiobook...")
    print(f"   Engine: {args.engine}")
    print(f"   Voice: {args.voice}")
    print(f"   Format: {output_format}")
    print(f"   Verify: {'Yes' if args.verify else 'No'}")
    print()

    # Progress callback
    async def on_progress(progress: JobProgress):
        print(f"\r{progress}", end="", flush=True)
        if progress.percent_complete >= 100:
            print()

    try:
        job = await job_service.create_and_run_job(
            book_title=os.path.basename(args.book_file),
            book_path=args.book_file,
            engine=args.engine,
            voice=args.voice,
            output_format=output_format,
            add_emotion_tags=args.emotion_tags,
            reference_audio_path=args.reference,
            verification_enabled=args.verify,
            on_progress=on_progress
        )
        
        # Poll if needed (create_and_run_job waits, but job_service might need explicit wait logic?)
        # job_service.create_and_run_job calls _run_job which does async for progress in pipeline.run()
        # So it awaits completion.
        
    except Exception as e:
        print(f"\n❌ CLI Generation Error: {e}")
        sys.exit(1)
    
    print()
    print("✅ Audiobook generated successfully!")
    print(f"📁 Output directory: generated_audiobooks/")

def generate_audiobook(args):
    """Wrapper for async audiobook generation."""
    asyncio.run(generate_audiobook_async(args))

def generate_sample(args):
    """Generate a voice sample."""
    from openai import OpenAI
    
    tts_base_url = os.environ.get("TTS_BASE_URL", "http://localhost:8880/v1")
    tts_api_key = os.environ.get("TTS_API_KEY", "not-needed")
    
    text = args.text or "Hello! This is a sample of my voice. I can read your books with emotion and expression."
    
    print(f"🔊 Generating sample for voice: {args.voice}")
    
    try:
        client = OpenAI(base_url=tts_base_url, api_key=tts_api_key)
        
        with client.audio.speech.with_streaming_response.create(
            model="orpheus",
            voice=args.voice,
            response_format="wav",
            speed=0.85,
            input=text,
            timeout=120
        ) as response:
            output_file = args.output or f"{args.voice}_sample.wav"
            response.stream_to_file(output_file)
        
        print(f"✅ Sample saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error generating sample: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Audiobook Creator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s extract mybook.epub                    Extract text from ebook
  %(prog)s extract mybook.pdf -o book.txt         Extract with custom output
  %(prog)s emotion-tags                           Add emotion tags
  %(prog)s generate mybook.epub                   Generate M4B audiobook
  %(prog)s generate mybook.epub --voice tara      Use Tara voice
  %(prog)s generate mybook.epub --format mp3      Output as MP3
  %(prog)s generate mybook.epub --engine maya --reference voice.wav
  %(prog)s sample zac                             Generate voice sample
  %(prog)s sample tara "Custom text here"         Sample with custom text
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract text from an ebook')
    extract_parser.add_argument('book_file', help='Path to ebook file (EPUB, PDF, TXT)')
    extract_parser.add_argument('-o', '--output', help='Output file (default: converted_book.txt)')
    extract_parser.set_defaults(func=extract_text)
    
    # Emotion tags command
    emotion_parser = subparsers.add_parser('emotion-tags', help='Add emotion tags to extracted text')
    emotion_parser.set_defaults(func=add_emotion_tags)
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate audiobook')
    gen_parser.add_argument('book_file', help='Path to ebook file')
    gen_parser.add_argument('--voice', '-v', default='zac', 
                           help='Voice name (default: zac). Options: tara, leah, jess, mia, leo, dan, zac, zoe')
    gen_parser.add_argument('--format', '-f', default='m4b',
                           help='Output format (default: m4b). Options: m4b, mp3, wav, aac, flac, opus')
    gen_parser.add_argument('--engine', '-e', default='orpheus',
                           help='TTS engine (default: orpheus). Options: orpheus, vibevoice, maya')
    gen_parser.add_argument('--reference', '-r', 
                           help='Reference audio for Maya voice cloning')
    gen_parser.add_argument('--emotion-tags', '-t', action='store_true',
                           help='Use emotion tags (Orpheus only, requires running emotion-tags first)')
    gen_parser.add_argument('--verify', action='store_true',
                           help='Enable Whisper verification (slow but accurate)')
    gen_parser.set_defaults(func=generate_audiobook)
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Generate a voice sample')
    sample_parser.add_argument('voice', help='Voice name (e.g., zac, tara)')
    sample_parser.add_argument('text', nargs='?', help='Text to speak (optional)')
    sample_parser.add_argument('-o', '--output', help='Output file (default: <voice>_sample.wav)')
    sample_parser.set_defaults(func=generate_sample)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)

if __name__ == "__main__":
    main()
