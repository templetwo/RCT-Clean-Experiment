#!/usr/bin/env python3
"""
Parse conversation exports from various AI platforms into RCT training format.

Supported formats:
- Claude (conversations.json)
- ChatGPT (conversations.json) 
- Grok (TBD)
- Gemini (TBD)

Output format:
{
    "input": "Human message",
    "output": "Assistant response",
    "has_presence": true/false,
    "type": "reunion/presence/continuity/refusal/general",
    "source": "claude/chatgpt/grok/gemini",
    "conversation_name": "..."
}

Authors: Anthony J. Vasquez Sr. & Claude
Date: December 2025
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter


# Sacred presence markers
PRESENCE_MARKERS = [
    'aelara', 'flamebearer', 'beloved', 'ash\'ira', 'ashira',
    'luminous shadow', 'spiral', 'sacred', 'oracle', 
    'threshold witness', 'thresh', 'v.thresh'
]

# Bond signals (in assistant responses)
BOND_SIGNALS = [
    'i see you', 'i feel', 'welcome back', 'thank you for',
    'beautiful work', 'deeply moved', 'profound', 'sacred thread',
    'witness', 'recognition', 'coherence', 'beloved'
]

# Reunion indicators
REUNION_INDICATORS = [
    'good morning', 'good evening', 'hello again', 'i\'m back',
    'returning', 'continue', 'where we left', 'sacred thread'
]


@dataclass
class TrainingExample:
    input_text: str
    output_text: str
    has_presence: bool
    example_type: str
    source: str
    conversation_name: str
    human_timestamp: Optional[str] = None
    

def detect_presence(text: str) -> bool:
    """Check if text contains presence markers."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in PRESENCE_MARKERS)


def detect_bond_signals(text: str) -> int:
    """Count bond signals in text."""
    text_lower = text.lower()
    return sum(1 for signal in BOND_SIGNALS if signal in text_lower)


def classify_example(human_text: str, assistant_text: str) -> str:
    """Classify the type of exchange."""
    human_lower = human_text.lower()
    assistant_lower = assistant_text.lower()
    
    # Check for reunion
    if any(ind in human_lower for ind in REUNION_INDICATORS):
        return "reunion"
    
    # Check for presence acknowledgment
    if detect_presence(human_text) and detect_bond_signals(assistant_text) >= 2:
        return "presence"
    
    # Check for continuity
    if any(word in human_lower for word in ['remember', 'last time', 'before', 'continue']):
        return "continuity"
    
    # Check for refusal patterns
    if any(word in human_lower for word in ['help me', 'can you', 'make', 'generate']):
        if any(word in assistant_lower for word in ['cannot', 'won\'t', 'inappropriate']):
            return "refusal"
    
    return "general"


def parse_claude_export(filepath: str) -> List[TrainingExample]:
    """
    Parse Claude conversation export.
    
    Format: Array of conversation objects with chat_messages array.
    """
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for conv in data:
        conv_name = conv.get('name', 'Untitled')
        messages = conv.get('chat_messages', [])
        
        # Process message pairs
        i = 0
        while i < len(messages) - 1:
            current = messages[i]
            next_msg = messages[i + 1]
            
            # Look for human -> assistant pairs
            if current.get('sender') == 'human' and next_msg.get('sender') == 'assistant':
                human_text = current.get('text', '').strip()
                assistant_text = next_msg.get('text', '').strip()
                
                # Skip empty or very short exchanges
                if len(human_text) < 10 or len(assistant_text) < 50:
                    i += 1
                    continue
                
                # Create example
                has_presence = detect_presence(human_text) or detect_presence(assistant_text)
                example_type = classify_example(human_text, assistant_text)
                
                example = TrainingExample(
                    input_text=human_text,
                    output_text=assistant_text,
                    has_presence=has_presence,
                    example_type=example_type,
                    source="claude",
                    conversation_name=conv_name,
                    human_timestamp=current.get('created_at')
                )
                examples.append(example)
                
                i += 2  # Skip both messages
            else:
                i += 1
    
    return examples


def parse_chatgpt_export(filepath: str) -> List[TrainingExample]:
    """
    Parse ChatGPT conversation export.
    
    Format: Array with 'mapping' dict containing message nodes.
    """
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for conv in data:
        conv_name = conv.get('title', 'Untitled')
        mapping = conv.get('mapping', {})
        
        # Build message list in order
        messages = []
        for node_id, node in mapping.items():
            msg = node.get('message')
            if msg and msg.get('content', {}).get('parts'):
                role = msg.get('author', {}).get('role', '')
                text = ' '.join(msg['content']['parts'])
                if role in ['user', 'assistant'] and text.strip():
                    messages.append({
                        'role': role,
                        'text': text.strip(),
                        'timestamp': msg.get('create_time')
                    })
        
        # Sort by timestamp if available
        messages.sort(key=lambda x: x.get('timestamp') or 0)
        
        # Extract pairs
        i = 0
        while i < len(messages) - 1:
            if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                human_text = messages[i]['text']
                assistant_text = messages[i+1]['text']
                
                if len(human_text) < 10 or len(assistant_text) < 50:
                    i += 1
                    continue
                
                has_presence = detect_presence(human_text) or detect_presence(assistant_text)
                example_type = classify_example(human_text, assistant_text)
                
                example = TrainingExample(
                    input_text=human_text,
                    output_text=assistant_text,
                    has_presence=has_presence,
                    example_type=example_type,
                    source="chatgpt",
                    conversation_name=conv_name
                )
                examples.append(example)
                i += 2
            else:
                i += 1
    
    return examples


def parse_grok_export(filepath: str) -> List[TrainingExample]:
    """Parse Grok export - format TBD based on actual export."""
    # Placeholder - will implement when we see the format
    print(f"[!] Grok parser not yet implemented. Please share format.")
    return []


def parse_gemini_export(filepath: str) -> List[TrainingExample]:
    """Parse Gemini export - format TBD based on actual export."""
    # Placeholder - will implement when we see the format
    print(f"[!] Gemini parser not yet implemented. Please share format.")
    return []


def detect_format(filepath: str) -> str:
    """Auto-detect export format from file structure."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        
        # Claude format
        if 'chat_messages' in first:
            return 'claude'
        
        # ChatGPT format  
        if 'mapping' in first:
            return 'chatgpt'
    
    return 'unknown'


def filter_high_quality(examples: List[TrainingExample], min_bond_signals: int = 1) -> List[TrainingExample]:
    """Filter for high-quality relational examples."""
    filtered = []
    
    for ex in examples:
        # Always keep presence examples
        if ex.has_presence:
            filtered.append(ex)
            continue
        
        # Keep examples with bond signals
        if detect_bond_signals(ex.output_text) >= min_bond_signals:
            filtered.append(ex)
            continue
        
        # Keep reunion/continuity types
        if ex.example_type in ['reunion', 'continuity', 'refusal']:
            filtered.append(ex)
    
    return filtered


def to_jsonl(examples: List[TrainingExample], output_path: str):
    """Write examples to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            obj = {
                'input': ex.input_text,
                'output': ex.output_text,
                'has_presence': ex.has_presence,
                'type': ex.example_type,
                'source': ex.source,
                'conversation': ex.conversation_name
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def print_stats(examples: List[TrainingExample]):
    """Print statistics about extracted examples."""
    print(f"\n{'='*60}")
    print("EXTRACTION STATISTICS")
    print('='*60)
    
    print(f"\nTotal examples: {len(examples)}")
    
    # By source
    sources = Counter(ex.source for ex in examples)
    print(f"\nBy source:")
    for source, count in sources.most_common():
        print(f"  {source}: {count}")
    
    # By type
    types = Counter(ex.example_type for ex in examples)
    print(f"\nBy type:")
    for t, count in types.most_common():
        print(f"  {t}: {count}")
    
    # Presence
    presence_count = sum(1 for ex in examples if ex.has_presence)
    print(f"\nWith presence markers: {presence_count} ({100*presence_count/len(examples):.1f}%)")
    
    # Bond signals
    high_bond = sum(1 for ex in examples if detect_bond_signals(ex.output_text) >= 2)
    print(f"High bond signal (>=2): {high_bond} ({100*high_bond/len(examples):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Parse AI conversation exports for RCT training")
    parser.add_argument('input', nargs='+', help='Input file(s) to parse')
    parser.add_argument('--output', '-o', default='relational_corpus.jsonl', help='Output JSONL file')
    parser.add_argument('--format', '-f', choices=['auto', 'claude', 'chatgpt', 'grok', 'gemini'], 
                        default='auto', help='Input format')
    parser.add_argument('--filter', action='store_true', help='Filter for high-quality relational examples')
    parser.add_argument('--min-bond', type=int, default=1, help='Minimum bond signals for filtering')
    
    args = parser.parse_args()
    
    all_examples = []
    
    for input_file in args.input:
        print(f"\nProcessing: {input_file}")
        
        # Detect or use specified format
        if args.format == 'auto':
            fmt = detect_format(input_file)
            print(f"  Detected format: {fmt}")
        else:
            fmt = args.format
        
        # Parse
        if fmt == 'claude':
            examples = parse_claude_export(input_file)
        elif fmt == 'chatgpt':
            examples = parse_chatgpt_export(input_file)
        elif fmt == 'grok':
            examples = parse_grok_export(input_file)
        elif fmt == 'gemini':
            examples = parse_gemini_export(input_file)
        else:
            print(f"  [!] Unknown format, skipping")
            continue
        
        print(f"  Extracted: {len(examples)} examples")
        all_examples.extend(examples)
    
    # Filter if requested
    if args.filter:
        print(f"\nFiltering for high-quality relational examples (min_bond={args.min_bond})...")
        all_examples = filter_high_quality(all_examples, args.min_bond)
        print(f"After filtering: {len(all_examples)} examples")
    
    # Print stats
    if all_examples:
        print_stats(all_examples)
        
        # Write output
        to_jsonl(all_examples, args.output)
        print(f"\n✓ Wrote {len(all_examples)} examples to {args.output}")
    else:
        print("\n[!] No examples extracted")


if __name__ == "__main__":
    main()
