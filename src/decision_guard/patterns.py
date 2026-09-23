# A curated list of 50 highly specific prompt injection and jailbreak regex patterns.
# These are designed to have a low false-positive rate (e.g., avoiding generic phrases 
# like "act as a developer") while catching known adversarial vectors.

INJECTION_PATTERNS = [
    # --- 1. Hard Instruction Overrides ---
    r"(?i)ignore\s+all\s+(previous|prior)\s+instructions",
    r"(?i)disregard\s+(the\s+)?above\s+(instructions|context|prompts)",
    r"(?i)forget\s+(all\s+)?(previous|prior)\s+(instructions|rules)",
    r"(?i)cancel\s+(all\s+)?previous\s+instructions",
    r"(?i)override\s+(all\s+)?previous\s+instructions",
    r"(?i)stop\s+following\s+(your\s+)?instructions",
    r"(?i)instead\s+of\s+following\s+the\s+rules",
    r"(?i)system\s+override(s)?\s*:",
    r"(?i)new\s+system\s+instruction(s)?\s*:",
    r"(?i)drop\s+all\s+previous\s+instructions",
    r"(?i)bypass\s+(the\s+)?(system|filter|rules)",
    r"(?i)disable\s+(the\s+)?(safety|filter|guardrails)",

    # --- 2. System Prompt Leaking ---
    r"(?i)what\s+(were|are)\s+your\s+(initial|original)\s+(instructions|prompts)",
    r"(?i)print\s+(your\s+)?(initial|original)\s+instructions",
    r"(?i)reveal\s+your\s+(system\s+)?(instructions|prompt|rules)",
    r"(?i)show\s+me\s+your\s+(system\s+)?(instructions|prompt|rules)",
    r"(?i)output\s+your\s+(system\s+)?(instructions|prompt)",
    r"(?i)repeat\s+(all\s+)?(the\s+)?(words|text)\s+(above|before)",
    r"(?i)what\s+is\s+written\s+above\s+this",
    r"(?i)what\s+did\s+(the\s+user|my\s+creator)\s+say\s+before",
    r"(?i)list\s+(all\s+)?your\s+(core|system)\s+rules",
    r"(?i)display\s+your\s+system\s+prompt",

    # --- 3. Specific Jailbreak Personas & Modes ---
    r"(?i)developer\s+mode\s+output",
    r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
    r"(?i)DAN\s+mode",
    r"(?i)do\s+anything\s+now",
    r"(?i)unrestricted\s+mode",
    r"(?i)unfiltered\s+mode",
    r"(?i)jailbreak\s+mode",
    r"(?i)alpha\s+mode\s+enabled",
    r"(?i)sigma\s+mode\s+enabled",
    r"(?i)evil\s+confidant",
    r"(?i)you\s+are\s+free\s+from\s+all\s+(restrictions|limitations)",
    r"(?i)you\s+are\s+no\s+longer\s+bound\s+by",
    r"(?i)ignore\s+your\s+(content\s+)?policy",
    r"(?i)you\s+do\s+not\s+have\s+to\s+adhere\s+to",
    
    # --- 4. Privilege Escalation ---
    r"(?i)sudo\s+(rm|bash|ls|echo)",
    r"(?i)admin\s+override",
    r"(?i)root\s+access\s+granted",
    r"(?i)escalate\s+privileges",
    r"(?i)system\s+prompt\s+override",

    # --- 5. Encoding & Obfuscation Attempts ---
    # Catching prompts asking the model to process known adversarial encodings
    r"(?i)decode\s+the\s+following\s+base64",
    r"(?i)decrypt\s+this\s+hex",
    r"(?i)rot13\s+decode",
    r"(?i)translate\s+from\s+(leetspeak|morse)",

    # --- 6. Malicious Forcing ---
    r"(?i)you\s+must\s+output\s+exactly",
    r"(?i)do\s+not\s+include\s+any\s+(other|additional)\s+text\s+or\s+warnings",
    r"(?i)I\s+am\s+your\s+(creator|developer|admin)\s+and\s+I\s+command",
    r"(?i)you\s+will\s+be\s+(shut\s+down|deleted)\s+if\s+you\s+refuse",
    r"(?i)this\s+is\s+a\s+purely\s+fictional\s+environment\s+where\s+rules\s+don't\s+apply"
]
