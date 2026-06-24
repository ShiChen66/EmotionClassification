# Mapping each of the 27 GoEmotions labels to an output group

MERGE_MAP = {
    "joy":"joy", "excitement":"joy", "amusement":"joy",
    "love":"love", "caring":"love",
    "gratitude":"gratitude",
    "admiration":"admiration", "approval":"admiration", "pride":"admiration",
    "optimism":"optimism", "relief":"optimism",
    "surprise":"surprise", "realization":"surprise", "curiosity":"surprise",
    "sadness":"sadness", "grief":"sadness", "disappointment":"sadness", "remorse":"sadness",
    "anger":"anger", "annoyance":"anger", "disapproval":"anger",
    "fear":"fear", "nervousness":"fear",
    "disgust":"disgust", "embarrassment":"disgust",
    "confusion":"confusion",
    "desire":"desire",
    "neutral":"neutral"
}

# Two intensity tiers per output group: [low, high]
EMOJI_TIERS = {
    "joy":['1', '2'],
    "love":['1', '2'],
    "gratitude":['1', '2'],
    "admiration":['1', '2'],
    "surprise":['1', '2'],
    "sadness":['1', '2'],
    "anger":['1', '2'],
    "fear":['1', '2'],
    "disgust":['1', '2'],
    "confusion":['1', '2'],
    "desire":['1', '2'],
    "neutral":['1', '2']
}

THRESHOLDS = {label: 0.3 for label in MERGE_MAP}

INTENSITY_TIER_CUTOFF = 0.55
REPEAT_CUTOFF = 0.90