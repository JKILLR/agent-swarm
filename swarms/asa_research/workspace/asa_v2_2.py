"""
ASA (Atomic Semantic Attention) v2.2 - Complete Implementation
==============================================================

All TODOs from reviews implemented:
- Subword alignment for BPE/WordPiece tokenizers
- Baseline comparison mode (standard transformer)
- Ablation modes (POS-only, features-only, full ASA)
- Expanded verb coverage (~300 verbs from VerbNet)
- Preprocessing/caching utilities

WHAT'S FIXED (predetermined, no learning):
- POS compatibility matrix (from Universal Dependencies v2)
- Feature extraction (from WordNet hypernyms)
- Verb requirements (from VerbNet frames)
- Pronoun requirements (from Binding Theory)

WHAT'S LEARNED (standard transformer):
- Token embeddings, QKV projections, FF layers, output projection

ABLATION MODES:
- 'full': POS mask + feature compatibility + feature blocking
- 'pos_only': POS mask only, no feature scoring
- 'features_only': Feature compatibility only, no POS mask
- 'none': Standard transformer (baseline)
"""

from dataclasses import dataclass
from typing import Dict, List, Set, FrozenSet, Optional, Tuple, Literal
from enum import IntEnum
from functools import lru_cache
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

class Feature(IntEnum):
    """Binary selectional features from VerbNet."""
    ANIMATE = 0
    HUMAN = 1
    ANIMAL = 2
    ORGANIZATION = 3
    CONCRETE = 4
    LOCATION = 5
    ABSTRACT = 6
    COMMUNICATION = 7
    COGNITION = 8
    EMOTION = 9
    EVENT = 10
    TIME = 11
    INSTRUMENT = 12
    COMESTIBLE = 13
    BODY_PART = 14

NUM_FEATURES = 15

POS_TAGS = [
    'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN', 'DET', 'ADP',
    'AUX', 'CCONJ', 'SCONJ', 'NUM', 'PART', 'INTJ', 'PUNCT', 'SYM', 'X'
]
POS_TO_ID = {pos: i for i, pos in enumerate(POS_TAGS)}
ID_TO_POS = {i: pos for pos, i in POS_TO_ID.items()}
NUM_POS = len(POS_TAGS)


# =============================================================================
# WORDNET MAPPINGS
# =============================================================================

SYNSET_TO_FEATURES: Dict[str, FrozenSet[Feature]] = {
    # Animate
    'person.n.01': frozenset({Feature.ANIMATE, Feature.HUMAN, Feature.CONCRETE}),
    'human.n.01': frozenset({Feature.ANIMATE, Feature.HUMAN, Feature.CONCRETE}),
    'animal.n.01': frozenset({Feature.ANIMATE, Feature.ANIMAL, Feature.CONCRETE}),
    'organism.n.01': frozenset({Feature.ANIMATE, Feature.CONCRETE}),
    'living_thing.n.01': frozenset({Feature.ANIMATE, Feature.CONCRETE}),
    
    # Organizations
    'organization.n.01': frozenset({Feature.ORGANIZATION}),
    'social_group.n.01': frozenset({Feature.ORGANIZATION}),
    'institution.n.01': frozenset({Feature.ORGANIZATION}),
    'company.n.01': frozenset({Feature.ORGANIZATION}),
    
    # Concrete
    'object.n.01': frozenset({Feature.CONCRETE}),
    'physical_entity.n.01': frozenset({Feature.CONCRETE}),
    'artifact.n.01': frozenset({Feature.CONCRETE, Feature.INSTRUMENT}),
    'tool.n.01': frozenset({Feature.CONCRETE, Feature.INSTRUMENT}),
    'device.n.01': frozenset({Feature.CONCRETE, Feature.INSTRUMENT}),
    'container.n.01': frozenset({Feature.CONCRETE}),
    'substance.n.01': frozenset({Feature.CONCRETE}),
    'whole.n.02': frozenset({Feature.CONCRETE}),
    
    # Locations
    'location.n.01': frozenset({Feature.LOCATION}),
    'region.n.01': frozenset({Feature.LOCATION}),
    'area.n.01': frozenset({Feature.LOCATION}),
    'place.n.02': frozenset({Feature.LOCATION}),
    'structure.n.01': frozenset({Feature.CONCRETE, Feature.LOCATION}),
    'building.n.01': frozenset({Feature.CONCRETE, Feature.LOCATION}),
    
    # Abstract
    'abstraction.n.06': frozenset({Feature.ABSTRACT}),
    'psychological_feature.n.01': frozenset({Feature.ABSTRACT}),
    'attribute.n.02': frozenset({Feature.ABSTRACT}),
    'relation.n.01': frozenset({Feature.ABSTRACT}),
    'measure.n.02': frozenset({Feature.ABSTRACT}),
    'group.n.01': frozenset({Feature.ABSTRACT}),
    'state.n.02': frozenset({Feature.ABSTRACT}),
    
    # Communication
    'communication.n.02': frozenset({Feature.ABSTRACT, Feature.COMMUNICATION}),
    'message.n.02': frozenset({Feature.ABSTRACT, Feature.COMMUNICATION}),
    'document.n.01': frozenset({Feature.CONCRETE, Feature.COMMUNICATION}),
    
    # Cognition
    'cognition.n.01': frozenset({Feature.ABSTRACT, Feature.COGNITION}),
    'idea.n.01': frozenset({Feature.ABSTRACT, Feature.COGNITION}),
    'concept.n.01': frozenset({Feature.ABSTRACT, Feature.COGNITION}),
    
    # Emotion
    'feeling.n.01': frozenset({Feature.ABSTRACT, Feature.EMOTION}),
    'emotion.n.01': frozenset({Feature.ABSTRACT, Feature.EMOTION}),
    
    # Events
    'event.n.01': frozenset({Feature.ABSTRACT, Feature.EVENT}),
    'act.n.02': frozenset({Feature.ABSTRACT, Feature.EVENT}),
    'activity.n.01': frozenset({Feature.ABSTRACT, Feature.EVENT}),
    
    # Time
    'time.n.01': frozenset({Feature.ABSTRACT, Feature.TIME}),
    'time_period.n.01': frozenset({Feature.ABSTRACT, Feature.TIME}),
    
    # Food
    'food.n.01': frozenset({Feature.CONCRETE, Feature.COMESTIBLE}),
    
    # Body
    'body_part.n.01': frozenset({Feature.CONCRETE, Feature.BODY_PART}),
}


# =============================================================================
# VERB REQUIREMENTS (EXPANDED - ~300 verbs)
# =============================================================================

VERB_CLASS_REQUIREMENTS: Dict[str, FrozenSet[Feature]] = {
    'perception': frozenset({Feature.ANIMATE}),
    'cognition': frozenset({Feature.ANIMATE}),
    'communication': frozenset({Feature.HUMAN}),
    'motion': frozenset({Feature.CONCRETE}),
    'motion_directed': frozenset({Feature.CONCRETE}),
    'change': frozenset({Feature.CONCRETE}),
    'change_state': frozenset({Feature.CONCRETE}),
    'creation': frozenset({Feature.ANIMATE}),
    'consumption': frozenset({Feature.ANIMATE}),
    'transfer': frozenset({Feature.ANIMATE}),
    'possession': frozenset({Feature.ANIMATE}),
    'contact': frozenset({Feature.CONCRETE}),
    'contact_impact': frozenset({Feature.CONCRETE}),
    'emotion': frozenset({Feature.ANIMATE}),
    'emotion_cause': frozenset({Feature.ANIMATE}),
    'social': frozenset({Feature.HUMAN}),
    'judgment': frozenset({Feature.ANIMATE}),
    'assessment': frozenset({Feature.ANIMATE}),
    'body_action': frozenset({Feature.ANIMATE}),
    'sound': frozenset({Feature.CONCRETE}),
    'light': frozenset({Feature.CONCRETE}),
    'weather': frozenset(),
    'stative': frozenset(),
    'existence': frozenset(),
    'spatial': frozenset({Feature.CONCRETE}),
}

# EXPANDED: ~300 common verbs mapped to classes
VERB_TO_CLASS: Dict[str, str] = {
    # === Perception (see-30.1, sight-30.2, peer-30.3) ===
    'see': 'perception', 'hear': 'perception', 'watch': 'perception',
    'observe': 'perception', 'examine': 'perception', 'notice': 'perception',
    'look': 'perception', 'listen': 'perception', 'feel': 'perception',
    'smell': 'perception', 'taste': 'perception', 'view': 'perception',
    'spot': 'perception', 'witness': 'perception', 'perceive': 'perception',
    'detect': 'perception', 'sense': 'perception', 'glimpse': 'perception',
    'overhear': 'perception', 'sight': 'perception', 'eye': 'perception',
    'scan': 'perception', 'inspect': 'perception', 'survey': 'perception',
    'study': 'perception', 'read': 'perception',
    
    # === Cognition (consider-29.9, characterize-29.2) ===
    'think': 'cognition', 'believe': 'cognition', 'know': 'cognition',
    'understand': 'cognition', 'consider': 'cognition', 'realize': 'cognition',
    'remember': 'cognition', 'forget': 'cognition', 'learn': 'cognition',
    'imagine': 'cognition', 'suppose': 'cognition', 'assume': 'cognition',
    'expect': 'cognition', 'hope': 'cognition', 'wish': 'cognition',
    'doubt': 'cognition', 'suspect': 'cognition', 'recognize': 'cognition',
    'recall': 'cognition', 'recollect': 'cognition', 'comprehend': 'cognition',
    'grasp': 'cognition', 'fathom': 'cognition', 'deduce': 'cognition',
    'infer': 'cognition', 'conclude': 'cognition', 'decide': 'cognition',
    'determine': 'cognition', 'figure': 'cognition', 'guess': 'cognition',
    'estimate': 'cognition', 'calculate': 'cognition', 'reckon': 'cognition',
    'ponder': 'cognition', 'contemplate': 'cognition', 'reflect': 'cognition',
    'meditate': 'cognition', 'speculate': 'cognition', 'wonder': 'cognition',
    
    # === Communication (say-37.7, tell-37.2, manner_speaking-37.3) ===
    'say': 'communication', 'tell': 'communication', 'speak': 'communication',
    'talk': 'communication', 'write': 'communication', 'announce': 'communication',
    'report': 'communication', 'ask': 'communication', 'answer': 'communication',
    'explain': 'communication', 'describe': 'communication', 'mention': 'communication',
    'state': 'communication', 'claim': 'communication', 'argue': 'communication',
    'suggest': 'communication', 'propose': 'communication', 'recommend': 'communication',
    'advise': 'communication', 'warn': 'communication', 'inform': 'communication',
    'notify': 'communication', 'declare': 'communication', 'proclaim': 'communication',
    'assert': 'communication', 'insist': 'communication', 'maintain': 'communication',
    'deny': 'communication', 'admit': 'communication', 'confess': 'communication',
    'acknowledge': 'communication', 'confirm': 'communication', 'promise': 'communication',
    'swear': 'communication', 'vow': 'communication', 'whisper': 'communication',
    'shout': 'communication', 'yell': 'communication', 'scream': 'communication',
    'murmur': 'communication', 'mutter': 'communication', 'mumble': 'communication',
    'chat': 'communication', 'converse': 'communication', 'discuss': 'communication',
    'debate': 'communication', 'negotiate': 'communication', 'communicate': 'communication',
    
    # === Motion (run-51.3.2, roll-51.3.1, vehicle-51.4.1) ===
    'go': 'motion', 'come': 'motion', 'run': 'motion', 'walk': 'motion',
    'move': 'motion', 'travel': 'motion', 'fly': 'motion', 'swim': 'motion',
    'jump': 'motion', 'leap': 'motion', 'hop': 'motion', 'skip': 'motion',
    'crawl': 'motion', 'creep': 'motion', 'slide': 'motion', 'glide': 'motion',
    'roll': 'motion', 'spin': 'motion', 'turn': 'motion', 'rotate': 'motion',
    'rush': 'motion', 'hurry': 'motion', 'race': 'motion', 'dash': 'motion',
    'sprint': 'motion', 'jog': 'motion', 'stroll': 'motion', 'wander': 'motion',
    'roam': 'motion', 'drift': 'motion', 'float': 'motion', 'sink': 'motion',
    'rise': 'motion', 'climb': 'motion', 'descend': 'motion', 'drop': 'motion',
    'fall': 'motion', 'tumble': 'motion', 'stumble': 'motion', 'trip': 'motion',
    'drive': 'motion', 'ride': 'motion', 'sail': 'motion', 'cruise': 'motion',
    
    # === Motion Directed (escape-51.1, arrive-51.2) ===
    'arrive': 'motion_directed', 'leave': 'motion_directed', 'depart': 'motion_directed',
    'enter': 'motion_directed', 'exit': 'motion_directed', 'return': 'motion_directed',
    'approach': 'motion_directed', 'reach': 'motion_directed', 'escape': 'motion_directed',
    'flee': 'motion_directed', 'retreat': 'motion_directed', 'advance': 'motion_directed',
    
    # === Change (break-45.1, bend-45.2, cooking-45.3) ===
    'break': 'change', 'destroy': 'change', 'damage': 'change',
    'fix': 'change', 'repair': 'change', 'restore': 'change',
    'change': 'change', 'transform': 'change', 'alter': 'change',
    'modify': 'change', 'adjust': 'change', 'adapt': 'change',
    'improve': 'change', 'worsen': 'change', 'increase': 'change',
    'decrease': 'change', 'reduce': 'change', 'expand': 'change',
    'shrink': 'change', 'grow': 'change', 'stretch': 'change',
    'bend': 'change', 'fold': 'change', 'twist': 'change',
    'crush': 'change', 'smash': 'change', 'shatter': 'change',
    'tear': 'change', 'rip': 'change', 'split': 'change',
    'cut': 'change', 'slice': 'change', 'chop': 'change',
    'cook': 'change', 'bake': 'change', 'fry': 'change',
    'boil': 'change', 'roast': 'change', 'grill': 'change',
    'melt': 'change', 'freeze': 'change', 'burn': 'change',
    
    # === Creation (build-26.1, create-26.4, performance-26.7) ===
    'make': 'creation', 'build': 'creation', 'create': 'creation',
    'produce': 'creation', 'develop': 'creation', 'design': 'creation',
    'construct': 'creation', 'assemble': 'creation', 'manufacture': 'creation',
    'generate': 'creation', 'form': 'creation', 'shape': 'creation',
    'craft': 'creation', 'forge': 'creation', 'compose': 'creation',
    'write': 'creation', 'draw': 'creation', 'paint': 'creation',
    'sculpt': 'creation', 'carve': 'creation', 'weave': 'creation',
    'knit': 'creation', 'sew': 'creation', 'cook': 'creation',
    'invent': 'creation', 'devise': 'creation', 'originate': 'creation',
    
    # === Consumption (eat-39.1, gobble-39.3) ===
    'eat': 'consumption', 'drink': 'consumption', 'consume': 'consumption',
    'devour': 'consumption', 'swallow': 'consumption', 'ingest': 'consumption',
    'chew': 'consumption', 'bite': 'consumption', 'sip': 'consumption',
    'gulp': 'consumption', 'gobble': 'consumption', 'nibble': 'consumption',
    
    # === Transfer (give-13.1, send-11.1, get-13.5.1) ===
    'give': 'transfer', 'send': 'transfer', 'bring': 'transfer',
    'take': 'transfer', 'receive': 'transfer', 'get': 'transfer',
    'pass': 'transfer', 'hand': 'transfer', 'deliver': 'transfer',
    'provide': 'transfer', 'supply': 'transfer', 'offer': 'transfer',
    'present': 'transfer', 'donate': 'transfer', 'contribute': 'transfer',
    'lend': 'transfer', 'borrow': 'transfer', 'return': 'transfer',
    'mail': 'transfer', 'ship': 'transfer', 'transport': 'transfer',
    
    # === Possession (own-100.1, get-13.5.1) ===
    'own': 'possession', 'possess': 'possession', 'have': 'possession',
    'hold': 'possession', 'keep': 'possession', 'retain': 'possession',
    'acquire': 'possession', 'obtain': 'possession', 'gain': 'possession',
    'lose': 'possession', 'lack': 'possession',
    
    # === Contact (hit-18.1, touch-20, poke-19) ===
    'hit': 'contact', 'touch': 'contact', 'kick': 'contact',
    'push': 'contact', 'pull': 'contact', 'strike': 'contact',
    'grab': 'contact', 'hold': 'contact', 'catch': 'contact',
    'throw': 'contact', 'toss': 'contact', 'hurl': 'contact',
    'pat': 'contact', 'tap': 'contact', 'slap': 'contact',
    'punch': 'contact', 'poke': 'contact', 'prod': 'contact',
    'rub': 'contact', 'stroke': 'contact', 'scratch': 'contact',
    'squeeze': 'contact', 'press': 'contact', 'pinch': 'contact',
    
    # === Emotion (admire-31.2, amuse-31.1, marvel-31.3) ===
    'love': 'emotion', 'hate': 'emotion', 'like': 'emotion',
    'dislike': 'emotion', 'fear': 'emotion', 'dread': 'emotion',
    'enjoy': 'emotion', 'prefer': 'emotion', 'want': 'emotion',
    'need': 'emotion', 'desire': 'emotion', 'crave': 'emotion',
    'admire': 'emotion', 'respect': 'emotion', 'appreciate': 'emotion',
    'value': 'emotion', 'cherish': 'emotion', 'treasure': 'emotion',
    'miss': 'emotion', 'regret': 'emotion', 'resent': 'emotion',
    'envy': 'emotion', 'pity': 'emotion', 'trust': 'emotion',
    
    # === Emotion Cause (amuse-31.1) ===
    'amuse': 'emotion_cause', 'please': 'emotion_cause', 'delight': 'emotion_cause',
    'satisfy': 'emotion_cause', 'annoy': 'emotion_cause', 'irritate': 'emotion_cause',
    'anger': 'emotion_cause', 'upset': 'emotion_cause', 'frighten': 'emotion_cause',
    'scare': 'emotion_cause', 'terrify': 'emotion_cause', 'surprise': 'emotion_cause',
    'shock': 'emotion_cause', 'amaze': 'emotion_cause', 'astonish': 'emotion_cause',
    'bore': 'emotion_cause', 'tire': 'emotion_cause', 'exhaust': 'emotion_cause',
    'excite': 'emotion_cause', 'thrill': 'emotion_cause', 'inspire': 'emotion_cause',
    'motivate': 'emotion_cause', 'encourage': 'emotion_cause', 'discourage': 'emotion_cause',
    'comfort': 'emotion_cause', 'calm': 'emotion_cause', 'relax': 'emotion_cause',
    'worry': 'emotion_cause', 'concern': 'emotion_cause', 'trouble': 'emotion_cause',
    'confuse': 'emotion_cause', 'puzzle': 'emotion_cause', 'perplex': 'emotion_cause',
    
    # === Social (meet-36.3, marry-36.2) ===
    'meet': 'social', 'marry': 'social', 'divorce': 'social',
    'befriend': 'social', 'date': 'social', 'visit': 'social',
    'greet': 'social', 'welcome': 'social', 'introduce': 'social',
    'invite': 'social', 'host': 'social', 'accompany': 'social',
    'join': 'social', 'follow': 'social', 'lead': 'social',
    'help': 'social', 'assist': 'social', 'support': 'social',
    'serve': 'social', 'hire': 'social', 'fire': 'social',
    'employ': 'social', 'manage': 'social', 'supervise': 'social',
    
    # === Judgment (judge-33, estimate-34.2) ===
    'judge': 'judgment', 'evaluate': 'judgment', 'assess': 'judgment',
    'rate': 'judgment', 'rank': 'judgment', 'grade': 'judgment',
    'criticize': 'judgment', 'praise': 'judgment', 'blame': 'judgment',
    'accuse': 'judgment', 'forgive': 'judgment', 'excuse': 'judgment',
    'approve': 'judgment', 'reject': 'judgment', 'accept': 'judgment',
    
    # === Stative/Existence (exist-47.1, seem-109) ===
    'be': 'stative', 'exist': 'stative', 'remain': 'stative',
    'stay': 'stative', 'seem': 'stative', 'appear': 'stative',
    'become': 'stative', 'look': 'stative', 'sound': 'stative',
    'feel': 'stative', 'smell': 'stative', 'taste': 'stative',
    'belong': 'stative', 'consist': 'stative', 'contain': 'stative',
    'include': 'stative', 'involve': 'stative', 'require': 'stative',
    'need': 'stative', 'deserve': 'stative', 'merit': 'stative',
    'equal': 'stative', 'resemble': 'stative', 'differ': 'stative',
    'matter': 'stative', 'count': 'stative', 'depend': 'stative',
    'fit': 'stative', 'suit': 'stative', 'match': 'stative',
    'last': 'stative', 'continue': 'stative', 'persist': 'stative',
    
    # === Body Actions (wink-40.3.1, crane-40.3.2) ===
    'breathe': 'body_action', 'blink': 'body_action', 'wink': 'body_action',
    'nod': 'body_action', 'shrug': 'body_action', 'yawn': 'body_action',
    'sneeze': 'body_action', 'cough': 'body_action', 'hiccup': 'body_action',
    'smile': 'body_action', 'frown': 'body_action', 'laugh': 'body_action',
    'cry': 'body_action', 'weep': 'body_action', 'sigh': 'body_action',
    'sleep': 'body_action', 'wake': 'body_action', 'rest': 'body_action',
    
    # === Spatial (put-9.1, remove-10.1) ===
    'put': 'spatial', 'place': 'spatial', 'set': 'spatial',
    'lay': 'spatial', 'position': 'spatial', 'locate': 'spatial',
    'remove': 'spatial', 'withdraw': 'spatial', 'extract': 'spatial',
    'insert': 'spatial', 'attach': 'spatial', 'connect': 'spatial',
    'separate': 'spatial', 'divide': 'spatial', 'combine': 'spatial',
    'fill': 'spatial', 'empty': 'spatial', 'cover': 'spatial',
    'wrap': 'spatial', 'surround': 'spatial', 'enclose': 'spatial',
    'hang': 'spatial', 'mount': 'spatial', 'install': 'spatial',
}

# Count for reference
_VERB_COUNT = len(VERB_TO_CLASS)  # Should be ~300


# =============================================================================
# PRONOUN REQUIREMENTS
# =============================================================================

PRONOUN_REQUIREMENTS: Dict[str, FrozenSet[Feature]] = {
    # Personal
    'he': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'she': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'him': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'her': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'his': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'hers': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'they': frozenset({Feature.ANIMATE}),
    'them': frozenset({Feature.ANIMATE}),
    'their': frozenset({Feature.ANIMATE}),
    'theirs': frozenset({Feature.ANIMATE}),
    'it': frozenset(),
    'its': frozenset(),
    
    # First/second person
    'i': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'me': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'my': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'mine': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'you': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'your': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'yours': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'we': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'us': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'our': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'ours': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    
    # Interrogative/relative
    'who': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'whom': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'whose': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'what': frozenset(),
    'which': frozenset(),
    
    # Indefinite
    'someone': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'somebody': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'something': frozenset(),
    'anyone': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'anybody': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'anything': frozenset(),
    'everyone': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'everybody': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'everything': frozenset(),
    'no one': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'nobody': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'nothing': frozenset(),
    'one': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    
    # Reflexive
    'myself': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'yourself': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'himself': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'herself': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'itself': frozenset(),
    'ourselves': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'yourselves': frozenset({Feature.ANIMATE, Feature.HUMAN}),
    'themselves': frozenset({Feature.ANIMATE}),
}


# =============================================================================
# TOKEN PROPERTIES
# =============================================================================

@dataclass
class TokenProperties:
    """Extracted properties for a single token."""
    text: str
    lemma: str
    pos: str
    pos_id: int
    features: List[float]
    requirements: List[float]


# =============================================================================
# PROPERTY EXTRACTOR
# =============================================================================

class PropertyExtractor:
    """Extracts ASA properties from text. Use for preprocessing, not runtime."""
    
    def __init__(self, use_wordnet: bool = True):
        self.use_wordnet = use_wordnet
        self._nlp = None
        self._wn = None
    
    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    @property
    def wn(self):
        if self._wn is None and self.use_wordnet:
            try:
                from nltk.corpus import wordnet
                self._wn = wordnet
            except ImportError:
                self.use_wordnet = False
        return self._wn
    
    def extract(self, text: str) -> List[TokenProperties]:
        doc = self.nlp(text)
        return [self._extract_token(tok) for tok in doc]
    
    def _extract_token(self, tok) -> TokenProperties:
        pos = tok.pos_
        lemma = tok.lemma_.lower()
        pos_id = POS_TO_ID.get(pos, POS_TO_ID['X'])
        features = self._get_features(lemma, pos)
        requirements = self._get_requirements(lemma, pos)
        
        return TokenProperties(
            text=tok.text, lemma=lemma, pos=pos, pos_id=pos_id,
            features=features, requirements=requirements,
        )
    
    def _get_features(self, lemma: str, pos: str) -> List[float]:
        features = [0.0] * NUM_FEATURES
        
        if pos in ('DET', 'ADP', 'CCONJ', 'SCONJ', 'PUNCT', 'PART', 'AUX', 'SYM'):
            return features
        
        feature_set: Set[Feature] = set()
        
        if pos in ('NOUN', 'PROPN'):
            if self.use_wordnet and self.wn:
                feature_set = set(self._get_wordnet_features_cached(lemma, 'n'))
            if not feature_set:
                feature_set = {Feature.CONCRETE}
        elif pos == 'PRON':
            req = PRONOUN_REQUIREMENTS.get(lemma, frozenset())
            feature_set = set(req)
        elif pos == 'ADJ':
            if self.use_wordnet and self.wn:
                feature_set = set(self._get_wordnet_features_cached(lemma, 'a'))
        
        for f in feature_set:
            features[f] = 1.0
        return features
    
    def _get_requirements(self, lemma: str, pos: str) -> List[float]:
        requirements = [0.0] * NUM_FEATURES
        
        if pos == 'VERB':
            verb_class = VERB_TO_CLASS.get(lemma)
            if verb_class:
                req_set = VERB_CLASS_REQUIREMENTS.get(verb_class, frozenset())
                for f in req_set:
                    requirements[f] = 1.0
        elif pos == 'PRON':
            req_set = PRONOUN_REQUIREMENTS.get(lemma, frozenset())
            for f in req_set:
                requirements[f] = 1.0
        
        return requirements
    
    @lru_cache(maxsize=10000)
    def _get_wordnet_features_cached(self, lemma: str, wn_pos: str) -> FrozenSet[Feature]:
        synsets = self.wn.synsets(lemma, pos=wn_pos)
        if not synsets:
            return frozenset()
        
        synset = synsets[0]
        features: Set[Feature] = set()
        visited: Set[str] = set()
        to_visit = [synset]
        
        while to_visit:
            current = to_visit.pop(0)
            name = current.name()
            if name in visited:
                continue
            visited.add(name)
            if name in SYNSET_TO_FEATURES:
                features |= SYNSET_TO_FEATURES[name]
            to_visit.extend(current.hypernyms())
        
        return frozenset(features)
    
    def cache_stats(self) -> Dict[str, int]:
        info = self._get_wordnet_features_cached.cache_info()
        return {'hits': info.hits, 'misses': info.misses, 'size': info.currsize}


# =============================================================================
# SUBWORD ALIGNMENT
# =============================================================================

@dataclass
class AlignmentResult:
    """Result of subword alignment."""
    input_ids: torch.Tensor
    pos_ids: torch.Tensor
    features: torch.Tensor
    requirements: torch.Tensor
    word_ids: List[Optional[int]]


class SubwordAligner:
    """Aligns word-level properties to subword tokens."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.unknown_pos_id = POS_TO_ID['X']
    
    def align(
        self,
        text: str,
        word_properties: List[TokenProperties],
        max_length: Optional[int] = None,
    ) -> AlignmentResult:
        kwargs = {'return_offsets_mapping': True, 'return_tensors': 'pt', 'add_special_tokens': True}
        if max_length:
            kwargs['max_length'] = max_length
            kwargs['truncation'] = True
        
        encoded = self.tokenizer(text, **kwargs)
        input_ids = encoded['input_ids'].squeeze(0)
        offsets = encoded['offset_mapping'].squeeze(0).tolist()
        
        seq_len = len(input_ids)
        char_to_word = self._build_char_map(text, word_properties)
        
        pos_ids = torch.full((seq_len,), self.unknown_pos_id, dtype=torch.long)
        features = torch.zeros(seq_len, NUM_FEATURES)
        requirements = torch.zeros(seq_len, NUM_FEATURES)
        word_ids = []
        
        for i, (start, end) in enumerate(offsets):
            if start == end:
                word_ids.append(None)
                continue
            
            word_idx = self._find_word(start, end, char_to_word)
            word_ids.append(word_idx)
            
            if word_idx is not None and word_idx < len(word_properties):
                props = word_properties[word_idx]
                pos_ids[i] = props.pos_id
                features[i] = torch.tensor(props.features)
                requirements[i] = torch.tensor(props.requirements)
        
        return AlignmentResult(input_ids, pos_ids, features, requirements, word_ids)
    
    def _build_char_map(self, text: str, props: List[TokenProperties]) -> Dict[int, int]:
        char_to_word = {}
        pos = 0
        for idx, p in enumerate(props):
            start = text.find(p.text, pos)
            if start == -1:
                start = text.lower().find(p.text.lower(), pos)
            if start != -1:
                for i in range(start, start + len(p.text)):
                    char_to_word[i] = idx
                pos = start + len(p.text)
        return char_to_word
    
    def _find_word(self, start: int, end: int, char_to_word: Dict[int, int]) -> Optional[int]:
        for pos in [start, (start + end) // 2] + list(range(start, end)):
            if pos in char_to_word:
                return char_to_word[pos]
        for offset in range(1, 5):
            if start - offset in char_to_word:
                return char_to_word[start - offset]
        return None
    
    def align_batch(
        self,
        texts: List[str],
        batch_properties: List[List[TokenProperties]],
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        results = [self.align(t, p, max_length) for t, p in zip(texts, batch_properties)]
        
        max_seq = max(len(r.input_ids) for r in results)
        batch_size = len(results)
        pad_id = self.tokenizer.pad_token_id or 0
        
        input_ids = torch.full((batch_size, max_seq), pad_id, dtype=torch.long)
        pos_ids = torch.full((batch_size, max_seq), self.unknown_pos_id, dtype=torch.long)
        features = torch.zeros(batch_size, max_seq, NUM_FEATURES)
        requirements = torch.zeros(batch_size, max_seq, NUM_FEATURES)
        attention_mask = torch.zeros(batch_size, max_seq, dtype=torch.bool)
        
        for i, r in enumerate(results):
            seq_len = len(r.input_ids)
            input_ids[i, :seq_len] = r.input_ids
            pos_ids[i, :seq_len] = r.pos_ids
            features[i, :seq_len] = r.features
            requirements[i, :seq_len] = r.requirements
            attention_mask[i, :seq_len] = True
        
        return input_ids, pos_ids, features, requirements, attention_mask


# =============================================================================
# BONDING COMPUTER
# =============================================================================

def _build_pos_compatibility_matrix() -> torch.Tensor:
    matrix = torch.zeros(NUM_POS, NUM_POS)
    
    valid_pairs = [
        ('NOUN', 'VERB'), ('PROPN', 'VERB'), ('PRON', 'VERB'),
        ('NOUN', 'ADJ'), ('PROPN', 'ADJ'), ('PRON', 'ADJ'),
        ('VERB', 'VERB'),
        ('ADJ', 'NOUN'), ('ADJ', 'PROPN'),
        ('DET', 'NOUN'), ('DET', 'PROPN'),
        ('NUM', 'NOUN'),
        ('ADP', 'NOUN'), ('ADP', 'PROPN'), ('ADP', 'PRON'),
        ('NOUN', 'NOUN'), ('PROPN', 'NOUN'), ('NOUN', 'PROPN'), ('PROPN', 'PROPN'),
        ('ADV', 'VERB'), ('ADV', 'ADJ'), ('ADV', 'ADV'),
        ('AUX', 'VERB'), ('SCONJ', 'VERB'),
        ('AUX', 'ADJ'), ('AUX', 'NOUN'),
        ('CCONJ', 'NOUN'), ('CCONJ', 'VERB'), ('CCONJ', 'ADJ'), ('CCONJ', 'ADV'), ('CCONJ', 'PROPN'),
        ('PRON', 'NOUN'), ('PRON', 'PROPN'),
    ]
    
    for dep, head in valid_pairs:
        d, h = POS_TO_ID.get(dep), POS_TO_ID.get(head)
        if d is not None and h is not None:
            matrix[d, h] = matrix[h, d] = 1
    
    for i in range(NUM_POS):
        matrix[i, i] = 1
    
    punct_id = POS_TO_ID['PUNCT']
    matrix[punct_id, :] = matrix[:, punct_id] = 1
    
    return matrix


POS_COMPATIBILITY_MATRIX = _build_pos_compatibility_matrix()


# Ablation mode type
AblationMode = Literal['full', 'pos_only', 'features_only', 'none']


class BondingComputer:
    """Computes bonding masks with ablation support."""
    
    def __init__(self, device: torch.device = None, mode: AblationMode = 'full'):
        self.device = device or torch.device('cpu')
        self.pos_matrix = POS_COMPATIBILITY_MATRIX.to(self.device)
        self.mode = mode
    
    def to(self, device: torch.device) -> 'BondingComputer':
        self.device = device
        self.pos_matrix = self.pos_matrix.to(device)
        return self
    
    def compute_bonding_mask(
        self,
        pos_ids: torch.Tensor,
        features: torch.Tensor,
        requirements: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq = pos_ids.shape
        device = pos_ids.device
        
        # Mode: none = standard transformer (no masking)
        if self.mode == 'none':
            mask = torch.ones(batch, seq, seq, dtype=torch.bool, device=device)
            scores = torch.zeros(batch, seq, seq, device=device)
            return mask, scores
        
        # Compute POS mask
        if self.mode in ('full', 'pos_only'):
            row_ids = pos_ids.unsqueeze(2).expand(-1, -1, seq)
            col_ids = pos_ids.unsqueeze(1).expand(-1, seq, -1)
            pos_mask = self.pos_matrix[row_ids, col_ids].bool()
        else:
            pos_mask = torch.ones(batch, seq, seq, dtype=torch.bool, device=device)
        
        # Compute feature compatibility
        if self.mode in ('full', 'features_only'):
            features_j = features.unsqueeze(1)
            requirements_i = requirements.unsqueeze(2)
            satisfaction = (requirements_i * features_j).sum(dim=-1)
            req_count = requirements.sum(dim=-1, keepdim=True).unsqueeze(2).squeeze(-1) + 1e-6
            compatibility = satisfaction / req_count
        else:
            compatibility = torch.zeros(batch, seq, seq, device=device)
        
        # Feature blocking for PRON-NOUN (only in 'full' mode)
        if self.mode == 'full':
            pron_id, noun_id, propn_id = POS_TO_ID['PRON'], POS_TO_ID['NOUN'], POS_TO_ID['PROPN']
            is_pron = (pos_ids == pron_id).unsqueeze(2)
            is_noun = ((pos_ids == noun_id) | (pos_ids == propn_id)).unsqueeze(1)
            is_pron_noun = is_pron & is_noun
            
            req_count_check = requirements.sum(dim=-1, keepdim=True).unsqueeze(2).squeeze(-1)
            has_requirements = (req_count_check > 0).expand(-1, -1, seq)
            satisfaction_check = (requirements.unsqueeze(2) * features.unsqueeze(1)).sum(dim=-1)
            fully_satisfied = (satisfaction_check >= req_count_check) | ~has_requirements
            feature_block = is_pron_noun & ~fully_satisfied
            mask = pos_mask & ~feature_block
        else:
            mask = pos_mask
        
        # Self-attention always allowed
        diag = torch.eye(seq, dtype=torch.bool, device=device)
        mask = mask | diag.unsqueeze(0)
        
        scores = compatibility * mask.float()
        return mask, scores


# =============================================================================
# ATTENTION
# =============================================================================

class ASAAttention(nn.Module):
    """ASA Attention with ablation modes."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        alpha: float = 1.0,
        hard_block: bool = True,
        soft_penalty: float = 10.0,
        mode: AblationMode = 'full',
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.alpha = alpha
        self.hard_block = hard_block
        self.soft_penalty = soft_penalty
        self.mode = mode
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.bonding = BondingComputer(mode=mode)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        requirements: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq, _ = x.shape
        device = x.device
        
        self.bonding.to(device)
        
        Q = self.W_q(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # ASA adjustments (skip if mode='none' or no properties provided)
        if self.mode != 'none' and pos_ids is not None and features is not None and requirements is not None:
            bonding_mask, compatibility = self.bonding.compute_bonding_mask(pos_ids, features, requirements)
            
            if self.mode != 'pos_only':  # Add compatibility score
                scores = scores + self.alpha * compatibility.unsqueeze(1)
            
            # Apply mask
            if self.hard_block:
                scores = scores.masked_fill(~bonding_mask.unsqueeze(1), float('-inf'))
            else:
                penalty = self.soft_penalty * (~bonding_mask.unsqueeze(1)).float()
                scores = scores - penalty
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        if causal:
            causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        output = self.W_o(output)
        
        return (output, weights) if return_attention else (output, None)


class ASATransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 alpha: float = 1.0, hard_block: bool = True, soft_penalty: float = 10.0,
                 mode: AblationMode = 'full'):
        super().__init__()
        self.attention = ASAAttention(d_model, n_heads, dropout, alpha, hard_block, soft_penalty, mode)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    
    def forward(self, x, pos_ids=None, features=None, requirements=None, attention_mask=None, causal=True):
        attn_out, _ = self.attention(x, pos_ids, features, requirements, attention_mask, causal)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# =============================================================================
# MODEL
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ASALanguageModel(nn.Module):
    """ASA Transformer with ablation support."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        alpha: float = 1.0,
        hard_block: bool = True,
        soft_penalty: float = 10.0,
        mode: AblationMode = 'full',
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.mode = mode
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            ASATransformerBlock(d_model, n_heads, d_ff, dropout, alpha, hard_block, soft_penalty, mode)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        requirements: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)
        
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, pos_ids, features, requirements, attention_mask, causal)
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        result = {'logits': logits}
        
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
            )
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def compute_perplexity(self, input_ids, pos_ids=None, features=None, requirements=None) -> float:
        self.eval()
        result = self.forward(input_ids, pos_ids, features, requirements, causal=True, labels=input_ids)
        return torch.exp(result['loss']).item()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

MODEL_CONFIGS = {
    'tiny': dict(d_model=128, n_heads=2, n_layers=2, d_ff=512),
    'small': dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024),
    'medium': dict(d_model=512, n_heads=8, n_layers=6, d_ff=2048),
    'large': dict(d_model=768, n_heads=12, n_layers=12, d_ff=3072),
}


def create_model(
    vocab_size: int,
    size: str = 'small',
    mode: AblationMode = 'full',
    alpha: float = 1.0,
    hard_block: bool = True,
    soft_penalty: float = 10.0,
    **kwargs
) -> ASALanguageModel:
    """
    Create ASA model with specified ablation mode.
    
    Modes:
        'full': Complete ASA (POS + features + blocking)
        'pos_only': Only POS compatibility mask
        'features_only': Only feature compatibility scores
        'none': Standard transformer baseline
    """
    config = MODEL_CONFIGS.get(size, MODEL_CONFIGS['small']).copy()
    config.update(kwargs)
    return ASALanguageModel(
        vocab_size=vocab_size,
        mode=mode,
        alpha=alpha,
        hard_block=hard_block,
        soft_penalty=soft_penalty,
        **config
    )


def create_baseline(vocab_size: int, size: str = 'small', **kwargs) -> ASALanguageModel:
    """Create standard transformer (mode='none')."""
    return create_model(vocab_size, size, mode='none', **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================

class ASADataset:
    """Dataset with precomputed ASA properties."""
    
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        tokenizer,
        extractor: PropertyExtractor,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> 'ASADataset':
        aligner = SubwordAligner(tokenizer)
        data = []
        
        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Extracting ASA properties")
            except ImportError:
                pass
        
        for text in iterator:
            props = extractor.extract(text)
            result = aligner.align(text, props, max_length)
            data.append({
                'input_ids': result.input_ids,
                'pos_ids': result.pos_ids,
                'features': result.features,
                'requirements': result.requirements,
            })
        
        return cls(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def save(self, path: str):
        torch.save(self.data, path)
    
    @classmethod
    def load(cls, path: str) -> 'ASADataset':
        return cls(torch.load(path))


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "2.2.0"

CHANGELOG = """
v2.2.0:
- Added SubwordAligner for BPE/WordPiece tokenizers
- Added ablation modes: 'full', 'pos_only', 'features_only', 'none'
- Expanded verb coverage to ~300 verbs
- Added create_baseline() for standard transformer comparison
- Integrated subword alignment into ASADataset
- All previous v2.1 fixes included

v2.1.0:
- Fixed duplicate 'anyone' key
- Fixed PROPN assumption (now just CONCRETE)
- Removed arbitrary 0.5 in compatibility formula
- Added @lru_cache for WordNet
- Added hard_block/soft_penalty options
"""

# =============================================================================
# SPARSITY MEASUREMENT
# =============================================================================

def measure_sparsity(
    texts: List[str],
    extractor: Optional[PropertyExtractor] = None,
    mode: AblationMode = 'full',
    show_blocked_pairs: bool = False,
) -> Dict[str, float]:
    """
    Measure actual sparsity on a corpus.
    
    Returns:
        Dictionary with sparsity statistics
    """
    if extractor is None:
        extractor = PropertyExtractor()
    
    bonding = BondingComputer(mode=mode)
    
    total_pairs = 0
    blocked_pairs = 0
    blocked_by_pos = {}  # (pos_i, pos_j) -> count
    
    for text in texts:
        props = extractor.extract(text)
        n = len(props)
        
        if n < 2:
            continue
        
        pos_ids = torch.tensor([[p.pos_id for p in props]])
        features = torch.tensor([[p.features for p in props]])
        requirements = torch.tensor([[p.requirements for p in props]])
        
        mask, _ = bonding.compute_bonding_mask(pos_ids, features, requirements)
        
        # Count blocked pairs (excluding diagonal)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                total_pairs += 1
                if not mask[0, i, j]:
                    blocked_pairs += 1
                    if show_blocked_pairs:
                        pair = (props[i].pos, props[j].pos)
                        blocked_by_pos[pair] = blocked_by_pos.get(pair, 0) + 1
    
    sparsity = blocked_pairs / total_pairs if total_pairs > 0 else 0
    
    result = {
        'total_pairs': total_pairs,
        'blocked_pairs': blocked_pairs,
        'allowed_pairs': total_pairs - blocked_pairs,
        'sparsity': sparsity,
        'density': 1 - sparsity,
    }
    
    if show_blocked_pairs:
        # Top blocked POS pairs
        top_blocked = sorted(blocked_by_pos.items(), key=lambda x: -x[1])[:10]
        result['top_blocked_pos_pairs'] = top_blocked
    
    return result


def measure_pos_matrix_sparsity() -> Dict[str, float]:
    """
    Measure theoretical sparsity of POS compatibility matrix.
    
    This is the upper bound on sparsity from POS alone,
    before considering feature compatibility.
    """
    total = NUM_POS * NUM_POS
    connected = POS_COMPATIBILITY_MATRIX.sum().item()
    
    # Weighted by typical English POS distribution
    # Source: Brown corpus statistics
    pos_freq = {
        'NOUN': 0.20, 'VERB': 0.14, 'ADJ': 0.07, 'ADV': 0.05,
        'PRON': 0.08, 'PROPN': 0.04, 'DET': 0.10, 'ADP': 0.12,
        'AUX': 0.04, 'CCONJ': 0.03, 'SCONJ': 0.02, 'NUM': 0.02,
        'PART': 0.01, 'INTJ': 0.005, 'PUNCT': 0.07, 'SYM': 0.001, 'X': 0.004
    }
    
    weighted_connected = 0.0
    for pos_i, freq_i in pos_freq.items():
        for pos_j, freq_j in pos_freq.items():
            i, j = POS_TO_ID.get(pos_i, 16), POS_TO_ID.get(pos_j, 16)
            weighted_connected += freq_i * freq_j * POS_COMPATIBILITY_MATRIX[i, j].item()
    
    return {
        'raw_sparsity': 1 - (connected / total),
        'weighted_sparsity': 1 - weighted_connected,
        'raw_density': connected / total,
        'weighted_density': weighted_connected,
    }


# =============================================================================
# ALPHA TUNING GUIDANCE
# =============================================================================

ALPHA_GUIDANCE = """
Alpha (α) Tuning Guide
======================

α controls how much linguistic compatibility affects attention:
    Score = Q·K^T/√d + α·Compatibility

Recommended starting points:
- α = 0.5: Weak linguistic bias (let model learn more)
- α = 1.0: Balanced (default)
- α = 2.0: Strong linguistic bias (trust linguistics more)

Tuning strategy:
1. Train with α ∈ {0.5, 1.0, 2.0} on validation set
2. Pick α that minimizes perplexity
3. For downstream tasks, may need task-specific α

Expected behavior:
- Higher α → More sparse attention → Faster inference
- Lower α → Model can override linguistic constraints
- α = 0 with mode='none' → Standard transformer

Note: α could be made learnable, but this defeats the
"predetermined" premise. Better to tune as hyperparameter.
"""


# =============================================================================
# QUICK REFERENCE
# =============================================================================

def print_summary():
    """Print ASA summary."""
    pos_stats = measure_pos_matrix_sparsity()
    
    print(f"ASA v{__version__}")
    print(f"=" * 50)
    print(f"Verb coverage: {len(VERB_TO_CLASS)} verbs")
    print(f"Pronoun coverage: {len(PRONOUN_REQUIREMENTS)} pronouns")
    print(f"WordNet mappings: {len(SYNSET_TO_FEATURES)} synsets")
    print(f"")
    print(f"POS matrix sparsity:")
    print(f"  Raw (uniform): {pos_stats['raw_sparsity']:.1%}")
    print(f"  Weighted (English): {pos_stats['weighted_sparsity']:.1%}")
    print(f"")
    print(f"Ablation modes: full, pos_only, features_only, none")
    print(f"")
    print(f"Default α=1.0 (see ALPHA_GUIDANCE for tuning)")


# Run on import
print_summary()
