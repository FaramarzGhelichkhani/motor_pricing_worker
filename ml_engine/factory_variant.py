"""Factory-specific canonical identity; leaves ads/used variant_v2 completely untouched.

One product: (catalog motorcycle_model_id, canonical model_variant).
Crawler motor_class is input evidence ONLY and never becomes a DB dimension.
Unknown/ambiguous classes fail closed, with explicit diagnostics.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

# The production project contains the authoritative map; these two maker aliases
# occur explicitly in the uploaded factory file. The normalizer accepts any
# additional brand aliases supplied by dictionary.mappings.BRAND_NAME_MAP.
FACTORY_MAKER_CLASSES = {
    'کبیر موتور': 'Kabir',
    'جهان همتا': 'Jahan Hamta',
}

EXTRA_ORDER = (
    'NO_ABS', 'ELECTRIC_START', 'KICK_START', 'REMOTE', 'KEYED',
    'SPOKE_RIM', 'ALLOY_RIM', 'LONG_HANDLEBAR', 'SHORT_HANDLEBAR',
    'TYPE_1', 'TYPE_2', 'TYPE_2S', 'TYPE_3', 'TYPE_4', 'TYPE_5', 'TYPE_P',
    'TYPE_R', 'TYPE_RR', 'FULL_OPTION', 'QUICKSHIFTER', 'TRIM_GT',
    'TRIM_SPORT', 'TRIM_ALLURE', 'TRIM_ACTIVE', 'TRIM_SE', 'TRIM_SE_NA',
    'ALPHA', 'SCOOPY_STYLE', 'HUNTER_STYLE', 'NEW_ODOMETER', 'YEAR_2023',
    'COLOR_GREEN', 'COLOR_WHITE_BLACK', 'COLOR_WHITE_BLACK_RED',
)

# Exact class meanings, after Unicode/spacing normalization. Every one of the
# uploaded normalized classes must map to a known meaning, or be rejected.
# Pairs also document non-obvious distinctions (e.g. KEYED != KEYLESS).
MANUAL_EXTRAS = {
    'ریموت دار': ('REMOTE',), 'ریموت دار با abs': ('REMOTE',),
    'تیپ 1': ('TYPE_1',), 'تیپ 2': ('TYPE_2',), 'تیپ 2s': ('TYPE_2S',),
    'تیپ p': ('TYPE_P',), 'تیپ 5': ('TYPE_5',),
    'تیپr': ('TYPE_R',), 'تیپ rr': ('TYPE_RR',),
    'تیپrrفول آپشن': ('TYPE_RR','FULL_OPTION'),
    'تیپ rrکوییک شیفتر': ('TYPE_RR','QUICKSHIFTER'),
    'سبز': ('COLOR_GREEN',),
    'سفید،مشکی': ('COLOR_WHITE_BLACK',),
    'سفید،مشکی،قرمز': ('COLOR_WHITE_BLACK_RED',),
    'gt': ('TRIM_GT',), 'sport': ('TRIM_SPORT',),
    'allure': ('TRIM_ALLURE',), 'active': ('TRIM_ACTIVE',),
    'آلفا': ('ALPHA',), 'آلفا توربو abs': ('ALPHA',),
    'طرح اسکوپی': ('SCOOPY_STYLE',),
    'کیلومتر جدید': ('NEW_ODOMETER',),
    'رینگ پره ای': ('SPOKE_RIM',),
    'رینگ آلومینیومی': ('ALLOY_RIM',),
    'فرمان بلند (تیپ2)': ('LONG_HANDLEBAR','TYPE_2'),
    'فرمان بلند+باکس (تیپ3)': ('LONG_HANDLEBAR','TYPE_3'),
    'فرمان کوتاه+باکس (تیپ4)': ('SHORT_HANDLEBAR','TYPE_4'),
    'هندلی': ('KICK_START',), 'هندلی+باکس دار': ('KICK_START',),
    'هندلی +abs': ('KICK_START',),
    'استارتی': ('ELECTRIC_START',), 'استارتی +abs': ('ELECTRIC_START',),
    'استارتی +باکس': ('ELECTRIC_START','BOX'),
    'abs استارتی': ('ELECTRIC_START',),
    'شکاری': ('HUNTER_STYLE',), 'شکاری +abs': ('HUNTER_STYLE',),
    'se/na': ('TRIM_SE_NA',), 'se': ('TRIM_SE',),
    'بدونabs': ('NO_ABS',), 'بدون abs': ('NO_ABS',),
    'سوئیچ std': ('KEYED',),
    '2023': ('YEAR_2023',),
}

# Other observed class values are fully represented by variant_v2's feature
# detector. Restrict this set: an unknown future class must not silently map
# to DEFAULT and merge distinct products/prices.
FEATURE_ONLY_CLASSES = {
    '', 'استاندارد', 'std', 'abs', 'آب خنک', 'هوا خنک', 'هواخنک',
    'آب خنک با abs', 'هوا خنک با abs', 'هواخنک با abs',
    'آب خنک | abs', 'هوا خنک | abs', 'abs هوا خنک',
    'باکس دار', 'باکس دار با abs',
    'رینگ اسپرت', 'رینگ اسپرت +باکس دار',
    'رینگ پره ای +باکس دار',
    'فیس جدید',
    'kls', 'کی لس', 'کیلس', 'کیلس + abs',
    'اندونزی', 'تایلند',
}
# Source typo ABC vs ABS, only in these explicit catalog classes. Audited
# as an alias, not silently treated as a new unrelated factory trim.
ABC_ALIASES = {'abc+اندونزی', 'abc+تایلند'}
# The class contains a redundant SKU also present in the same product title/model.
CLASS_SKU_RE = re.compile(r'(?i)^(?:abs\s+([a-z]{1,5}\s*\d{2,4})|([a-z]{1,5}\s*\d{2,4})\s*\|\s*abs)$')


def normalize(value):
    if value is None or pd.isna(value): return ''
    s = unicodedata.normalize('NFKC',str(value)).replace('ي','ی').replace('ك','ک')
    s = s.replace('ـ','').replace('\u200c',' ').strip().casefold()
    s = re.sub(r'\s+', ' ', s)
    return s


def compact(value):
    return ''.join(re.findall(r'[a-z0-9]+', normalize(value)))


def normalize_brand(value, brand_aliases=None):
    key = normalize(value)
    aliases = {
        normalize(x): y for x,y in (brand_aliases or {}).items()
        if isinstance(x,str) and isinstance(y,str) and x.strip() and y.strip()
    }
    # Resolve localized and ASCII map values, as well as the observed maker name.
    aliases.update({normalize(k):v for k,v in FACTORY_MAKER_CLASSES.items()})
    canonical = aliases.get(key, value)
    return normalize(canonical)


@dataclass(frozen=True)
class FactoryIdentity:
    brand: str
    model_variant: str
    status: str = 'ok'
    note: str = ''


def class_brand_override(class_value, brand_aliases=None):
    """Return canonical maker when the entire class is an exact brand alias.

    Known factory feature/trim classes take precedence, so a feature cannot
    accidentally be interpreted as a brand. Includes company aliases found in
    dictionary.mappings.BRAND_NAME_MAP, not just the two observed in the CSV.
    """
    cl = normalize(class_value)
    if cl in FACTORY_MAKER_CLASSES:
        return normalize_brand(FACTORY_MAKER_CLASSES[cl],brand_aliases)
    if (not cl or cl in MANUAL_EXTRAS or cl in FEATURE_ONLY_CLASSES
        or cl in ABC_ALIASES or CLASS_SKU_RE.fullmatch(cl)):
        return None
    aliases = {normalize(k):v for k,v in (brand_aliases or {}).items()
               if isinstance(k,str) and isinstance(v,str)}
    if cl in aliases:
        return normalize_brand(aliases[cl],brand_aliases)
    return None


def variant_from_sources(row, base_variant, conflict, brand_aliases=None):
    """Create a stable factory variant from the existing v2 extraction + class.

    ``base_variant`` is obtained by calling existing _set_model_variant over
    raw_title + motor_class, so its feature and order compatibility is retained.
    """
    if conflict:
        return FactoryIdentity('', '', 'conflicting_variant', 'conflicting cylinder evidence')
    source_brand = normalize_brand(row.get('brand_name'),brand_aliases)
    cl = normalize(row.get('motor_class'))
    brand_override = class_brand_override(cl,brand_aliases)
    if brand_override:
        resolved_brand = brand_override
        extras = ()
    else:
        resolved_brand = source_brand
        extras = MANUAL_EXTRAS.get(cl, ())

    note = ''
    if cl in ABC_ALIASES:
        note = 'source_abc_corrected_to_abs'
    elif cl and cl != 'استاندارد' and cl not in FEATURE_ONLY_CLASSES and cl not in MANUAL_EXTRAS and not brand_override:
        m = CLASS_SKU_RE.fullmatch(cl)
        if m:
            sku = compact(next(g for g in m.groups() if g is not None))
            title_and_model = compact(str(row.get('raw_title') or '') + ' ' + str(row.get('model_name') or ''))
            if sku not in title_and_model:
                return FactoryIdentity(resolved_brand, '', 'unknown_class',f'sku {sku} not found in title/model')
        elif cl not in ABC_ALIASES:
            return FactoryIdentity(resolved_brand, '', 'unknown_class',cl)

    # Correct the two explicitly observed ABC factory class typos. If title itself
    # also contains features, base_variant remains useful; class is authoritative
    # only for correcting these two finite class forms.
    features = set(str(base_variant or 'DEFAULT').split('+')) - {'DEFAULT',''}
    if cl in ABC_ALIASES:
        features.add('ABS')
    features.update(extras)
    if 'NO_ABS' in features:
        # variant_v2's canonical-token detector sees ABS even in 'بدون ABS'.
        # Negated class explicitly overrides that token unless the *title* also
        # asserts ABS, which is a true conflict that must not be guessed away.
        title = normalize(row.get('raw_title'))
        positive_title_abs = bool(re.search(r'(?<!\w)abs(?!\w)|ای[\s‌\-]*بی[\s‌\-]*اس', title))
        if positive_title_abs:
            return FactoryIdentity(resolved_brand, '', 'conflicting_variant','NO_ABS class contradicts ABS title')
        features.discard('ABS')
    if cl in {'سوئیچ std'} and 'KEYLESS' in features:
        return FactoryIdentity(resolved_brand,'','conflicting_variant','keyed and keyless')
    # v2 already keeps factory feature order; keep it for backward compatibility.
    from collections import OrderedDict
    from ml_engine.variant_labeling_v2 import MODEL_VARIANT_ORDER
    ordered = [k for k in MODEL_VARIANT_ORDER if k in features]
    ordered += [k for k in EXTRA_ORDER if k in features]
    residual = features.difference(ordered)
    if residual:
        return FactoryIdentity(resolved_brand, '', 'unknown_feature', ','.join(sorted(residual)))
    return FactoryIdentity(resolved_brand, '+'.join(ordered) or 'DEFAULT', 'ok', note)


def brand_for_row(row, brand_aliases=None):
    cl = normalize(row.get('motor_class'))
    return class_brand_override(cl,brand_aliases) or normalize_brand(row.get('brand_name'),brand_aliases)


def audit_coverage_of_classes(values):
    """Return normalized classes that require a new rule before production sync."""
    unknown = {}
    for value in values:
        cl = normalize(value)
        if (cl in FEATURE_ONLY_CLASSES or cl in MANUAL_EXTRAS or cl in FACTORY_MAKER_CLASSES
            or cl in ABC_ALIASES or CLASS_SKU_RE.fullmatch(cl)):
            continue
        unknown[cl] = unknown.get(cl, 0) + 1
    return unknown
