import pandas as pd
from random import choice, sample

# Industry definitions & business descriptions
industries = {
    "tech":       ["AI consultancy", "cloud services", "cybersecurity firm", "autonomous vehicle software", "driver behavior analytics", "fleet optimization service", "smart security system", "predictive energy manager", "network anomaly monitoring"],
    "health":     ["telemedicine platform", "medical imaging diagnostics", "personalized health monitoring", "fitness studio"],
    "finance":    ["fraud detection service", "algorithmic trading platform", "credit scoring startup", "personal budgeting app", "investment advisory service", "cryptocurrency exchange"],
    "ecommerce":  ["product recommendation engine", "dynamic pricing tool", "visual search shopping app", "handcrafted jewelry store", "organic soap store", "pet supplies retailer"],
    "education":  ["adaptive learning system", "automated grading tool", "language learning chatbot", "online tutoring platform", "STEM learning center", "language school"],
    "legal":      ["contract review automation", "legal research assistant", "compliance monitoring tool", "family law firm", "intellectual property practice"],
    "marketing":  ["customer sentiment analyzer", "automated A/B testing tool", "targeted ad optimization platform"],
    "realestate": ["property price predictor", "AI-powered listing recommender", "tenant screening platform", "urban property developer", "luxury real estate broker", "vacation rental manager"],
    "travel":     ["personalized itinerary planner", "flight price prediction engine", "chatbot travel assistant", "boutique travel agency", "adventure tour operator", "luxury cruise planner"],
    "media":      ["music recommendation system", "automated video tagging tool", "game difficulty adaptation engine", "content personalization service"],
    "manufacturing": ["predictive maintenance system", "quality inspection with computer vision", "industrial process optimization"],
    "agriculture":   ["crop yield predictor", "drone-based field analysis", "smart irrigation system"],
    "environment":   ["climate risk modeling service", "wildlife tracking AI", "carbon footprint estimator"],
    "fashion":       ["sustainable clothing brand", "vintage apparel store", "designer footwear line"],
    "cafe":          ["coffee shop", "vegan bakery", "tea lounge"]
}

# Complexity templates (1=short, 2=moderate, 3=rich)
complexity_templates = {
    1: "A {desc} in {city}.",
    2: "A {desc} in {city}, specializing in {specialty}.",
    3: "A {desc} located in {city} that offers {specialty}, targets {audience}, and emphasizes {brand_voice}.",
}

# Context placeholders
cities       = ["Paris", "London", "Berlin", "Tokyo", "New York", "Rome", "Sydney", "Toronto", "Dubai", "Beirut", "Toulouse", "Istanbul", "Los Angelos", "Lille"]
specialties = {
    "tech":         ["cloud migrations", "penetration testing", "AI-driven analytics", "edge computing", "blockchain solutions"],
    "health":       ["virtual consultations", "group fitness programs", "wellness coaching", "remote patient monitoring", "AI-assisted diagnostics"],
    "finance":      ["real-time analytics", "risk assessment", "portfolio management", "fraud detection systems", "automated reporting"],
    "ecommerce":    ["handcrafted designs", "organic ingredients", "pet wellness products", "custom gifts", "personalized recommendations"],
    "education":    ["interactive lessons", "certified tutors", "gamified modules", "adaptive learning paths", "real-time progress tracking"],
    "legal":        ["estate planning", "trademark filings", "corporate compliance", "contract automation", "case law search tools"],
    "realestate":   ["market analysis", "staging services", "holiday rentals", "AI-driven property valuations", "virtual home tours"],
    "travel":       ["custom itineraries", "24/7 support", "off-the-beaten-path tours", "dynamic pricing engines", "language-aware assistants"],
    "fashion":      ["organic fabrics", "artisan craftsmanship", "limited editions", "AI-based style matching", "virtual fitting rooms"],
    "cafe":         ["single-origin pour-overs", "vegan pastries", "latte art classes", "fresh brew", "artisan beans", "organic roasts"],
    "media":        ["automated content tagging", "music personalization", "dynamic video previews", "AI-generated subtitles", "viewer engagement metrics"],
    "manufacturing": ["predictive maintenance", "automated defect detection", "robotic process control", "supply chain forecasting", "real-time quality assurance"],
    "agriculture":  ["precision irrigation", "disease detection via drones", "soil quality monitoring", "yield forecasting", "automated crop spraying"],
    "environment":  ["air quality monitoring", "climate risk modeling", "waste reduction analytics", "renewable energy optimization", "wildlife movement tracking"],
     "marketing": ["customer sentiment analysis", "automated campaign testing", "conversion rate optimization", "real-time engagement tracking", "predictive audience segmentation"
]
}

audiences    = ["millennials", "local foodies", "small businesses", "health-conscious clients", "students", "high-net-worth individuals"]
brand_voices = ["eco-friendly ethos", "luxury experience", "community focus", "tech-savvy vibe", "educational excellence", "financial empowerment"]

# Industry-specific adjectives
adjectives_by_industry = {
    "tech":         ["edge", "quantum", "secure", "smart", "digital", "scalable"],
    "health":       ["well", "fit", "vital", "holistic", "care", "healthy"],
    "finance":      ["secure", "trusted", "wealth", "capital", "insightful", "stable"],
    "ecommerce":    ["handmade", "custom", "organic", "boutique", "eco", "curated"],
    "education":    ["bright", "smart", "engaging", "knowledgeable", "adaptive", "scholarly"],
    "legal":        ["trusted", "prime", "secure", "expert", "legal", "compliant"],
    "realestate":   ["prime", "estate", "lux", "home", "urban", "residence"],
    "travel":       ["wander", "globe", "explore", "journey", "escape", "adventure"],
    "fashion":      ["chic", "vogue", "elegant", "couture", "stylish", "trend"],
    "cafe":         ["fresh", "artisan", "roasted", "barista", "urban", "cozy"],
    "media":        ["viral", "dynamic", "engaging", "immersive", "visual", "creative"],
    "manufacturing":["automated", "precise", "efficient", "industrial", "reliable", "smart"],
    "agriculture":  ["sustainable", "organic", "green", "rural", "fresh", "fertile"],
    "environment":  ["clean", "green", "climate-smart", "eco-friendly", "resilient", "sustainable"],
     "marketing": ["targeted","dynamic","insightful","creative","engaging","data-driven"]
}


# TLD pools (weighted by industry)
TLD_BY_INDUSTRY = {
    "tech":         [".io"]*5 + [".ai"]*5 + [".tech"]*3 + [".com", ".net"],
    "health":       [".health"]*5 + [".fit"]*3 + [".com"],
    "finance":      [".finance", ".invest", ".money", ".com", ".net"],
    "ecommerce":    [".shop"]*5 + [".store"]*5 + [".com", ".net", ".biz"],
    "education":    [".academy", ".edu", ".online", ".co"] + [".com"]*2,
    "legal":        [".law", ".legal", ".com", ".org"],
    "realestate":   [".estate", ".homes", ".property", ".com", ".net"],
    "travel":       [".travel", ".tours", ".vacations", ".com", ".net"],
    "fashion":      [".fashion", ".style", ".boutique", ".com", ".net"],
    "cafe":         [".com", ".net", ".coffee", ".cafe"],
    "media":        [".media", ".tv", ".video", ".studio", ".com"],
    "manufacturing":[ ".industry", ".engineering", ".systems", ".solutions", ".com"],
    "agriculture":  [".farm", ".organic", ".ag", ".green", ".com"],
    "environment":  [".eco", ".green", ".earth", ".solutions", ".org"],
     "marketing" : [".marketing", ".ads", ".media", ".agency", ".com"],
    "default":      [".com", ".net", ".org", ".app", ".co", ".dev", ".info", ".online", ".site"]
}


def sample_tld(industry):
    return choice(TLD_BY_INDUSTRY.get(industry, TLD_BY_INDUSTRY["default"]))

# Domain-pattern functions (now all take industry)
def pattern_hyphen(tokens, city, industry,*kwrags):
    return "-".join(tokens + [city.lower()])

def pattern_concat(tokens, city, industry,*kwrags):
    return "".join(tokens + [city.lower()])

def pattern_get(tokens, city, industry,*kwrags):
    return "-".join(["get"] + tokens + [city.lower()])

def pattern_hq(tokens, city, industry,*kwrags):
    return "".join(tokens) + "-hq"

def pattern_adj_noun_city(tokens, city, industry,*kwrags):
    # pick from this industry's adjectives
    adj = choice(adjectives_by_industry.get(industry, []))
    noun = tokens[0]
    return f"{adj}{noun.capitalize()}{city.lower()}"

DOMAIN_PATTERNS = [
    pattern_hyphen,
    pattern_concat,
    #pattern_get,
    #pattern_hq,
    pattern_adj_noun_city,
]

# “Semantic” pattern: industry-aware adjective + noun + city
def semantic_pattern(tokens, city, industry, spec_phrase):
    """
    Exactly the same as before, but uses the spec_phrase
    from biz_text instead of sampling from all specialties.
    """
    # 1. Take only the first word of the spec you already picked
    spec_first = spec_phrase.split()[0]
    # 2. Build the pool: industry adjectives + that one spec‐word
    pool = adjectives_by_industry.get(industry, []) + [spec_first]
    # 3. Sample your adjective (or that spec‐word)
    adj = choice(pool) if pool else ""
    # 4. Pick a noun‐like token from the description
    noun_tokens = [t for t in tokens if len(t) > 2]
    noun = choice(noun_tokens) if noun_tokens else tokens[0]
    # 5. Return the same concatenation as before
    return f"{adj}{noun}{city.lower()}"


DOMAIN_PATTERNS.append(semantic_pattern)

