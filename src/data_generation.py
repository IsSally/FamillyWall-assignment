import pandas as pd
from random import choice
from .domain_patterns import industries, complexity_templates, cities, specialties, audiences, brand_voices, sample_tld, DOMAIN_PATTERNS
from .utils import ensure_dir
import os

def generate_synthetic_data(output_csv_path: str) -> pd.DataFrame:
  rows = []
  for industry, descs in industries.items():
      for desc in descs:
          tokens = desc.lower().split()
          for city in cities:#sample(cities, k=4):           # if you want to sample k cities per desc
              for level, tmpl in complexity_templates.items():
                  spec     = choice(specialties[industry])
                  aud      = choice(audiences)
                  bv       = choice(brand_voices)
                  biz_text = tmpl.format(
                      desc=desc, city=city,
                      specialty=spec, audience=aud, brand_voice=bv
                  )
                  # sample one pattern + TLD
                  pat    = choice(DOMAIN_PATTERNS)
                  name   = pat(tokens, city, industry,spec)
                  tld    = sample_tld(industry)
                  domain = f"{name}{tld}"

                  rows.append({
                      "complexity": level,
                      "industry": industry,
                      "business_description": biz_text,
                      "domain": domain
                  })
  df = pd.DataFrame(rows)
  ensure_dir(os.path.dirname(output_csv_path))
  df.to_csv(output_csv_path, index=False)
  return df
