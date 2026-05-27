from flask import Flask, request, jsonify, send_from_directory
import re
import requests
import json
import os
import unicodedata
import concurrent.futures

app = Flask(__name__, static_folder="static")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def is_time_token(s):
    return bool(re.search(r'\b([01]?\d|2[0-3])[:.][0-5]\d\b', s))

def is_ordinal_or_decimal_context(s, digit):
    if re.search(r'(?<!\d)' + re.escape(digit) + r'\.$', s.strip()):
        return True
    if re.search(r'(?<!\d)' + re.escape(digit) + r'[,\.]\d', s):
        return True
    if re.search(r'(?<!\d)' + re.escape(digit) + r':', s):
        return True
    return False

def dedupe_issues(issues):
    seen = set()
    result = []
    for issue in issues:
        key = (issue.get("line"), issue.get("rule"))
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result

def split_subtitle_blocks(lines):
    blocks = []
    current = []
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            if current:
                blocks.append({"lines": current, "start_lineno": start})
                current = []
                start = None
        else:
            if start is None:
                start = i + 1
            current.append((i + 1, line.strip()))
    if current:
        blocks.append({"lines": current, "start_lineno": start})
    return blocks


# ─────────────────────────────────────────────
#  MECHANICAL RULE CHECKS
# ─────────────────────────────────────────────

def check_mechanical_rules(text):
    issues = []
    raw_lines = text.split("\n")
    blocks = split_subtitle_blocks(raw_lines)

    for i, line in enumerate(raw_lines):
        lineno = i + 1
        s = line.strip()
        if not s:
            continue

        # TIME FORMAT
        if re.search(r'\b([01]?\d|2[0-3]):[0-5]\d\b', s):
            issues.append({
                "line": lineno, "rule": "Kellonajan muoto",
                "original": s,
                "suggestion": re.sub(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', r'\1.\2', s),
                "explanation": "Kellonajassa käytetään pistettä erottimena, ei kaksoispistettä. Esim. '16.23' ei '16:23'."
            })

        # LEADING ZERO IN TIME
        if re.search(r'\b0([1-9])\.\d{2}\b', s):
            issues.append({
                "line": lineno, "rule": "Etunolla kellonajassa",
                "original": s,
                "suggestion": re.sub(r'\b0([1-9])(\.\d{2})\b', r'\1\2', s),
                "explanation": "Kellonaikoihin ei merkitä etunollia. '6.23' ei '06.23'. Poikkeuksena aamuyön tunnit (00.xx)."
            })

        # NUMBERS 1-10, 100, 1000 -> words
        if not is_time_token(s):
            digit_words = [
                ('1000', 'tuhat'), ('100', 'sata'), ('10', 'kymmenen'),
                ('9', 'yhdeksän'), ('8', 'kahdeksan'), ('7', 'seitsemän'),
                ('6', 'kuusi'), ('5', 'viisi'), ('4', 'neljä'),
                ('3', 'kolme'), ('2', 'kaksi'), ('1', 'yksi')
            ]
            suggestion = s
            substitutions = []
            for digit, word in digit_words:
                pattern = r'(?<!\d)' + re.escape(digit) + r'(?!\d)'
                for m in re.finditer(pattern, suggestion):
                    context = suggestion[max(0, m.start()-2):m.end()+3]
                    after = suggestion[m.end():m.end()+2]
                    before = suggestion[max(0, m.start()-1):m.start()]
                    if re.search(r'^[e€\$%‰]', after) or re.search(r'[\$€%‰]$', before):
                        break
                    if not is_ordinal_or_decimal_context(context, digit):
                        suggestion = suggestion[:m.start()] + word + suggestion[m.end():]
                        substitutions.append(f"'{digit}' -> '{word}'")
                        break
            if substitutions:
                issues.append({
                    "line": lineno, "rule": "Luvun kirjoitusmuoto",
                    "original": s, "suggestion": suggestion,
                    "explanation": "Luvut 1-10, 100 ja 1000 kirjoitetaan tekstityksessä sanoin. " + ", ".join(substitutions) + "."
                })

        # THOUSANDS GROUPING
        thousands_match = re.search(r'\b(\d{4,})\b', s)
        if thousands_match:
            num_str = thousands_match.group(1)
            if ' ' not in num_str:
                formatted = re.sub(r'(\d)(?=(\d{3})+$)', r'\1 ', num_str)
                issues.append({
                    "line": lineno, "rule": "Tuhansien ryhmittely",
                    "original": s,
                    "suggestion": s.replace(num_str, formatted),
                    "explanation": f"Tuhannesta ylöspäin numerot ryhmitellään kolmen numeron ryhmiin välilyönnillä. '{num_str}' -> '{formatted}'."
                })

        # CURRENCY SYMBOLS
        currency_patterns = [
            (r'\$\s*(\d[\d\s]*)',   r'\1 dollaria', 'Dollarin merkki -> kirjoita auki "dollaria".'),
            (r'(\d[\d\s]*)\s*\$',  r'\1 dollaria', 'Dollarin merkki -> kirjoita auki "dollaria".'),
            (r'€\s*(\d[\d\s]*)',    r'\1 euroa',    'Euron merkki -> kirjoita auki "euroa".'),
            (r'(\d[\d\s,]*)\s*€',  r'\1 euroa',    'Euron merkki -> kirjoita auki "euroa".'),
            (r'(\d)\s*e\b',        r'\1 euroa',    '"3e" -> kirjoita auki "euroa".'),
        ]
        for pat, repl, expl in currency_patterns:
            if re.search(pat, s):
                issues.append({
                    "line": lineno, "rule": "Valuuttamerkki",
                    "original": s, "suggestion": re.sub(pat, repl, s),
                    "explanation": f"Valuuttamerkkejä ei käytetä tekstityksessä. {expl}"
                })
                break

        # PERCENT / PER MILLE
        if re.search(r'\d\s*%', s):
            issues.append({
                "line": lineno, "rule": "Prosenttimerkki",
                "original": s,
                "suggestion": re.sub(r'(\d)\s*%', r'\1 prosenttia', s),
                "explanation": "Prosentti kirjoitetaan auki vuorosanoissa: 'prosenttia' ei '%'."
            })
        if re.search(r'\d\s*‰', s):
            issues.append({
                "line": lineno, "rule": "Promillemerkki",
                "original": s,
                "suggestion": re.sub(r'(\d)\s*‰', r'\1 promillea', s),
                "explanation": "Promille kirjoitetaan auki vuorosanoissa: 'promillea' ei '‰'."
            })

        # TITLE ABBREVIATIONS
        if re.search(r'\b(tri|hra|rva|nti|os\.)\b', s, re.IGNORECASE):
            issues.append({
                "line": lineno, "rule": "Tittelin lyhenne",
                "original": s, "suggestion": s,
                "explanation": "Titteliä vastaavat lyhenteet (tri, hra, rva, nti) kirjoitetaan auki tai jätetään pois tekstityksessä."
            })

        # WRITTEN ABBREVIATIONS
        if re.search(r'\b(esim\.|mm\.|jne\.|ns\.|ko\.|em\.)\b', s):
            issues.append({
                "line": lineno, "rule": "Kirjoitettu lyhenne",
                "original": s, "suggestion": s,
                "explanation": "Lyhenteet kuten 'esim.', 'mm.', 'jne.' kirjoitetaan auki tekstityksessä."
            })

        # EM DASH
        if '—' in s:
            issues.append({
                "line": lineno, "rule": "Pitkä viiva",
                "original": s, "suggestion": s,
                "explanation": "Pitkää viivaa '—' ei käytetä tekstityksessä. Muotoile lause uudelleen tai käytä pilkkua / pistettä."
            })

        # ?! COMBINATION
        if '?!' in s or '!?' in s:
            issues.append({
                "line": lineno, "rule": "Välimerkkiyhdistelmä ?!",
                "original": s,
                "suggestion": s.replace('?!', '?').replace('!?', '?'),
                "explanation": "Kysymys- ja huutomerkin yhdistelmää '?!' ei pitäisi käyttää suomalaisessa tekstityksessä."
            })

        # ELLIPSIS + !? COMBINATIONS
        if re.search(r'…[!?]', s):
            issues.append({
                "line": lineno, "rule": "Ellipsin ja välimerkin yhdistelmä",
                "original": s, "suggestion": s,
                "explanation": "Kolmen pisteen ja huuto- tai kysymysmerkin yhdistelmää käytetään erittäin harkiten."
            })

        # SEMICOLON IN DIALOGUE
        if ';' in s:
            issues.append({
                "line": lineno, "rule": "Puolipiste dialogissa",
                "original": s, "suggestion": s.replace(';', '.'),
                "explanation": "Puolipistettä tulee välttää puhuttua kieltä simuloivassa dialogissa."
            })

        # CONTINUATION MARK spacing
        if s.endswith('-') and len(s) > 1 and s[-2] != ' ':
            issues.append({
                "line": lineno, "rule": "Jatkoviivan välilyönti",
                "original": s, "suggestion": s[:-1] + ' -',
                "explanation": "Jatkoviivan eteen tulee välilyönti: 'sana -' ei 'sana-'."
            })

        # DIALOGUE MARK spacing
        if re.match(r'^-[^\s\-]', s):
            issues.append({
                "line": lineno, "rule": "Vuorosanaviivan välilyönti",
                "original": s, "suggestion": '- ' + s[1:],
                "explanation": "Vuorosanaviivan jälkeen tulee välilyönti: '- Teksti' ei '-Teksti'."
            })

        # CONTRADICTORY END MARK + CONTINUATION MARK
        if re.search(r'[.!?]\s*-\s*$', s):
            issues.append({
                "line": lineno, "rule": "Ristiriitainen jatkoviiva",
                "original": s, "suggestion": s,
                "explanation": "Repliikki ei voi päättyä sekä lauseen loppumerkillä (.!?) että jatkoviivalla (-). Käytä joko toista tai muotoile lause toisin."
            })

        # IMPERIAL / NON-METRIC UNITS
        imperial_units = [
            (r'\b\d[\d\.,]*\s*(miles?|mailia)\b',        'Mailia -> muunna kilometreiksi.'),
            (r'\b\d[\d\.,]*\s*(feet|foot|jalka[a]?)\b',  'Jalka/feet -> muunna metreiksi.'),
            (r'\b\d[\d\.,]*\s*(inch(es)?|tuuma[a]?)\b',  'Tuumaa -> muunna senttimetreiksi.'),
            (r'\b\d[\d\.,]*\s*(pounds?|punta[a]?)\b',    'Puntaa -> muunna kilogrammoiksi.'),
            (r'\b\d[\d\.,]*\s*(stone[s]?)\b',            'Stone -> muunna kilogrammoiksi.'),
            (r'\b\d[\d\.,]*\s*(°F|fahrenheit)\b',        'Fahrenheit -> muunna Celsiukseksi.'),
            (r'\b\d[\d\.,]*\s*(mph|miles per hour)\b',   'Mph -> muunna km/h-yksikköön.'),
            (r'\b\d[\d\.,]*\s*(gallons?|gallona[a]?)\b', 'Gallonaa -> muunna litroiksi.'),
            (r'\b\d[\d\.,]*\s*(yard[s]?|jardi[a]?)\b',  'Jardia -> muunna metreiksi.'),
        ]
        for pat, expl in imperial_units:
            if re.search(pat, s, re.IGNORECASE):
                issues.append({
                    "line": lineno, "rule": "Mittayksikkö",
                    "original": s, "suggestion": s,
                    "explanation": f"Mittayksiköt muunnetaan suomalaisiin yksiköihin tekstityksessä. {expl}"
                })
                break

        # BRACKETS IN DIALOGUE
        if re.search(r'[\(\)\[\]]', s):
            issues.append({
                "line": lineno, "rule": "Sulkeet dialogissa",
                "original": s, "suggestion": s,
                "explanation": "Sulkeita ei yleensä käytetä tekstityksessä, koska ne rikkovat luonnollisen puheen illuusion. Muotoile lause toisin."
            })

        # MULTIPLE EXCLAMATION MARKS
        if re.search(r'!{2,}', s):
            issues.append({
                "line": lineno, "rule": "Useita huutomerkkejä",
                "original": s,
                "suggestion": re.sub(r'!{2,}', '!', s),
                "explanation": "Useita peräkkäisiä huutomerkkejä ei käytetä tekstityksessä. Käytä enintään yhtä huutomerkkiä."
            })

        # COLON OVERUSE
        if re.search(r'\w:\s+\w', s) and not re.search(r'(sanoi|totesi|kysyi|huusi|kuiskasi|vastasi)\s*:', s, re.IGNORECASE):
            if not re.search(r'\b\d{1,2}:\d{2}\b', s):
                issues.append({
                    "line": lineno, "rule": "Kaksoispisteen käyttö",
                    "original": s, "suggestion": s,
                    "explanation": "Kaksoispistettä tulee käyttää harkiten. Se sopii suoran lainauksen tai luettelon edelle, muuten suosi muuta rakennetta."
                })

    # ── PER-BLOCK CHECKS ──
    for block in blocks:
        block_lines = block["lines"]
        text_lines = [t for _, t in block_lines]
        block_text = "\n".join(text_lines)
        first_lineno = block_lines[0][0]

        non_empty = [t for t in text_lines if t.strip()]
        if len(non_empty) > 2:
            issues.append({
                "line": first_lineno, "rule": "Liikaa rivejä repliikissä",
                "original": block_text, "suggestion": block_text,
                "explanation": f"Repliikissä on {len(non_empty)} riviä. Enintään 2 riviä per repliikki sallitaan."
            })

        full_text = " ".join(text_lines)
        clean_text = re.sub(r'\s-$', '', full_text)
        sentence_endings = re.findall(r'[.!?](?=\s|$)', clean_text)
        if len(sentence_endings) > 2:
            issues.append({
                "line": first_lineno, "rule": "Liikaa virkkeitä repliikissä",
                "original": block_text, "suggestion": block_text,
                "explanation": f"Repliikissä on enemmän kuin 2 virkettä. Enintään 2 virkettä per repliikki sallitaan."
            })

        speaker_turns_total = len(re.findall(r'(?m)^-\s', block_text))
        if speaker_turns_total > 2:
            issues.append({
                "line": first_lineno, "rule": "Liikaa puhujia",
                "original": block_text, "suggestion": block_text,
                "explanation": "Yhdessä repliikissä ei tulisi olla enemmän kuin kahden puhujan vuorosanoja."
            })

        for idx in range(len(text_lines) - 1):
            this_line = text_lines[idx]
            next_line = text_lines[idx + 1]
            this_lineno = block_lines[idx][0]
            if (this_line.endswith('-') and len(this_line) > 1 and
                    this_line[-2] != ' ' and next_line and next_line[0].islower()):
                issues.append({
                    "line": this_lineno, "rule": "Sana jaettu eri riveille",
                    "original": this_line + " / " + next_line,
                    "suggestion": this_line.rstrip('-') + next_line,
                    "explanation": "Sanaa ei suositella jakamaan eri riveille tekstityksessä."
                })

        # LINE LENGTH BALANCE
        if len(non_empty) == 2:
            first_is_dialogue = non_empty[0].startswith('- ')
            second_is_dialogue = non_empty[1].startswith('- ')
            if not first_is_dialogue and not second_is_dialogue:
                if len(non_empty[0]) > len(non_empty[1]) + 10:
                    issues.append({
                        "line": first_lineno, "rule": "Rivijakauman epätasapaino",
                        "original": block_text, "suggestion": block_text,
                        "explanation": (
                            f"Ensimmäinen rivi ({len(non_empty[0])}) on huomattavasti pidempi kuin toinen ({len(non_empty[1])}). "
                            "Suositus: ensimmäinen rivi tulisi olla lyhyempi."
                        )
                    })

        # LINE BREAK SPLITTING NOUN/VERB PHRASE
        if len(non_empty) == 2:
            line1 = non_empty[0].rstrip(' -')
            line2 = non_empty[1]
            binding_words = r'\b(on|ei|se|ne|hän|he|ja|tai|vai|että|kun|jos|kuin|niin|myös|vaan|mutta|sekä|kuitenkin|jo|vielä|enää|aina|siis|koska|vaikka|jotta|ennen|jälkeen|kanssa|vuoksi|takia)\s*$'
            if re.search(binding_words, line1, re.IGNORECASE):
                issues.append({
                    "line": first_lineno, "rule": "Lauserajan rikkominen",
                    "original": block_text, "suggestion": block_text,
                    "explanation": "Rivinvaihto katkaisee lausekkeen keskeisen sidesanan jälkeen. Suosi rivinvaihtoa lauserajan tai luontevan tauon kohdalla."
                })

        # QUOTATION MARK CONTINUATION ACROSS SUBTITLES
        joined = " ".join(text_lines)
        open_quotes = joined.count('"')
        if open_quotes % 2 != 0:
            issues.append({
                "line": first_lineno, "rule": "Lainausmerkit useassa repliikissä",
                "original": block_text, "suggestion": block_text,
                "explanation": "Lainaus jatkuu yli repliikkirajojen. Lainausmerkit tulee toistaa jokaisen virkkeen alussa ja lopussa."
            })

        # SENTENCE STARTING LOWERCASE
        for idx, (lno, tline) in enumerate(block_lines):
            stripped = tline.strip()
            if stripped.startswith('-') or stripped.startswith('…'):
                continue
            if idx == 0 and stripped and stripped[0].islower():
                issues.append({
                    "line": lno, "rule": "Pieni alkukirjain repliikissä",
                    "original": stripped, "suggestion": stripped[0].upper() + stripped[1:],
                    "explanation": "Repliikki alkaa pienellä kirjaimella ilman jatkoviivaa. Tarkista onko kyseessä tarkoituksellinen jatko vai kirjoitusvirhe."
                })

    # ── MULTI-BLOCK CHECKS ──

    # ELLIPSIS CONTINUATION PAIRING
    for idx in range(len(blocks) - 1):
        current_lines = [t for _, t in blocks[idx]["lines"]]
        next_lines = [t for _, t in blocks[idx + 1]["lines"]]
        if not current_lines or not next_lines:
            continue
        last_line = current_lines[-1]
        first_next = next_lines[0]
        current_lineno = blocks[idx]["lines"][-1][0]
        if last_line.endswith('…') and not re.search(r'…[!?]$', last_line):
            if not first_next.startswith('…') and not first_next.startswith('- …'):
                issues.append({
                    "line": current_lineno, "rule": "Ellipsin jatkuminen puuttuu",
                    "original": last_line + " -> " + first_next,
                    "suggestion": last_line + " -> …" + first_next,
                    "explanation": "Kun repliikki katkeaa ellipsiin (…) ja jatkuu seuraavassa, seuraavan tulee alkaa ellipsillä: '…jatkuu'."
                })

    # SENTENCE SPANNING MORE THAN 3 SUBTITLES
    consecutive_continuation = 0
    continuation_start_lineno = None
    for idx, block in enumerate(blocks):
        block_text_lines = [t for _, t in block["lines"]]
        last_line = block_text_lines[-1] if block_text_lines else ""
        if re.search(r'\s-$', last_line):
            consecutive_continuation += 1
            if continuation_start_lineno is None:
                continuation_start_lineno = block["lines"][-1][0]
            if consecutive_continuation >= 3:
                issues.append({
                    "line": continuation_start_lineno, "rule": "Lause liian monessa repliikissä",
                    "original": last_line, "suggestion": last_line,
                    "explanation": "Sama lause jatkuu yli kolmessa repliikissä. Enintään 3 repliikkiä sallitaan."
                })
                consecutive_continuation = 0
                continuation_start_lineno = None
        else:
            consecutive_continuation = 0
            continuation_start_lineno = None

    # CONTINUATION HYPHEN MISUSE
    for idx in range(len(blocks) - 1):
        current_block_lines = [t for _, t in blocks[idx]["lines"]]
        next_block_lines = [t for _, t in blocks[idx + 1]["lines"]]
        if not current_block_lines or not next_block_lines:
            continue
        last_line = current_block_lines[-1]
        first_next = next_block_lines[0]
        current_lineno = blocks[idx]["lines"][-1][0]
        if (re.search(r'\s-$', last_line) and
                first_next and first_next[0].isupper() and
                not first_next.startswith('- ')):
            issues.append({
                "line": current_lineno, "rule": "Jatkoviivan mahdollinen virhe",
                "original": last_line + " -> " + first_next,
                "suggestion": last_line,
                "explanation": "Jatkoviiva viittaa lauseen jatkumiseen, mutta seuraava repliikki alkaa isolla kirjaimella. Tarkista onko jatkoviiva tarpeellinen."
            })

    return dedupe_issues(issues)


# ─────────────────────────────────────────────
#  LLM CHECK
# ─────────────────────────────────────────────

LLM_VALID_RULES = {
    "Turhat pronominit",
    "Epäselvä hän-viittaus",
    "Ylikotoistaminen",
    "Ellipsin väärinkäyttö",
    "Väärä kysymysmuoto",
    "Murteen ylilitterointi",
    "Lainaukset useassa repliikissä",
}

LLM_RULE_CANONICAL = {r.lower(): r for r in LLM_VALID_RULES}

LLM_REVIEW_ONLY_RULES = {
    "Epäselvä hän-viittaus",
    "Ylikotoistaminen",
    "Ellipsin väärinkäyttö",
    "Väärä kysymysmuoto",
    "Murteen ylilitterointi",
    "Lainaukset useassa repliikissä",
}

LLM_SAFE_RULES = {
    "Turhat pronominit",
}

SYSTEM_PROMPT = """Olet suomalainen tekstitysten laaduntarkistaja.
Tehtäväsi: löydä VAIN alla luetellut 7 virhettä. ÄLÄ liputa mitään muuta.

TÄRKEÄÄ:
- Liputa vain selvät tapaukset. Epävarmoissa ÄLÄ liputa.
- Puhekieli (sulla, mulla, sä, joo, okei) on AINA hyväksyttävää.
- Mekaaniset säännöt (numerot, kellonajat, valuutat, välimerkit, rivimäärät) on jo tarkistettu — ÄLÄ liputa niitä.

SALLITUT 7 SÄÄNNÖT — käytä TÄSMÄLLEEN näitä nimiä "rule"-kentässä:

1. "Turhat pronominit"
   Minä/sinä ovat turhia kun lause ei painota tekijää.
   ✓ "Minä olen väsynyt" → "Olen väsynyt"
   ✗ EI: "Minä teen sen, en sinä" (painotus on tarkoituksellinen)
   ✗ EI: hän/he/se — niitä ei koskaan liputa

2. "Epäselvä hän-viittaus"
   Vain kun SAMASSA lauseessa on useita henkilöitä JA hän on aidosti epäselvä.
   ✓ "Bob on Janen naapuri. Hän rakastaa häntä." (kumpi?)
   ✗ EI: lauseita joissa on vain yksi henkilö

3. "Ylikotoistaminen"
   Vieraan kulttuurin asia on korvattu suomalaisella vastineella.
   ✓ "Saturday Night Live" on korvattu "Putous"
   ✗ EI: tavallisia käännöksiä tai selityksiä

4. "Ellipsin väärinkäyttö"
   … on käytetty tavalliseen taukoon, ei aitoon epäröintiin tai katkaisuun.
   ✓ "Kyllä… mutta ehkä." (tavallinen tauko)
   ✗ EI: "Hän on… sinun isäsi." (aito dramaattinen tauko — ÄLÄ liputa)

5. "Väärä kysymysmuoto"
   Lauseessa on ? mutta EI -ko/-kö eikä kysymyssanaa.
   ✓ "Kolmelta?" → "Kolmeltako?"
   ✗ EI: lauseet joissa jo on -ko/-kö tai kysymyssana (kuka, missä, milloin…)
   ✗ EI: lauseet joissa EI ole kysymysmerkkiä — tätä sääntöä SAA käyttää VAIN ?-lauseisiin

6. "Murteen ylilitterointi"
   ✓ Vain jos KOKO teksti on niin murteellista että lukeminen on vaikeaa
   ✗ EI: yksittäiset murresanat

7. "Lainaukset useassa repliikissä"
   Lainaus jatkuu yli repliikin ilman lainausmerkkejä jokaisessa virkkeessä.
   ✓ Repliikki alkaa lainauksen keskeltä ilman avaavaa lainausmerkkiä

Vastaa VAIN JSON-taulukkona, ei mitään muuta tekstiä:
[{"line": <rivinumero>, "rule": "<yksi 7 säännöstä>", "original": "<alkuperäinen rivi>", "suggestion": "<korjattu rivi>", "explanation": "<yksi lause suomeksi>"}]

Jos virheitä ei löydy: []

MUISTA: "rule"-kentässä SAA olla VAIN yksi näistä arvoista: "Turhat pronominit", "Epäselvä hän-viittaus", "Ylikotoistaminen", "Ellipsin väärinkäyttö", "Väärä kysymysmuoto", "Murteen ylilitterointi", "Lainaukset useassa repliikissä"."""


def chunk_by_blocks(text, max_blocks=5):
    raw_lines = text.split("\n")
    blocks = []
    in_block = False
    block_start = 0
    for i, line in enumerate(raw_lines):
        if line.strip():
            if not in_block:
                block_start = i
                in_block = True
        else:
            if in_block:
                blocks.append((block_start, i - 1))
                in_block = False
    if in_block:
        blocks.append((block_start, len(raw_lines) - 1))

    chunks = []
    for i in range(0, len(blocks), max_blocks):
        group = blocks[i:i + max_blocks]
        first_line = group[0][0]
        last_line = group[-1][1]
        chunk_lines = raw_lines[first_line:last_line + 1]
        chunks.append({
            "text": "\n".join(chunk_lines),
            "line_offset": first_line
        })
    return chunks


def _validate_llm_issue(issue, source_lines):
    rule_raw = issue.get("rule", "").strip()
    canonical = LLM_RULE_CANONICAL.get(rule_raw.lower())
    if not canonical:
        return None

    lineno = issue.get("line")

    # FIX: coerce string line numbers to int
    if isinstance(lineno, str):
        try:
            lineno = int(lineno)
            issue["line"] = lineno
        except (ValueError, TypeError):
            return None

    original = issue.get("original", "").strip()
    suggestion = issue.get("suggestion", "").strip()

    if not isinstance(lineno, int):
        return None

    idx = lineno - 1
    if not (0 <= idx < len(source_lines)):
        return None

    source_line = source_lines[idx].strip()
    if original not in source_lines[idx] and original != source_line:
        found = False
        for offset in [-1, 1]:
            adj = idx + offset
            if 0 <= adj < len(source_lines):
                if original in source_lines[adj] or original == source_lines[adj].strip():
                    issue["line"] = adj + 1
                    found = True
                    break
        if not found:
            return None

    if not suggestion or suggestion == original:
        return None

    if "\n" in original or "\n" in suggestion:
        return None

    if canonical == "Väärä kysymysmuoto" and "?" not in original:
        return None

    if canonical == "Epäselvä hän-viittaus":
        if not re.search(r'\bhän(en|tä|elle|llä|ltä|ksi)?\b', original, re.IGNORECASE):
            return None

    if canonical == "Turhat pronominit":
        bare = original.lstrip("- ").strip()
        if not re.match(r'^(Minä|minä|Sinä|sinä)\s+\w', bare):
            issue["auto_apply_blocked"] = True

    issue["rule"] = canonical
    return issue


def check_with_llm(text):
    import time
    source_lines = text.split("\n")
    chunks = chunk_by_blocks(text, max_blocks=5)
    all_issues = []
    seen_keys = set()

    t_start = time.time()

    def process_chunk(args):
        chunk_index, chunk = args
        line_offset = chunk["line_offset"]
        numbered = "\n".join(
            f"{line_offset + j + 1}: {l}"
            for j, l in enumerate(chunk["text"].split("\n"))
        )
        prompt = f"""Tarkista seuraava tekstitysteksti (osa {chunk_index + 1}/{len(chunks)}).
Vastaa VAIN JSON-taulukkona:

{numbered}"""
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 1000},
                "think": False
            }, timeout=300)
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            raw = "".join(c for c in raw if unicodedata.category(c) != 'Cc' or c in '\n\r\t')
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not match:
                return chunk_index, []

            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', match.group())
            parsed = json.loads(cleaned)
            return chunk_index, parsed

        except json.JSONDecodeError:
            return chunk_index, []
        except Exception as e:
            return chunk_index, [{"line": "system", "rule": "LLM-virhe", "original": "—",
                                   "suggestion": "—", "explanation": f"Osa {chunk_index+1}: {str(e)}"}]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        results = dict(executor.map(lambda a: process_chunk(a), enumerate(chunks)))

    for chunk_index in sorted(results.keys()):
        for issue in results[chunk_index]:
            validated = _validate_llm_issue(issue, source_lines)
            if validated is None:
                continue
            key = (validated.get("line"), validated.get("rule"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            validated["chunk"] = f"{chunk_index + 1}/{len(chunks)}"
            all_issues.append(validated)

    elapsed = round(time.time() - t_start, 1)
    print(f"\n⏱  Model: {MODEL} | Chunks: {len(chunks)} | LLM issues: {len(all_issues)} | Time: {elapsed}s")

    return all_issues, elapsed


# ─────────────────────────────────────────────
#  AUTO-APPLY RULE-BASED FIXES
# ─────────────────────────────────────────────

def apply_rule_fixes(text, mechanical_issues):
    lines = text.split("\n")

    RULE_PRIORITY = [
        "Kellonajan muoto",
        "Etunolla kellonajassa",
        "Valuuttamerkki",
        "Prosenttimerkki",
        "Promillemerkki",
        "Tuhansien ryhmittely",
        "Luvun kirjoitusmuoto",
        "Välimerkkiyhdistelmä ?!",
        "Useita huutomerkkejä",
        "Puolipiste dialogissa",
        "Jatkoviivan välilyönti",
        "Vuorosanaviivan välilyönti",
        "Pieni alkukirjain repliikissä",
    ]

    SKIP_RULES = {
        "Mittayksikkö",
        "Tittelin lyhenne",
        "Kirjoitettu lyhenne",
        "Pitkä viiva",
        "Ellipsin ja välimerkin yhdistelmä",
        "Ristiriitainen jatkoviiva",
        "Sulkeet dialogissa",
        "Rivijakauman epätasapaino",
        "Liikaa rivejä repliikissä",
        "Liikaa virkkeitä repliikissä",
        "Liikaa puhujia",
        "Lause liian monessa repliikissä",
        "Ellipsin jatkuminen puuttuu",
        "Lauserajan rikkominen",
        "Lainausmerkit useassa repliikissä",
        "Kaksoispisteen käyttö",
        "Jatkoviivan mahdollinen virhe",
    }

    from collections import defaultdict
    line_fixes_by_rule = defaultdict(list)

    for issue in mechanical_issues:
        rule = issue.get("rule", "")
        lineno = issue.get("line")
        original = issue.get("original", "")
        suggestion = issue.get("suggestion", "")

        if rule in SKIP_RULES:
            continue
        if not isinstance(lineno, int):
            continue
        if suggestion == original or not suggestion:
            continue
        if "\n" in original or "\n" in suggestion:
            continue

        line_fixes_by_rule[lineno].append((rule, original, suggestion))

    for lineno, fixes in line_fixes_by_rule.items():
        idx = lineno - 1
        if not (0 <= idx < len(lines)):
            continue

        def rule_order(fix):
            try:
                return RULE_PRIORITY.index(fix[0])
            except ValueError:
                return len(RULE_PRIORITY)

        fixes_sorted = sorted(fixes, key=rule_order)

        current = lines[idx]
        for rule, original, suggestion in fixes_sorted:
            if original in current:
                current = current.replace(original, suggestion, 1)
            elif original.strip() == current.strip():
                current = suggestion
        lines[idx] = current

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  APPLY LLM FIXES
# ─────────────────────────────────────────────

def apply_llm_fixes(text_after_rules, llm_issues):
    if not llm_issues:
        return text_after_rules

    lines = text_after_rules.split("\n")

    for issue in llm_issues:
        rule = issue.get("rule", "").strip()
        if rule not in LLM_SAFE_RULES:
            continue
        if issue.get("auto_apply_blocked"):
            continue

        lineno = issue.get("line")
        original = issue.get("original", "").strip()
        suggestion = issue.get("suggestion", "").strip()

        if not isinstance(lineno, int):
            continue
        if not suggestion or suggestion == original:
            continue
        if "\n" in original or "\n" in suggestion:
            continue

        idx = lineno - 1
        if not (0 <= idx < len(lines)):
            continue

        current = lines[idx]
        if original in current:
            lines[idx] = current.replace(original, suggestion, 1)
        elif original.strip() == current.strip():
            lines[idx] = suggestion

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/check", methods=["POST"])
def check():
    import time
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    t0 = time.time()
    mechanical = check_mechanical_rules(text)
    linguistic, llm_elapsed = check_with_llm(text)

    for m in mechanical:
        m["source"] = "rule-based"
    for l in linguistic:
        l["source"] = "llm"

    all_issues = mechanical + linguistic
    total_elapsed = round(time.time() - t0, 1)

    print(f"  Total: {total_elapsed}s")

    return jsonify({
        "issues": all_issues,
        "total": len(all_issues),
        "mechanical_count": len(mechanical),
        "llm_count": len(linguistic),
        "model_used": MODEL,
        "timing": {
            "llm_seconds": llm_elapsed,
            "total_seconds": total_elapsed
        }
    })


@app.route("/fix", methods=["POST"])
def fix():
    data = request.get_json()
    text = data.get("text", "").strip()
    issues = data.get("issues", [])
    if not text:
        return jsonify({"error": "No text provided"}), 400

    mechanical_issues = [i for i in issues if i.get("source") == "rule-based"]
    llm_issues = [i for i in issues if i.get("source") == "llm"]

    text_after_rules = apply_rule_fixes(text, mechanical_issues)
    final_text = apply_llm_fixes(text_after_rules, llm_issues)

    return jsonify({
        "fixed_text": final_text,
        "text_after_rules": text_after_rules,
    })


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = int(os.getenv("PORT", 5003))
    print(f"✅ Subtitle Checker running on port {port}")
    print(f"   Model: {MODEL}")
    print(f"   Ollama URL: {OLLAMA_URL}")
    app.run(host="0.0.0.0", debug=False, port=port)
